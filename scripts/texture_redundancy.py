#!/usr/bin/env python3
"""Redondance d'une source de features avec un embedding de modèle de fondation.

── La question à laquelle ce script répond ──────────────────────────────────────────
« J'ai ajouté mes features X (texture, CHM, NIR, indices…) à mon embedding : Δ = 0. X est-il
(a) sans information, (b) informatif mais déjà contenu dans le FM, ou (c) mon pipeline est-il
cassé ? » Un Δ nul ne distingue pas ces trois cas. Ce script les sépare, et c'est ce qui
transforme un résultat négatif en résultat exploitable.

── Les trois mesures ────────────────────────────────────────────────────────────────
1. **F1(X seul)** — les features savent-elles classer sans le FM ? Si oui, elles portent un
   vrai signal ; si c'est ≈ hasard, le jeu de features est dégénéré et le Δ nul ne dit rien
   sur le FM.
2. **R²(X | embedding)** — régression ridge de l'embedding vers les features : quelle part de
   X le FM contient-il déjà ? R² ≈ 1 ⇒ redondance.
3. **F du résidu** — ANOVA par classe sur X − X̂ (la part de X que l'embedding ne prédit
   PAS). Si le résidu a un F plat, alors la part orthogonale au FM est du bruit pour la
   tâche : le Δ nul est *structurel*, pas un artefact.

La décomposition qui rend le résultat interprétable : ``X = (sous-espace du FM) ⊕ (résidu)``.
Si (1) est élevé, (2) élevé et (3) plat, alors ajouter X ne peut rien apporter — quel que
soit l'entraînement en aval. C'est un résultat *a priori*, testable sans entraîner de modèle.

Résultat mesuré sur Arctic-TVC (2026-09-16, job Narval 3155757) : texture 7 familles,
X seul F1 = 0,3348, R² = 0,884–0,916, F résidu médian = 3,3–4,2 contre ~1 000 pour les
directions de l'embedding → texture = sous-espace du FM ⊕ bruit. Voir
``scripts/texture_README.md`` §Résultats.

── Usage ────────────────────────────────────────────────────────────────────────────
    python3 scripts/texture_redundancy.py --emb embeddings/simdinov2_vitb16 \\
        --label-schema 12cls --features-dir results/texture --tag simdinov2_vitb16

Générique : ``--features-dir`` doit contenir ``{train,val,test}.npy`` + ``{split}.json``
(clé ``feature_names``), c'est-à-dire la même convention que la sortie de
``scripts/texture_features.py`` — donc n'importe quelle future source d'information
(CHM, NIR, indices spectraux) peut être testée en redondance AVANT toute campagne
d'acquisition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── MONO-THREAD OBLIGATOIRE — AVANT tout import de numpy/sklearn (AGENTS.md §4.8) ────
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from sklearn.feature_selection import f_classif  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
# texture_ablation.py vit dans scripts/ (pas dans le package src/) : il faut les deux
# chemins pour importer src.* ET scripts/*.
for _p in (str(PROJ), str(PROJ / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from texture_ablation import load_embeddings, load_texture, probe  # noqa: E402

SPLITS = ("train", "val", "test")


def _r2(X_emb_tr, X_feat_tr, X_emb_va, X_feat_va, n_sub, n_val, seed=0, alpha=1.0):
    """R² par feature de la régression ridge embedding → features (sous-échantillonnée)."""
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    i_tr = rng.choice(len(X_feat_tr), min(n_sub, len(X_feat_tr)), replace=False)
    i_va = rng.choice(len(X_feat_va), min(n_val, len(X_feat_va)), replace=False)
    sc_e, sc_f = StandardScaler(), StandardScaler()
    A = sc_e.fit_transform(X_emb_tr[i_tr])
    B = sc_f.fit_transform(X_feat_tr[i_tr])
    model = Ridge(alpha=alpha).fit(A, B)
    pred = model.predict(sc_e.transform(X_emb_va[i_va]))
    truth = sc_f.transform(X_feat_va[i_va])
    ss_res = ((truth - pred) ** 2).sum(axis=0)
    ss_tot = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    # Résidu ré-étiqueté pour l'ANOVA : on renvoie aussi le résidu standardisé.
    resid = B - model.predict(A)
    return r2, resid, i_tr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True, help="préfixe/dossier d'embeddings (cf. texture_ablation)")
    ap.add_argument("--label-schema", default="11cls", choices=("11cls", "12cls"))
    ap.add_argument("--features-dir", default="results/texture",
                    help="dossier contenant {train,val,test}.npy + .json")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-sub", type=int, default=20000, help="lignes de train pour le ridge")
    ap.add_argument("--n-val", type=int, default=4000, help="lignes de val pour le R²")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--skip-probes", action="store_true",
                    help="ne calculer que la redondance (F1 seul / R² / F résidu)")
    args = ap.parse_args()

    t0 = time.time()
    tag = args.tag or Path(args.emb).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = load_embeddings(args.emb, args.label_schema)
    feat = {s: load_texture(Path(args.features_dir), s)[0] for s in SPLITS}
    names = load_texture(Path(args.features_dir), "train")[1]
    for s in SPLITS:
        if feat[s].shape[0] != emb[s][0].shape[0]:
            raise ValueError(f"désalignement {s} : {feat[s].shape[0]} features "
                             f"pour {emb[s][0].shape[0]} embeddings")

    Etr, ytr = emb["train"]
    Eva, yva = emb["val"]
    Ete, yte = emb["test"]
    ytr = np.asarray(ytr)
    print(f"[redundance] tag={tag}  emb={Etr.shape[1]} dims  features={feat['train'].shape[1]} "
          f"({len(names)} noms)", flush=True)

    res: dict = {"tag": tag, "emb_prefix": args.emb, "label_schema": args.label_schema,
                 "n_dims_emb": int(Etr.shape[1]), "n_features": int(feat["train"].shape[1]),
                 "feature_names": names,
                 "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ── 1. Les features seules savent-elles classer ? ─────────────────────────
    if not args.skip_probes:
        f_feat = probe(feat["train"], ytr, feat["val"], yva, feat["test"], yte,
                       with_8cls_sep=False)
        f_emb = probe(Etr, ytr, Eva, yva, Ete, yte, with_8cls_sep=False)
        f_both = probe(np.hstack([Etr, feat["train"]]), ytr,
                       np.hstack([Eva, feat["val"]]), yva,
                       np.hstack([Ete, feat["test"]]), yte, with_8cls_sep=False)
        res["f1_11cls_features_seules"] = f_feat["f1_macro_pres"]
        res["f1_11cls_embedding_seul"] = f_emb["f1_macro_pres"]
        res["f1_11cls_embedding_plus_features"] = f_both["f1_macro_pres"]
        res["delta_ajout"] = f_both["f1_macro_pres"] - f_emb["f1_macro_pres"]
        res["best_C"] = {"features": f_feat["best_C"], "embedding": f_emb["best_C"],
                         "concat": f_both["best_C"]}
        n_cls = len(set(int(v) for v in yte))
        print(f"  features seules            F1 = {f_feat['f1_macro_pres']:.4f}   "
              f"(hasard ≈ 1/{n_cls} = {1 / n_cls:.3f})", flush=True)
        print(f"  embedding seul             F1 = {f_emb['f1_macro_pres']:.4f}", flush=True)
        print(f"  embedding + features       F1 = {f_both['f1_macro_pres']:.4f}   "
              f"Δ = {res['delta_ajout']:+.4f}", flush=True)

    # ── 2. Redondance : l'embedding prédit-il les features ? ──────────────────
    r2, resid, i_tr = _r2(Etr, feat["train"], Eva, feat["val"],
                          args.n_sub, args.n_val, alpha=args.ridge_alpha)
    res["r2_median"] = float(np.median(r2))
    res["r2_moyen"] = float(r2.mean())
    res["r2_fraction_sup_0.5"] = float((r2 > 0.5).mean())
    res["r2_quintiles"] = [float(x) for x in np.percentile(r2, [10, 25, 50, 75, 90])]
    print(f"  R² (features | embedding)  médian = {np.median(r2):.3f}   "
          f"{(r2 > 0.5).sum()}/{len(r2)} features > 0,5", flush=True)

    # ── 3. Le résidu (part non contenue dans le FM) porte-t-il du signal ? ────
    F_all, _ = f_classif(feat["train"], ytr)
    F_emb, _ = f_classif(Etr, ytr)
    F_res, _ = f_classif(resid, ytr[i_tr])
    res["F_features_median"] = float(np.median(F_all))
    res["F_features_max"] = float(F_all.max())
    res["F_embedding_median"] = float(np.median(F_emb))
    res["F_embedding_max"] = float(F_emb.max())
    res["F_residu_median"] = float(np.median(F_res))
    res["F_residu_max"] = float(F_res.max())
    res["F_residu_gt50"] = int((F_res > 50).sum())
    order = np.argsort(-F_res)[:10]
    res["top_features_residu"] = [{"name": names[i], "F": float(F_res[i])} for i in order]
    print(f"  F médian  features         = {np.median(F_all):9.1f}  (max {F_all.max():.1f})", flush=True)
    print(f"  F médian  embedding        = {np.median(F_emb):9.1f}  (max {F_emb.max():.1f})", flush=True)
    print(f"  F médian  RÉSIDU           = {np.median(F_res):9.2f}  (max {F_res.max():.1f}) "
          f"→ {int((F_res > 50).sum())} feature(s) au-dessus de F=50", flush=True)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict = ("features sans signal — Δ nul non informatif sur le FM"
               if res.get("f1_11cls_features_seules", 1.0) < 2.0 / 11 else
               "REDONDANT : le FM contient déjà l'information, le résidu est plat"
               if res["F_residu_max"] < 50 else
               "NON REDONDANT : du signal de classe survit hors du FM — creuser "
               "(une fusion apprise pourrait l'exploiter)")
    res["verdict"] = verdict
    print(f"\n  VERDICT : {verdict}", flush=True)

    out = Path(out_dir) / f"redundance_{tag}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"[redundance] OK → {out}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
