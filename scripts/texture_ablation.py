#!/usr/bin/env python3
"""Ablation des familles de texture concaténées à l'embedding — sonde canonique + bootstrap apparié.

── Question posée ───────────────────────────────────────────────────────────────────
Est-ce que la texture fine (sous le patch d'un ViT) ajoute de l'information à
l'embedding, famille par famille ? Arctic-TVC est à 2,2 mm/px : un patch de 16 px couvre
3,5 cm, donc la moucheture du lichen, le tapis de mousse et les linaires de linaigrette
sont lissés avant d'entrer dans le réseau.

── Protocole (identique à probe.py / src.probe.linear_probe) ────────────────────────
  - X = hstack([embedding, features de texture de la famille]) ;
  - StandardScaler ajusté sur le TRAIN seul, appliqué à val/test — indispensable : les
    colonnes de texture (contrast 0–500) et l'embedding (moyenne ~0) n'ont pas la même
    échelle, une sonde non standardisée mettrait toute sa masse sur la texture ;
  - LogisticRegression lbfgs multinomial, max_iter=2000, seed 42, BLAS MONO-THREAD ;
  - best_C choisi sur la VALIDATION dans la grille canonique 1e-4 … 10 (une sélection
    PAR jeu de features : la grille reste identique, ce que §4.3 exige) ;
  - Δ vs baseline par BOOTSTRAP APPARIÉ : mêmes indices de tuiles pour les deux modèles
    sur chaque rééchantillonnage, donc le Δ porte sur les mêmes tuiles.

── ⚠️ DEUX DÉFINITIONS DU « 8 CLASSES » COEXISTENT DANS LE DÉPÔT ─────────────────────
Elles ne donnent PAS le même chiffre, et l'écart est systématique (+0,004 à +0,006 sur
les 9 modèles gelés où les deux sont calculables) :

  1. **Sonde séparée** (``f1_macro_8cls_sep``, colonne principale ici) — on retire les
     tuiles ARCA/DRYI/RUBC des trois splits (49281→48473 train, 17598→17277 test), on
     compacte les indices 0..7 et on RÉ-AJUSTE une sonde sur 8 classes. C'est la
     définition canonique (``results/relance2/relance2_8cls.json``,
     ``results/8cls/probe_knn_cgrid.json``, AGENTS.md §3).
  2. **Ré-moyenne** (``f1_macro_8cls_remoy``, diagnostic) — on garde la sonde 11 classes
     et on ne moyenne que le F1 par classe des 8 classes restantes. C'est ce qu'écrivent
     ``scripts/datacurve_one_run.py`` et ``scripts/context_distill.py`` dans le champ
     ``f1_macro_8cls_test`` des ``metrics.json``.

**Ne jamais mélanger les deux dans un même tableau.** Comme la définition 2 est plus basse
de ~0,005 (≈ 60 % de l'écart-type inter-seed de 0,008), un classement qui compare des
lignes issues des deux sources peut inverser des paires proches.

── Sortie ───────────────────────────────────────────────────────────────────────────
``results/texture/ablation_<tag>.json`` : un bloc par jeu de features (baseline, une
famille, combiné) avec best_C, F1 test (11 cls, 8 cls séparé, 8 cls ré-moyenné) et
Δ ± IC95 + p vs baseline.

── Usage ────────────────────────────────────────────────────────────────────────────
    # Modèle gelé (embeddings/<key>_{split}.npy, labels 12 classes → remap 11)
    python3 scripts/texture_ablation.py --emb embeddings/simdinov2_vitb16 \\
        --label-schema 12cls --tag simdinov2_vitb16

    # Run affiné (fichiers sans préfixe modèle, labels déjà 11 classes)
    python3 scripts/texture_ablation.py \\
        --emb DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed0 \\
        --label-schema 11cls --tag dinov3b_lora8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── MONO-THREAD OBLIGATOIRE — AVANT tout import de numpy/sklearn (AGENTS.md §4.8) ────
# Le nombre de threads BLAS change l'ordre de réduction des sommes, donc la solution vers
# laquelle lbfgs converge : DINOv3-B MHSA seed0 vaut 0,4821 à 1 thread (valeur publiée)
# contre 0,4836 à 3 threads, alors que le solveur converge dans les deux cas. L'écart
# (0,0015) est du même ordre que les effets testés ici.
# Le parallélisme, s'il en faut, vient de processus séparés — jamais de BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.texture import FAMILIES, family_of  # noqa: E402
from src.utils import CLASS_NAMES_11, LABEL_REMAP_12TO11, make_canonical_lr  # noqa: E402

SPLITS = ("train", "val", "test")
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # configs/base.yaml — NE PAS réduire
MAX_ITER = 2000
SEED = 42

#: Classes retirées de la sonde 8 classes séparée (cf. AGENTS.md §4.1 et _DROP_8CLS_NAMES).
DROP_8CLS = ("ARCA", "DRYI", "RUBC")
#: Indices 11-class conservés, dans l'ordre → nouveaux labels 0..7.
KEEP_8CLS = [i for i, c in enumerate(CLASS_NAMES_11) if c not in DROP_8CLS]


# ────────────────────────────────────────────────────────────────────── chargement


def load_embeddings(prefix: str, schema: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Charge ``<prefix>_<split>.npy`` + labels, en schéma 11 classes."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in SPLITS:
        ep = f"{prefix}_{s}.npy"
        lp = f"{prefix}_{s}_labels.npy"
        if not (os.path.exists(ep) and os.path.exists(lp)):
            raise FileNotFoundError(f"embeddings introuvables : {ep} / {lp}")
        X = np.load(ep).astype(np.float32)
        y = np.load(lp).astype(np.int64).ravel()
        if schema == "12cls":
            # RHOL (idx 7) absente du split : on la retire et on décale les indices > 7.
            keep = np.array([v != 7 for v in y])
            X, y = X[keep], np.array([LABEL_REMAP_12TO11[int(v)] for v in y[keep]], dtype=np.int64)
        elif schema != "11cls":
            raise ValueError(f"--label-schema doit valoir 11cls ou 12cls (reçu {schema})")
        out[s] = (X, y)
    return out


def load_texture(cache_dir: Path, split: str) -> tuple[np.ndarray, list[str]]:
    """Charge le cache de texture d'un split + vérifie le contrat de colonnes."""
    npy = cache_dir / f"{split}.npy"
    js = cache_dir / f"{split}.json"
    if not (npy.exists() and js.exists()):
        raise FileNotFoundError(
            f"cache de texture absent : {npy}\n"
            f"  → lancer d'abord : python3 scripts/texture_features.py --split {split}")
    meta = json.loads(js.read_text())
    X = np.load(npy).astype(np.float32)
    names = list(meta["feature_names"])
    if X.shape[1] != len(names):
        raise ValueError(f"{npy} : {X.shape[1]} colonnes pour {len(names)} noms déclarés")
    return X, names


def to_8cls(y: np.ndarray) -> np.ndarray:
    """Compacte les labels 11-class sur les 8 classes retenues (0..7).

    L'appelant doit avoir retiré les tuiles des 3 classes (cf. ``np.isin(y, KEEP_8CLS)``) :
    un label hors des 8 retenues lève, plutôt que de produire un décalage silencieux.
    """
    remap = {c: j for j, c in enumerate(KEEP_8CLS)}
    bad = sorted({int(v) for v in y if int(v) not in remap})
    if bad:
        raise ValueError(f"labels hors des 8 classes retenues : {bad} — masquer les tuiles d'abord")
    return np.array([remap[int(v)] for v in y], dtype=np.int64)


# ──────────────────────────────────────────────────────────────────────── sonde


def _fit_select(Ztr, ytr, Zva, yva, c_grid) -> tuple[float, object]:
    """Sélectionne C sur la validation, retourne (best_C, modèle ré-ajusté)."""
    va_labels = sorted({int(v) for v in yva})
    best_c, best = c_grid[0], -1.0
    for c in c_grid:
        clf = make_canonical_lr(C=c, max_iter=MAX_ITER)
        clf.fit(Ztr, ytr)
        f1v = float(f1_score(yva, clf.predict(Zva), average="macro",
                             labels=va_labels, zero_division=0))
        if f1v > best:
            best_c, best = c, f1v
    clf = make_canonical_lr(C=best_c, max_iter=MAX_ITER)
    clf.fit(Ztr, ytr)
    return best_c, clf


def probe(Xtr, ytr, Xva, yva, Xte, yte, c_grid=C_GRID, with_8cls_sep: bool = True) -> dict:
    """Sonde canonique 11 classes, + sonde 8 classes SÉPARÉE (définition canonique).

    Retourne aussi les prédictions (11 classes sur tout le test, 8 classes sur le
    sous-ensemble test) pour le bootstrap apparié.
    """
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr)
    Zva = sc.transform(Xva)
    Zte = sc.transform(Xte)
    best_c, clf = _fit_select(Ztr, ytr, Zva, yva, c_grid)
    pred11 = clf.predict(Zte)
    lab11 = sorted({int(v) for v in yte})
    out = dict(
        best_C=float(best_c),
        f1_macro_pres=float(f1_score(yte, pred11, average="macro", labels=lab11, zero_division=0)),
        # Ré-moyenne : même sonde, on ne garde que les 8 classes qui comptent.
        f1_macro_8cls_remoy=float(f1_score(yte, pred11, average="macro",
                                           labels=KEEP_8CLS, zero_division=0)),
        n_features=int(Xtr.shape[1]),
        _pred11=pred11, _yte=yte,
    )
    if with_8cls_sep:
        # Sonde SÉPARÉE : on retire les tuiles des 3 classes, on compacte et on ré-ajuste.
        m_tr = np.isin(ytr, KEEP_8CLS)
        m_va = np.isin(yva, KEEP_8CLS)
        m_te = np.isin(yte, KEEP_8CLS)
        if not (m_tr.any() and m_va.any() and m_te.any()):
            raise ValueError("aucune tuile des 8 classes retenues — vérifier le schéma de labels")
        sc8 = StandardScaler()
        Ztr8 = sc8.fit_transform(Xtr[m_tr])
        Zva8 = sc8.transform(Xva[m_va])
        Zte8 = sc8.transform(Xte[m_te])
        ytr8, yva8, yte8 = to_8cls(ytr[m_tr]), to_8cls(yva[m_va]), to_8cls(yte[m_te])
        c8, clf8 = _fit_select(Ztr8, ytr8, Zva8, yva8, c_grid)
        pred8 = clf8.predict(Zte8)
        out.update(
            best_C_8cls=float(c8),
            f1_macro_8cls_sep=float(f1_score(yte8, pred8, average="macro",
                                             labels=list(range(8)), zero_division=0)),
            n_train_8cls=int(m_tr.sum()), n_test_8cls=int(m_te.sum()),
            _pred8=pred8, _yte8=yte8)
    else:
        out.update(f1_macro_8cls_sep=None, best_C_8cls=None,
                   n_train_8cls=None, n_test_8cls=None, _pred8=None, _yte8=None)
    return out


def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray,
                     labels: list[int], n_boot: int, seed: int = SEED) -> dict:
    """Δ = F1(b) − F1(a) par bootstrap APPARIÉ sur les tuiles de test.

    Les mêmes indices sont utilisés pour les deux modèles à chaque tirage : le Δ porte
    donc sur les mêmes tuiles, ce qui annule la variance d'échantillonnage commune.
    """
    rng = np.random.RandomState(seed)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    obs_a = float(f1_score(y, pred_a, average="macro", labels=labels, zero_division=0))
    obs_b = float(f1_score(y, pred_b, average="macro", labels=labels, zero_division=0))
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[i] = (f1_score(y[idx], pred_b[idx], average="macro", labels=labels, zero_division=0)
                     - f1_score(y[idx], pred_a[idx], average="macro", labels=labels, zero_division=0))
    p = 2.0 * min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    return dict(delta=obs_b - obs_a, ci95=[float(np.percentile(deltas, 2.5)),
                                           float(np.percentile(deltas, 97.5))],
                p=min(p, 1.0), n_boot=n_boot, f1_a=obs_a, f1_b=obs_b)


def _deltas(ref: dict, r: dict, n_boot: int) -> tuple[dict, dict]:
    lab11 = sorted({int(v) for v in ref["_yte"]})
    d11 = paired_bootstrap(ref["_pred11"], r["_pred11"], ref["_yte"], lab11, n_boot)
    if ref["_pred8"] is None or r["_pred8"] is None:
        return d11, None
    d8 = paired_bootstrap(ref["_pred8"], r["_pred8"], ref["_yte8"], list(range(8)), n_boot)
    return d11, d8


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────── main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", required=True, help="préfixe des embeddings (sans _<split>.npy)")
    ap.add_argument("--label-schema", default="11cls", choices=("11cls", "12cls"))
    ap.add_argument("--texture-dir", default="results/texture", help="cache de texture")
    ap.add_argument("--tag", default=None, help="nom de sortie (défaut : stem de --emb)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--include-combined", action="store_true",
                    help="tester aussi embedding + TOUTES les familles")
    ap.add_argument("--baseline-only", action="store_true",
                    help="ne faire que la baseline (contrôle de non-régression vs registre)")
    ap.add_argument("--no-8cls-sep", action="store_true",
                    help="ne pas ajuster la sonde 8 classes séparée (≈ 2× plus rapide, "
                         "mais on perd la métrique canonique)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    tag = args.tag or Path(args.emb).stem
    out_dir = Path(args.out_dir) if args.out_dir else (PROJ / "results" / "texture")
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = load_embeddings(args.emb, args.label_schema)
    tex, names = load_texture(Path(args.texture_dir), "train")
    tex = {s: load_texture(Path(args.texture_dir), s)[0] for s in SPLITS}
    for s in SPLITS:
        if tex[s].shape[0] != emb[s][0].shape[0]:
            raise ValueError(
                f"désalignement {s} : {tex[s].shape[0]} features de texture pour "
                f"{emb[s][0].shape[0]} embeddings — le cache de texture doit être extrait "
                f"sur le MÊME CSV que celui des embeddings")

    family_cols: dict[str, list[int]] = {f: [] for f in FAMILIES}
    for i, n in enumerate(names):
        family_cols.setdefault(family_of(n), []).append(i)
    wanted = [f.strip() for f in args.families.split(",") if f.strip()]
    missing = [f for f in wanted if f not in family_cols]
    if missing:
        print(f"[ablation] familles absentes du cache : {missing} (dispo : {sorted(family_cols)})",
              file=sys.stderr)
        return 2

    print(f"[ablation] tag={tag}  emb={args.emb}  {emb['train'][0].shape[1]} dims  "
          f"texture={tex['train'].shape[1]} colonnes", flush=True)
    print(f"[ablation] familles testées : {wanted}  |  sonde 8cls séparée : "
          f"{not args.no_8cls_sep}", flush=True)

    Xtr_e, ytr = emb["train"]
    Xva_e, yva = emb["val"]
    Xte_e, yte = emb["test"]

    ref = probe(Xtr_e, ytr, Xva_e, yva, Xte_e, yte, with_8cls_sep=not args.no_8cls_sep)
    print(f"  {'baseline':<26s} 11cls={ref['f1_macro_pres']:.4f}  "
          f"8cls_sep={ref['f1_macro_8cls_sep'] if ref['f1_macro_8cls_sep'] is None else round(ref['f1_macro_8cls_sep'], 4)}  "
          f"8cls_remoy={ref['f1_macro_8cls_remoy']:.4f}  C={ref['best_C']:g}", flush=True)

    results: dict = {
        "tag": tag, "emb_prefix": args.emb, "label_schema": args.label_schema,
        "n_features_emb": int(Xtr_e.shape[1]), "n_features_texture": int(tex["train"].shape[1]),
        "n_tiles": {s: int(emb[s][0].shape[0]) for s in SPLITS},
        "n_boot": args.n_boot, "seed": SEED, "c_grid": list(C_GRID),
        "protocol": "Sonde canonique lbfgs mono-thread, best_C sur val, bootstrap apparié",
        "note_8cls": ("f1_macro_8cls_sep = sonde SÉPARÉE ajustée sur 8 classes, tuiles des "
                      "3 classes retirées (définition canonique, AGENTS.md §3) ; "
                      "f1_macro_8cls_remoy = ré-moyenne du F1 par classe de la sonde 11 classes "
                      "(définition des metrics.json de FT). Écart systématique +0,004 à +0,006 "
                      "— ne jamais mélanger les deux dans un tableau."),
        "runs": {}, "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results["runs"]["baseline"] = _strip(ref)

    if args.baseline_only:
        out = out_dir / f"ablation_{tag}.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"[ablation] baseline seule → {out}  ({time.time() - t0:.0f}s)", flush=True)
        return 0

    def _run(name: str, cols: list[int]) -> None:
        Xtr = np.hstack([Xtr_e, tex["train"][:, cols]])
        Xva = np.hstack([Xva_e, tex["val"][:, cols]])
        Xte = np.hstack([Xte_e, tex["test"][:, cols]])
        r = probe(Xtr, ytr, Xva, yva, Xte, yte, with_8cls_sep=not args.no_8cls_sep)
        d11, d8 = _deltas(ref, r, args.n_boot)
        r = _strip(r)
        r["texture_cols"] = len(cols)
        r["delta_11cls"] = d11
        r["delta_8cls_sep"] = d8
        results["runs"][name] = r
        s8 = r["f1_macro_8cls_sep"]
        print(f"  {name:<26s} 11cls={r['f1_macro_pres']:.4f}  "
              f"8cls_sep={'—' if s8 is None else f'{s8:.4f}'}  "
              f"Δ11={d11['delta']:+.4f} p={d11['p']:.3f}"
              + (f"  Δ8={d8['delta']:+.4f} p={d8['p']:.3f}" if d8 else ""), flush=True)

    for fam in wanted:
        _run(fam, family_cols[fam])
    if args.include_combined:
        _run("combined", [i for f in wanted for i in family_cols[f]])

    out = out_dir / f"ablation_{tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(f"[ablation] OK → {out}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
