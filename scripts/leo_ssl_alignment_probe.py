#!/usr/bin/env python3
"""Sonde canonique des 3 compteurs d'alignement — checkpoints SSL Léo × Arctic-TVC.

Lit les embeddings produits par ``scripts/leo_ssl_extract_arctic.py`` (tuile et
contexte sauvegardés SÉPARÉMENT) et calcule, pour chaque checkpoint, trois F1 avec
LA MÊME sonde canonique que le chapitre contexte
(``context_distill._run_probe_with_balanced_acc`` : StandardScaler float32,
``make_canonical_lr`` lbfgs multinomial, grille C ∈ {1e-4..10}, sélection sur val
``f1_macro_pres``, refit best_C ; BLAS mono-thread) :

  - ``tile``  : features tuile seule (768) ;
  - ``ctx``   : contexte seul (768) — le mètre qui sépare le mieux les backbones
                selon leur alignement (SimB-iNat 0.4931 vs DINOv3-LVD 0.4592 @512) ;
  - ``fused`` : [tuile ; contexte] (1536) — la référence gelée (SimB-iNat @512 = 0.5059).

Lecture : une trajectoire en fonction de la dose de SSL aérien. Si l'alignement est
le levier, les trois compteurs montent et le Δ(fused − tile) se contracte quand le
backbone s'aligne.

Sorties (sous --out-dir) :
    probes/<tag>_<variant>.json   un JSON par (checkpoint, variante) — repartable
    alignment_summary.csv         une ligne par (checkpoint, variante)
    alignment_summary.md          tableau lisible + moyennes par époque (3 seeds)

Usage :
    python scripts/leo_ssl_alignment_probe.py --workers 4
    python scripts/leo_ssl_alignment_probe.py --summary-only      # régénère les tables
"""
from __future__ import annotations

# --- Mono-thread BLAS AVANT numpy/sklearn (AGENTS.md §4.8) -----------------------
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]   # grille canonique (base.yaml)
MAX_ITER = 2000
SPLITS = ("train", "val", "test")
VARIANTS = ("tile", "ctx", "fused")
DEFAULT_SIG = PROJ / "results" / "leo_explora_ssl" / "sig_embeddings"
DEFAULT_OUT = PROJ / "results" / "leo_explora_ssl"

_TAG_RE = re.compile(r"^leossl_b16(?:_seed(?P<seed>\d+))?_(?P<rest>.+)$")


def parse_tag(tag: str) -> dict:
    """``leossl_b16_seed0_ep004`` → {seed: 0, epoch_0based: 4} ; ``..._INAT_init`` → seed None."""
    out: dict = {"seed": None, "epoch_0based": None, "kind": tag}
    m = _TAG_RE.match(tag)
    if not m:
        return out
    if m.group("seed") is not None:
        out["seed"] = int(m.group("seed"))
    rest = m.group("rest")
    me = re.match(r"^ep(\d+)$", rest)
    if me:
        out["epoch_0based"] = int(me.group(1))
    elif rest != "INAT_init":
        out["kind"] = rest
    return out


def load_features(sig_dir: Path, tag: str, variant: str) -> dict:
    """Charge les 3 splits et construit la variante demandée."""
    feats = {}
    for s in SPLITS:
        E_t = np.load(sig_dir / tag / f"{s}_tile.npy")
        L = np.load(sig_dir / tag / f"{s}_labels.npy")
        if variant == "tile":
            E = E_t
        else:
            E_c = np.load(sig_dir / tag / f"{s}_ctx.npy")
            E = np.hstack([E_t, E_c]) if variant == "fused" else E_c
        feats[s] = (E, L)
    return feats


def probe_with_proba(feats: dict, C_grid=C_GRID, max_iter=MAX_ITER):
    """COPIÉ de ``context_distill._run_probe_with_balanced_acc`` (même protocole canonique :
    StandardScaler, make_canonical_lr, sélection de C sur val f1_macro_pres, refit) mais
    renvoie EN PLUS les probabilités test — nécessaires pour moyenner les seeds.
    Les métriques retournées sont comparées à celles de la fonction canonique (garde-fou).
    """
    from src.utils import make_canonical_lr
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
    from sklearn.preprocessing import StandardScaler

    from datacurve_one_run import CLASS_NAMES_11, LABELS_8CLS

    E_tr, L_tr = feats["train"]
    E_va, L_va = feats["val"]
    E_te, L_te = feats["test"]
    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr.astype(np.float32))
    X_va = sc.transform(E_va.astype(np.float32))
    X_te = sc.transform(E_te.astype(np.float32))
    best_c, best_f1v = C_grid[0], -1.0
    for c in C_grid:
        clf = make_canonical_lr(C=c, max_iter=max_iter)
        clf.fit(X_tr, L_tr)
        f1v = f1_score(L_va, clf.predict(X_va), average="macro", labels=list(range(11)),
                       zero_division=0)
        if f1v > best_f1v:
            best_c, best_f1v = c, f1v
    clf = make_canonical_lr(C=best_c, max_iter=max_iter)
    clf.fit(X_tr, L_tr)
    P_va = clf.predict_proba(X_va)
    P_te = clf.predict_proba(X_te)
    pred_va, pred_te = clf.classes_[P_va.argmax(1)], clf.classes_[P_te.argmax(1)]
    pres_va, pres_te = sorted({int(v) for v in L_va}), sorted({int(v) for v in L_te})
    f1_all = f1_score(L_te, pred_te, average=None, zero_division=0, labels=list(range(11)))
    metrics = {
        "best_C": float(best_c),
        "f1_macro_pres_val": float(f1_score(L_va, pred_va, average="macro", labels=pres_va,
                                            zero_division=0)),
        "f1_macro_pres_test": float(f1_score(L_te, pred_te, average="macro", labels=pres_te,
                                             zero_division=0)),
        "f1_macro_8cls_test": float(f1_score(L_te, pred_te, average="macro",
                                             labels=LABELS_8CLS, zero_division=0)),
        "accuracy_test": float(accuracy_score(L_te, pred_te)),
        "balanced_accuracy_val": float(balanced_accuracy_score(L_va, pred_va)),
        "balanced_accuracy_test": float(balanced_accuracy_score(L_te, pred_te)),
        "f1_per_class_test": {CLASS_NAMES_11[i]: float(f1_all[i]) for i in range(11)},
    }
    return metrics, P_te, clf.classes_, L_te


def metrics_from_proba(P: np.ndarray, classes: np.ndarray, L_te: np.ndarray) -> dict:
    """Mêmes métriques que la sonde canonique, à partir de probabilités (pour l'ensemble)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from datacurve_one_run import CLASS_NAMES_11, LABELS_8CLS
    pred = classes[P.argmax(1)]
    pres = sorted({int(v) for v in L_te})
    f1_all = f1_score(L_te, pred, average=None, zero_division=0, labels=list(range(11)))
    return {"f1_macro_pres_test": float(f1_score(L_te, pred, average="macro", labels=pres,
                                                 zero_division=0)),
            "f1_macro_8cls_test": float(f1_score(L_te, pred, average="macro",
                                                 labels=LABELS_8CLS, zero_division=0)),
            "accuracy_test": float(accuracy_score(L_te, pred)),
            "balanced_accuracy_test": float(balanced_accuracy_score(L_te, pred)),
            "f1_per_class_test": {CLASS_NAMES_11[i]: float(f1_all[i]) for i in range(11)}}


def run_one(task: tuple) -> tuple:
    """Sonde canonique pour (tag, variante). Renvoie (out_path, message)."""
    tag, variant, sig_dir, out_dir, want_proba = task
    probe_dir = Path(out_dir) / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    out_path = probe_dir / f"{tag}_{variant}.json"
    proba_path = probe_dir / f"{tag}_{variant}_proba.npy"
    # --ensemble : il faut AUSSI les probabilités. Si un premier passage sans --ensemble a
    # déjà écrit le JSON, on ne saute pas : on refait pour produire les proba.
    if out_path.exists() and not (want_proba and not proba_path.exists()):
        return str(out_path), "skip (déjà fait)"

    from context_distill import _run_probe_with_balanced_acc

    t0 = time.time()
    feats = load_features(Path(sig_dir), tag, variant)
    E_tr, L_tr = feats["train"]
    metrics = _run_probe_with_balanced_acc(
        {"val": feats["val"], "test": feats["test"]}, (E_tr, L_tr), C_GRID, MAX_ITER)
    dim = int(E_tr.shape[1])

    # probabilités test (optionnel) : moyenne des seeds + garde-fou de cohérence avec
    # la fonction canonique (les deux calculs DOIVENT donner les mêmes métriques).
    proba_file = None
    if want_proba:
        m2, P_te, classes, L_te = probe_with_proba(feats)
        for k in ("f1_macro_pres_test", "accuracy_test", "best_C"):
            if abs(float(m2[k]) - float(metrics[k])) > 1e-9:
                raise RuntimeError(f"[ensemble] divergence sonde canonique vs locale sur {k} "
                                   f"({m2[k]} vs {metrics[k]}) — protocole modifié ?")
        proba_file = probe_dir / f"{tag}_{variant}_proba.npy"
        np.save(proba_file, P_te.astype(np.float16))
        np.save(probe_dir / f"{tag}_{variant}_classes.npy", classes)
        np.save(probe_dir / f"{tag}_{variant}_labels.npy", L_te)
    del feats, E_tr

    meta_path = Path(sig_dir) / tag / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    ssl_cfg = meta.get("ssl_config") or {}
    parsed = parse_tag(tag)
    epoch0 = parsed["epoch_0based"]
    result = {
        "tag": tag,
        "variant": variant,
        "seed": parsed["seed"],
        "epoch_0based": epoch0,
        "ssl_epochs_done": None if epoch0 is None else epoch0 + 1,
        "ssl_total_epochs": ssl_cfg.get("epochs"),
        "dim": dim,
        "ckpt_name": meta.get("ckpt_name"),
        "position": ("baseline (backbone SimDINOv2-B iNat d'origine)" if tag.endswith("INAT_init")
                     else "checkpoint SSL ExPLoRA-Léo"),
        "schema": "11cls_no_rhol",
        "csv_dir": meta.get("csv_dir"),
        "protocol": ("StandardScaler float32 + make_canonical_lr lbfgs multinomial, "
                     f"C grille {C_GRID}, sélection val f1_macro_pres, refit best_C ; "
                     "BLAS mono-thread ; parallélisation par processus ; features frozen"),
        "proba_file": None if proba_file is None else str(proba_file),
        "runtime_s": round(time.time() - t0, 1),
        **metrics,
    }
    out_path.write_text(json.dumps(result, indent=2))
    return (str(out_path),
            f"f1_test={metrics['f1_macro_pres_test']:.4f} best_C={metrics['best_C']} "
            f"({result['runtime_s']}s)")


def ensemble_by_epoch(out_dir: Path, tags: list[str], variants) -> dict:
    """Moyenne des probabilités des seeds partageant le même epoch (diagnostic §4.4).

    ⚠️ L'estimateur canonique du mémoire reste ``mean±std`` sur les seeds ; cette
    moyenne de probas sert à mesurer si les seeds sont COMPLÉMENTAIRES (diversité)
    ou de simples copies bruitées. Elle ne doit pas être citée comme F1 de modèle.
    """
    probe_dir = out_dir / "probes"
    groups: dict[tuple, list[str]] = {}
    for tag in tags:
        ep = parse_tag(tag)["epoch_0based"]
        if ep is None:
            continue
        for v in variants:
            if (probe_dir / f"{tag}_{v}_proba.npy").exists():
                groups.setdefault((ep, v), []).append(tag)
    out: dict = {}
    for (ep, v), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        Ps, classes, L_ref = [], None, None
        for t in members:
            Ps.append(np.load(probe_dir / f"{t}_{v}_proba.npy").astype(np.float32))
            classes = np.load(probe_dir / f"{t}_{v}_classes.npy")
            L = np.load(probe_dir / f"{t}_{v}_labels.npy")
            if L_ref is None:
                L_ref = L
            elif not np.array_equal(L, L_ref):
                raise RuntimeError(f"[ensemble] labels test différents entre {members}")
        key = f"ep{ep:03d}_{v}"
        out[key] = {"epoch_0based": ep, "variant": v, "members": sorted(members),
                    "n_members": len(members),
                    **metrics_from_proba(np.mean(Ps, axis=0), classes, L_ref)}
    return out


# --------------------------------------------------------------------- tables
def _rows_from_probes(out_dir: Path, tags: list[str]) -> list[dict]:
    rows = []
    for tag in tags:
        for v in VARIANTS:
            p = Path(out_dir) / "probes" / f"{tag}_{v}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rows.append({
                "tag": tag,
                "seed": d.get("seed"),
                "epoch_0based": d.get("epoch_0based"),
                "ssl_epochs_done": d.get("ssl_epochs_done"),
                "variant": v,
                "f1_test": d.get("f1_macro_pres_test"),
                "f1_val": d.get("f1_macro_pres_val"),
                "f1_8cls_test": d.get("f1_macro_8cls_test"),
                "bacc_test": d.get("balanced_accuracy_test"),
                "best_C": d.get("best_C"),
                "dim": d.get("dim"),
            })
    rows.sort(key=lambda r: (r["epoch_0based"] is None, r["epoch_0based"] or 0,
                             r["seed"] if r["seed"] is not None else -1, r["variant"]))
    return rows


def write_tables(out_dir: Path, rows: list[dict], ensemble: dict | None = None) -> None:
    import csv

    csv_path = Path(out_dir) / "alignment_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# Dose-réponse d'alignement — SSL Léo (ExPLoRA, SimDINOv2-B) → Arctic-TVC", ""]
    baseline = [r for r in rows if r["epoch_0based"] is None]
    if baseline:
        b = {r["variant"]: r for r in baseline}
        lines += ["## Point époque 0 — SimDINOv2-B iNat-Plantae (aucun SSL aérien)", "",
                  "| variante | F1 test | F1 8cls | bacc | best_C |", "|---|---|---|---|---|"]
        for v in VARIANTS:
            if v in b:
                r = b[v]
                lines.append(f"| {v} | {r['f1_test']:.4f} | {r['f1_8cls_test']:.4f} | "
                             f"{r['bacc_test']:.4f} | {r['best_C']} |")
        lines.append("")

    # moyennes par époque (3 seeds)
    by_epoch: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        if r["epoch_0based"] is None:
            continue
        by_epoch.setdefault(r["epoch_0based"], {}).setdefault(r["variant"], []).append(r["f1_test"])
    if by_epoch:
        lines += ["## Trajectoire SSL (moyenne ± écart-type sur les seeds)", "",
                  "| epoch (0-based) | époques faites | tile | ctx | fused | Δctx |",
                  "|---|---|---|---|---|---|"]
        for ep in sorted(by_epoch):
            d = by_epoch[ep]
            n = max(len(v) for v in d.values())

            def ms(vals):
                if not vals:
                    return "—"
                a = np.asarray(vals, dtype=float)
                return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}" if len(a) > 1 else f"{a[0]:.4f}"

            tile = d.get("tile", [])
            fused = d.get("fused", [])
            dctx = ("—" if not (tile and fused)
                    else f"{np.mean(fused) - np.mean(tile):+.4f}")
            lines.append(f"| {ep} | {ep + 1} | {ms(d.get('tile', []))} | {ms(d.get('ctx', []))} | "
                         f"{ms(d.get('fused', []))} | {dctx} |")
        lines.append("")

    if ensemble:
        lines += ["## Ensemble des seeds (moyenne des probas) — DIAGNOSTIC, non canonique", "",
                  "L'estimateur canonique est `mean±std` sur les seeds (AGENTS.md §4.4) ; cette "
                  "moyenne de probas ne doit pas être citée comme F1 de modèle. Elle répond à "
                  "une autre question : les seeds sont-ils complémentaires ou redondants ?", "",
                  "| epoch | variante | membres | F1 test | F1 8cls | bacc |",
                  "|---|---|---|---|---|---|"]
        for k, d in ensemble.items():
            lines.append(f"| {d['epoch_0based']} | {d['variant']} | {d['n_members']} | "
                         f"{d['f1_macro_pres_test']:.4f} | {d['f1_macro_8cls_test']:.4f} | "
                         f"{d['balanced_accuracy_test']:.4f} |")
        lines.append("")

    lines += ["## Rappel des références du chapitre contexte (spatial v3, 3 seeds)", "",
              "| Représentation | F1 test |", "|---|---|",
              "| SimB-iNat gelé-fusionné @512 (aucun entraînement) | **0.5059** |",
              "| SimB-iNat gelé, tuile seule @512 | 0.4717 |",
              "| SimB-iNat gelé, contexte seul @512 | 0.4931 |",
              "| SimB Design B entraîné (r2a4) | 0.5030 ± 0.0012 |",
              "| DINOv3-B LVD gelé-fusionné @512 | 0.4953 |", "",
              "Écart-type inter-seed ≈ 0.008 (AGENTS.md §4.4) : aucune conclusion sous "
              "~0.01 sans IC bootstrap apparié.", ""]
    (Path(out_dir) / "alignment_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sig-dir", default=str(DEFAULT_SIG))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tags", default=None, help="sous-chaîne de filtre (défaut : tous)")
    ap.add_argument("--variants", default="tile,ctx,fused")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ensemble", action="store_true",
                    help="moyenne des probas des seeds par époque (diagnostic, cf. §4.4) — "
                         "ajoute des sondes refaites avec sauvegarde des probabilités")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    global VARIANTS
    VARIANTS = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    sig_dir = Path(args.sig_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sig_dir.is_dir():
        raise SystemExit(f"[probe] {sig_dir} introuvable — rsync les sig_embeddings de Narval d'abord")

    tags = sorted(d.name for d in sig_dir.iterdir() if d.is_dir())
    if args.tags:
        tags = [t for t in tags if args.tags in t]
    if not tags:
        raise SystemExit(f"[probe] aucun tag dans {sig_dir}")
    print(f"[probe] {len(tags)} checkpoint(s) × {len(VARIANTS)} variante(s) = "
          f"{len(tags) * len(VARIANTS)} sondes (workers={args.workers}, mono-thread BLAS)", flush=True)

    if not args.summary_only:
        tasks = [(t, v, str(sig_dir), str(out_dir), args.ensemble) for t in tags for v in VARIANTS]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, t): t for t in tasks}
            for fut in as_completed(futs):
                tag, variant = futs[fut][0], futs[fut][1]
                try:
                    _path, msg = fut.result()
                    print(f"  [{tag}/{variant}] {msg}", flush=True)
                except Exception as e:  # noqa: BLE001 — journaliser et continuer
                    print(f"  [ERREUR {tag}/{variant}] {type(e).__name__}: {e}", flush=True)

    ensemble = None
    if args.ensemble:
        ensemble = ensemble_by_epoch(out_dir, tags, VARIANTS)
        (out_dir / "probe_ensembles.json").write_text(json.dumps(ensemble, indent=2))
        print(f"[probe] ensemble : {len(ensemble)} époque(s)×variante(s) → "
              f"{out_dir / 'probe_ensembles.json'}", flush=True)
    elif (out_dir / "probe_ensembles.json").exists():
        ensemble = json.loads((out_dir / "probe_ensembles.json").read_text())

    rows = _rows_from_probes(out_dir, tags)
    if not rows:
        raise SystemExit("[probe] aucune sonde trouvée — rien à résumer")
    write_tables(out_dir, rows, ensemble)
    print(f"[probe] {len(rows)} lignes → {out_dir / 'alignment_summary.csv'} "
          f"+ alignment_summary.md", flush=True)


if __name__ == "__main__":
    main()
