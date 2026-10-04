#!/usr/bin/env python3
"""Trajectoire du SSL Léo sur les annotations — TOUS les checkpoints, pas seulement `last`.

Question : le SSL aérien n'améliore pas SimDINOv2-B iNat gelé quand on prend `last.pth`
(résultat de `bench_annot.md`). Mais `leo-explora-ssl/README.md` dit explicitement
« on choisit l'époque sur le probe — **pas `last` par défaut** ». Ce script trace donc la
trajectoire complète `ep004 … ep049` des 3 seeds :

  1. **multiclass** (5 espèces, folds spatiaux appariés) — logreg et/ou kNN ;
  2. **détection** par espèce (AUPRC) — mêmes folds, mêmes négatifs ;
  3. **ensemble** par époque (moyenne des probas des 3 seeds) vs `mean±std` des seeds ;
  4. **géométrie** de l'espace latent (RankMe, anisotropie, α-ReQ, NESum) — teste
     l'hypothèse « l'adaptation aérienne tasse l'espace, d'où le kNN qui décroche
     alors que le logreg tient » (cf. src/latent.py).

Le point de référence `base` (SimDINOv2-B iNat gelé) est tracé comme ligne horizontale.

Pré-requis : avoir extrait tous les checkpoints avec
    ALL_EPOCHS=1 sbatch scripts/slurm_leo_ssl_annot_eval.sh
(dossiers `<emb-dir>/ssl_seed{S}_ep{EEE}/bank.npz`).

Sorties :
    <out-dir>/trajectory.json, trajectory.md, trajectory.png/pdf

Usage :
    python scripts/leo_ssl_annot_trajectory.py --clf logreg,knn
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
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import leo_ssl_annot_bench as B          # noqa: E402 — on réutilise TOUT le protocole
from src import latent as L              # noqa: E402

TAG_RE = re.compile(r"^ssl_seed(?P<seed>\d+)_(?P<ep>ep\d+)$")


def parse_tag(tag: str):
    m = TAG_RE.match(tag)
    return (None, None) if not m else (int(m.group("seed")), m.group("ep"))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-dir", default=str(B.DEFAULT_EMB))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--clf", default="logreg", help="logreg | knn (liste possible)")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--blocks", type=float, default=B.BLOCK_M)
    ap.add_argument("--n-splits", type=int, default=B.N_SPLITS)
    ap.add_argument("--task", default="both", choices=["multiclass", "detection", "both"])
    ap.add_argument("--no-geometry", action="store_true")
    args = ap.parse_args()

    B.BLOCK_M, B.N_SPLITS = args.blocks, args.n_splits
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir) if args.out_dir else emb_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tags = B.discover_tags(emb_dir, None)
    ssl_tags = [t for t in all_tags if parse_tag(t)[0] is not None]
    if not ssl_tags:
        raise SystemExit(
            f"[traj] aucun tag 'ssl_seed{{S}}_ep{{EEE}}' dans {emb_dir}.\n"
            "        Lancer d'abord :  ALL_EPOCHS=1 sbatch scripts/slurm_leo_ssl_annot_eval.sh")
    epochs = sorted({parse_tag(t)[1] for t in ssl_tags})
    seeds = sorted({parse_tag(t)[0] for t in ssl_tags})
    tags = (["base"] if "base" in all_tags else []) + ssl_tags
    log(f"{len(ssl_tags)} checkpoints SSL ({len(seeds)} seeds × {len(epochs)} époques) "
        f"+ base={'oui' if 'base' in all_tags else 'non'}")

    banks = {t: B.load_bank(emb_dir, t) for t in tags}
    B.assert_aligned(banks)
    no_ctx = [t for t in tags if not banks[t]["has_ctx"]]
    if no_ctx:
        log(f"[info] pas de contexte 512 pour {len(no_ctx)} tags → variante 'tile' uniquement")
    negs = {t: B.load_negatives(emb_dir, t) for t in tags}
    if any(v is not None for v in negs.values()):
        ref_neg = next(t for t in tags if negs[t] is not None)
        B.assert_same_negatives({k: v for k, v in negs.items() if v is not None}, ref_neg)
    if args.task in ("detection", "both") and negs.get(tags[0]) is None:
        log("[traj] pas de negatives.npz → détection ignorée")
        args.task = "multiclass"

    ref = banks[tags[0]]
    keep = np.isin(ref["species"], B.SPECIES)
    y = np.array([B.SPECIES.index(s) for s in ref["species"][keep]])
    classes = np.arange(len(B.SPECIES))
    folds_mc = B.make_folds(y, B.spatial_groups(ref["x"][keep], ref["y"][keep]))
    base_tag = "base" if "base" in banks else None
    clf_names = [c.strip() for c in args.clf.split(",")]

    result: dict = {
        "n_points": int(keep.sum()), "seeds": seeds, "epochs": epochs,
        "blocks_m": B.BLOCK_M, "n_splits": B.N_SPLITS,
        "protocol": ("réutilise le protocole de leo_ssl_annot_bench (folds spatiaux "
                     "appariés, L2+StandardScaler/cosine) ; mean±std = canonique, "
                     "ensemble = diagnostic (AGENTS.md §4.4)"),
        "multiclass": {}, "detection": {}, "geometry": {},
    }

    # ---------------- géométrie ----------------
    if not args.no_geometry:
        t0 = time.time()
        for t in tags:
            Z = banks[t]["tile"]
            result["geometry"][t] = {
                "seed": parse_tag(t)[0], "epoch": parse_tag(t)[1],
                **L.metrics_with_spectrum(Z, Z, subsample_n=min(20000, Z.shape[0])),
            }
        log(f"géométrie : {len(tags)} tags ({time.time() - t0:.0f}s)")

    # ---------------- multiclass ----------------
    if args.task in ("multiclass", "both"):
        for clf in clf_names:
            t0 = time.time()
            kw = {"k": args.k} if clf == "knn" else {"C": args.C}
            P = {t: B.oof_proba(B.pick(banks[t]["tile"][keep], banks[t]["ctx"][keep], "tile"),
                                y, folds_mc, classes, clf, **kw) for t in tags}
            per_ep = {}
            for ep in epochs:
                members = [t for t in ssl_tags if parse_tag(t)[1] == ep]
                accs = [B.metrics_multiclass(P[t], y, classes)["accuracy"] for t in members]
                f1s = [B.metrics_multiclass(P[t], y, classes)["f1_macro"] for t in members]
                P_ens = np.mean([P[t] for t in members], axis=0)
                per_ep[ep] = {
                    "n_seeds": len(members),
                    "per_seed": {t: B.metrics_multiclass(P[t], y, classes) for t in members},
                    "mean_std_accuracy": B.agg_mean_std(accs),
                    "mean_std_f1_macro": B.agg_mean_std(f1s),
                    "ensemble": {"members": sorted(members),
                                 **B.metrics_multiclass(P_ens, y, classes)},
                }
            result["multiclass"][clf] = {
                "base": (B.metrics_multiclass(P[base_tag], y, classes) if base_tag else None),
                "per_epoch": per_ep,
            }
            log(f"multiclass {clf} fait ({time.time() - t0:.0f}s)")

    # ---------------- détection ----------------
    if args.task in ("detection", "both"):
        for clf in clf_names:
            t0 = time.time()
            kw = {"k": args.k} if clf == "knn" else {"C": args.C}
            per_sp_epoch: dict = {}
            for sp in B.SPECIES:
                if (ref["species"][keep] == sp).sum() < 5:
                    continue
                # folds construits sur le tag de référence (négatifs appariés vérifiés plus haut)
                _, yv, xs, ys = B.detection_matrix(ref, negs[tags[0]], sp, "tile")
                folds = B.make_folds(yv, B.spatial_groups(xs, ys))
                bcls = np.array([0, 1])
                scores = {}
                for t in tags:
                    if negs.get(t) is None:
                        continue
                    X = B.detection_matrix(banks[t], negs[t], sp, "tile")[0]
                    scores[t] = B.oof_proba(X, yv, folds, bcls, clf, **kw)[:, 1]
                per_ep = {}
                for ep in epochs:
                    members = [t for t in ssl_tags if parse_tag(t)[1] == ep and t in scores]
                    if not members:
                        continue
                    aps = [B.metrics_detection(scores[t], yv)["AUPRC"] for t in members]
                    per_ep[ep] = {
                        "n_seeds": len(members),
                        "per_seed": {t: B.metrics_detection(scores[t], yv) for t in members},
                        "mean_std_AUPRC": B.agg_mean_std(aps),
                        "ensemble": {"members": sorted(members), **B.metrics_detection(
                            np.mean([scores[t] for t in members], axis=0), yv)},
                    }
                per_sp_epoch[sp] = {
                    "n_pos": int(yv.sum()), "n_neg": int((yv == 0).sum()),
                    "base": (B.metrics_detection(scores[base_tag], yv) if base_tag in scores
                             else None),
                    "per_epoch": per_ep,
                }
            result["detection"][clf] = per_sp_epoch
            log(f"détection {clf} faite ({time.time() - t0:.0f}s)")

    (out_dir / "trajectory.json").write_text(json.dumps(result, indent=2, default=str))
    write_md(out_dir, result)
    log(f"→ {out_dir / 'trajectory.json'} + trajectory.md")


# ------------------------------------------------------------------ rapport
def write_md(out_dir: Path, r: dict) -> None:
    L_ = ["# Trajectoire du SSL Léo sur les annotations (tous les checkpoints)", "",
          f"{r['n_points']} points, {len(r['seeds'])} seeds × {len(r['epochs'])} époques, "
          f"folds spatiaux appariés ({r['n_splits']} plis, blocs {r['blocks_m']} m).", "",
          "> `mean±std` sur les seeds = estimateur canonique (AGENTS.md §4.4) ; l'ensemble "
          "(moyenne des probas) est un **diagnostic**, pas un chiffre de manuscrit.", ""]
    for clf, d in r.get("multiclass", {}).items():
        b = d.get("base")
        L_ += [f"## Multiclass — {clf}", "",
               f"Référence gelée `base` : accuracy **{b['accuracy']:.4f}**, "
               f"F1-macro **{b['f1_macro']:.4f}**" if b else "_(pas de base)_", "",
               "| époque | seeds | acc (mean±std) | F1 (mean±std) | F1 ensemble | Δ F1 ens−base |",
               "|---|---|---|---|---|---|"]
        for ep, e in d["per_epoch"].items():
            ms, mf = e["mean_std_accuracy"], e["mean_std_f1_macro"]
            e_f1 = e["ensemble"]["f1_macro"]
            dlt = (f"{e_f1 - b['f1_macro']:+.4f}" if b else "—")
            L_.append(f"| {ep} | {e['n_seeds']} | {ms['mean']:.4f} ± {ms['std']:.4f} | "
                      f"{mf['mean']:.4f} ± {mf['std']:.4f} | **{e_f1:.4f}** | {dlt} |")
        L_.append("")
    for clf, per_sp in r.get("detection", {}).items():
        L_ += [f"## Détection (AUPRC) — {clf}", ""]
        for sp, d in per_sp.items():
            b = d.get("base")
            L_ += [f"### {sp} (n_pos={d['n_pos']})", "",
                   f"Référence gelée `base` : **{b['AUPRC']:.4f}**" if b else "_(pas de base)_",
                   "", "| époque | seeds | AUPRC (mean±std) | AUPRC ensemble | Δ ens−base |",
                   "|---|---|---|---|---|"]
            for ep, e in d["per_epoch"].items():
                ms = e["mean_std_AUPRC"]
                a = e["ensemble"]["AUPRC"]
                delta = f"{a - b['AUPRC']:+.4f}" if b else "—"
                L_.append(f"| {ep} | {e['n_seeds']} | {ms['mean']:.4f} ± {ms['std']:.4f} | "
                          f"**{a:.4f}** | {delta} |")
            L_.append("")
    g = r.get("geometry")
    if g:
        L_ += ["## Géométrie de l'espace latent (bank, tuile 768)", "",
              "| tag | RankMe | anisotropie | α-ReQ | NESum |", "|---|---|---|---|---|"]
        for t, v in g.items():
            L_.append(f"| {t} | {v['rankme']:.1f} | {v['anisotropy']:+.3f} | "
                      f"{v['alpha_req']:.3f} | {v['nesum']:.2f} |")
        L_.append("")
    (out_dir / "trajectory.md").write_text("\n".join(L_) + "\n")


if __name__ == "__main__":
    main()
