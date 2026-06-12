#!/usr/bin/env python3
"""Courbes d'efficacité-échantillon : f1_macro_pres (test) vs nombre de labels/classe.

Pour chaque modèle gelé, on entraîne le linear probe sur un sous-échantillon stratifié
du train (cap de N exemples PAR classe), C re-griddé sur la validation complète à chaque
taille, et on évalue sur le test complet. Plusieurs graines pour la moyenne ± écart-type.

Message pratique : « combien de labels chaque FM gelé exige pour atteindre son plafond ».

Réutilise ``src.probe.linear_probe`` (même protocole : StandardScaler train, lbfgs, sélection
C sur val). Nécessite les embeddings train/val/test des modèles (Colab/GPU pour les extraire).

Usage :
  python scripts/sample_efficiency.py \\
      --config configs/frozen_eval.yaml \\
      --caps 5 10 20 50 100 200 500 1000 \\
      --seeds 3 \\
      --output results/sample_efficiency.json \\
      --fig results/figures/sample_efficiency.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import load_features
from src.probe import linear_probe
from src.utils import CLASS_NAMES, N_CLASSES

ALL_MODELS = ["resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
              "simdinov2_vitb16", "dinov3_vitl16_sat", "dinov3_vitl16_lvd",
              "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16"]


def _stratified_cap(L: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Indices d'un sous-échantillon avec au plus ``cap`` exemples par classe."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(L):
        ci = np.where(L == c)[0]
        if ci.shape[0] > cap:
            ci = rng.choice(ci, cap, replace=False)
        idx.append(ci)
    out = np.concatenate(idx)
    rng.shuffle(out)
    return out


def run_model(cfg, key: str, caps: list[int], seeds: int) -> dict:
    feats = load_features(cfg, key)
    Etr, Ltr = feats["train"]
    Etr = np.asarray(Etr, np.float32)
    Ltr = np.asarray(Ltr)
    C_grid = tuple(cfg.probe.C_grid)
    curve = {}
    for cap in caps:
        vals = []
        for s in range(seeds):
            idx = _stratified_cap(Ltr, cap, seed=cfg.train.seed + s)
            sub = {"train": (Etr[idx], Ltr[idx]),
                   "val": feats["val"], "test": feats["test"]}
            res = linear_probe(sub, N_CLASSES, CLASS_NAMES, C_grid,
                               cfg.probe.max_iter, cfg.train.seed,
                               cfg.probe.selection_metric)
            vals.append(res["test"]["f1_macro_pres"])
        vals = np.asarray(vals, float)
        curve[str(cap)] = {
            "n_per_class": cap,
            "n_train_used": int(_stratified_cap(Ltr, cap, cfg.train.seed).shape[0]),
            "f1_macro_pres_mean": float(vals.mean()),
            "f1_macro_pres_std": float(vals.std(ddof=1)) if seeds > 1 else 0.0,
            "seeds": seeds,
        }
        print(f"[sample-eff] {key:22} cap={cap:>5}/cl  "
              f"F1={vals.mean():.4f}±{vals.std(ddof=1) if seeds>1 else 0:.4f}")
    return curve


def make_figure(results: dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, curve in results.items():
        caps = sorted((int(k) for k in curve), key=int)
        x = [curve[str(c)]["n_per_class"] for c in caps]
        y = [curve[str(c)]["f1_macro_pres_mean"] for c in caps]
        e = [curve[str(c)]["f1_macro_pres_std"] for c in caps]
        ax.errorbar(x, y, yerr=e, marker="o", capsize=2, label=key, linewidth=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("Nombre de labels par classe (train)", fontsize=10)
    ax.set_ylabel("f1_macro_pres (test)", fontsize=10)
    ax.set_title("Efficacité-échantillon des foundation models gelés", fontsize=11,
                 fontweight="bold")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--caps", nargs="*", type=int,
                    default=[5, 10, 20, 50, 100, 200, 500, 1000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--output", default="results/sample_efficiency.json")
    ap.add_argument("--fig", default="results/figures/sample_efficiency.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or ALL_MODELS
    emb_dir = cfg.paths.emb_dir

    results = {}
    for key in models:
        if not os.path.exists(os.path.join(emb_dir, f"{key}_train.npy")):
            print(f"[skip] {key}: embeddings train absents")
            continue
        results[key] = run_model(cfg, key, args.caps, args.seeds)

    out = {
        "caps": args.caps,
        "seeds": args.seeds,
        "metric": "f1_macro_pres",
        "selection_metric": cfg.probe.selection_metric,
        "C_grid": list(cfg.probe.C_grid),
        "models": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output}  ({len(results)} modèles)")
    if results:
        make_figure(results, args.fig)


if __name__ == "__main__":
    main()
