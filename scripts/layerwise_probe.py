#!/usr/bin/env python3
"""Linear probe F1 *par couche* + géométrie par couche, et corrélation géométrie ↔ F1.

``analyze.py --layerwise`` ne calcule que la GÉOMÉTRIE par couche (RankMe/anisotropie).
Ici on ajoute le **F1 du linear probe par couche** : pour chaque bloc transformer, on
entraîne la sonde sur la représentation CLS de ce bloc (train) et on évalue sur le test.

Intérêt : transforme la corrélation géométrie↔F1 (n=9 modèles, IC95 énormes) en une
histoire INTRA-modèle à beaucoup plus de points — « en profondeur, le rang effectif monte,
l'anisotropie chute, ET le F1 du probe monte » — plus un nuage poolé (Σ couches × modèles).

Prérequis : embeddings layerwise train/val/test extraits via
``extract.py --layerwise --splits train val test`` pour chaque modèle (ViT uniquement ;
les MAE ne sont pas pertinents en CLS — à exclure).

Usage :
  python scripts/layerwise_probe.py \\
      --config configs/frozen_eval.yaml \\
      --models dinov3_vitl16_lvd dinov3_vitl16_sat \\
      --output results/layerwise_probe.json \\
      --fig results/figures/layerwise_probe_pooled.png
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
from src.features import count_layerwise_layers, load_layerwise
from src.latent import anisotropy, rankme, subsample
from src.probe import linear_probe
from src.utils import CLASS_NAMES, N_CLASSES

DEFAULT_MODELS = ["dinov3_vitl16_lvd", "dinov3_vitl16_sat", "dinov3_vitb16_lvd",
                  "simdinov2_vitl16", "simdinov2_vitb16", "vitb16_imagenet"]


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def _kendall(x, y):
    from scipy.stats import kendalltau
    return float(kendalltau(x, y)[0])


def run_model(cfg, key: str, subsample_n: int = 20000) -> dict:
    n_layers = count_layerwise_layers(cfg, key, "train")
    lw = {s: load_layerwise(cfg, key, s, n_layers) for s in ("train", "val", "test")}
    C_grid = tuple(cfg.probe.C_grid)
    layers = {}
    for li in range(n_layers):
        feats = {"train": lw["train"][li], "val": lw["val"][li], "test": lw["test"][li]}
        res = linear_probe(feats, N_CLASSES, CLASS_NAMES, C_grid,
                           cfg.probe.max_iter, cfg.train.seed, cfg.probe.selection_metric)
        Ete = np.asarray(lw["test"][li][0], np.float32)
        Es = subsample(Ete, subsample_n, cfg.train.seed)
        rm = rankme(Es)
        dim = int(Ete.shape[1])
        layers[li] = {
            "f1_macro_pres": res["test"]["f1_macro_pres"],
            "f1_macro_all": res["test"]["f1_macro_all"],
            "best_C": res["best_C"],
            "rankme": rm,
            "rankme_normalized": rm / dim,
            "anisotropy": anisotropy(Es, seed=cfg.train.seed),
            "dim": dim,
        }
        print(f"[layerwise-probe] {key:20} L{li:02d}  F1={layers[li]['f1_macro_pres']:.4f}  "
              f"RankMe={rm:7.1f}  Aniso={layers[li]['anisotropy']:+.3f}")
    # corrélation intra-modèle (sur les couches)
    f1 = [layers[li]["f1_macro_pres"] for li in range(n_layers)]
    rmn = [layers[li]["rankme_normalized"] for li in range(n_layers)]
    ani = [layers[li]["anisotropy"] for li in range(n_layers)]
    within = {
        "n_layers": n_layers,
        "spearman_rankme_norm_vs_f1": _spearman(rmn, f1),
        "spearman_anisotropy_vs_f1": _spearman(ani, f1),
    }
    return {"n_layers": n_layers, "layers": layers, "within_model_correlation": within}


def make_figure(results: dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, r in results.items():
        xs = [r["layers"][li]["rankme_normalized"] for li in range(r["n_layers"])]
        ys = [r["layers"][li]["f1_macro_pres"] for li in range(r["n_layers"])]
        ax.plot(xs, ys, marker="o", ms=3, linewidth=1.0, alpha=0.8, label=key)
    ax.set_xlabel("RankMe normalisé (par couche)", fontsize=10)
    ax.set_ylabel("f1_macro_pres du probe (par couche)", fontsize=10)
    ax.set_title("Géométrie ↔ F1, intra-modèle couche-par-couche", fontsize=11,
                 fontweight="bold")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=7.5)
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
    ap.add_argument("--output", default="results/layerwise_probe.json")
    ap.add_argument("--fig", default="results/figures/layerwise_probe_pooled.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or DEFAULT_MODELS

    results = {}
    for key in models:
        try:
            results[key] = run_model(cfg, key)
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")

    # nuage poolé : toutes les paires (géométrie, F1) couche × modèle
    pooled_f1, pooled_rmn, pooled_ani = [], [], []
    for r in results.values():
        for li in range(r["n_layers"]):
            L = r["layers"][li]
            pooled_f1.append(L["f1_macro_pres"])
            pooled_rmn.append(L["rankme_normalized"])
            pooled_ani.append(L["anisotropy"])

    pooled = {}
    if len(pooled_f1) >= 3:
        pooled = {
            "n_points": len(pooled_f1),
            "spearman_rankme_norm_vs_f1": _spearman(pooled_rmn, pooled_f1),
            "kendall_rankme_norm_vs_f1": _kendall(pooled_rmn, pooled_f1),
            "spearman_anisotropy_vs_f1": _spearman(pooled_ani, pooled_f1),
            "kendall_anisotropy_vs_f1": _kendall(pooled_ani, pooled_f1),
        }

    out = {
        "metric": "f1_macro_pres",
        "selection_metric": cfg.probe.selection_metric,
        "models": results,
        "pooled_correlation": pooled,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output}  ({len(results)} modèles)")
    if pooled:
        print(f"\nNuage poolé ({pooled['n_points']} points couche×modèle) :")
        print(f"  RankMe norm. ↔ F1 : ρ={pooled['spearman_rankme_norm_vs_f1']:+.3f}  "
              f"τ={pooled['kendall_rankme_norm_vs_f1']:+.3f}")
        print(f"  Anisotropie  ↔ F1 : ρ={pooled['spearman_anisotropy_vs_f1']:+.3f}  "
              f"τ={pooled['kendall_anisotropy_vs_f1']:+.3f}")
    if results:
        make_figure(results, args.fig)


if __name__ == "__main__":
    main()
