#!/usr/bin/env python3
"""Figures canoniques 9 modèles (post mean-pool MAE) pour le support de réunion.

Remplace les figures périmées (7 modèles, sans les MAE, pré-correction du pooling) :
``f1_barplot_with_rhol.png``, ``anisotropy_vs_f1_with_rhol.png``, ``tsne_all_models.png``.

Sorties (dans --outdir, défaut results/figures_support/) :
  - f1_ranking_ci.png   : classement f1_macro_pres + IC95 bootstrap, coloré par palier.
  - geometry_vs_f1.png  : RankMe normalisé ↔ F1 et anisotropie ↔ F1 (9 modèles, MAE = ◇).

Lit uniquement des JSON canoniques (aucun embedding) :
  significance_matrix.json (F1 + IC95), correlations.json (rankme_normalized, anisotropie).

Usage :
  python scripts/make_support_figures.py \\
      --sig results/significance_matrix.json \\
      --corr results/correlations.json \\
      --outdir results/figures_support
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Paliers statistiques (cf. significance_matrix.json — IC95 disjoints de la tête A).
TIER = {
    "dinov3_vitl16_lvd": "A", "simdinov2_vitl16": "A",
    "simdinov2_vitb16": "A", "dinov3_vitb16_lvd": "A",
    "dinov3_vitl16_sat": "B",
    "vitb16_imagenet": "C", "scalemae_vitl16": "C",
    "satmae_vitl16": "D", "resnet50_imagenet": "D",
}
TCOL = {"A": "#1b7837", "B": "#7fbf7b", "C": "#fdae61", "D": "#d73027"}
MAE = {"satmae_vitl16", "scalemae_vitl16"}


def _short(m: str) -> str:
    return (m.replace("_vitl16", "_L").replace("_vitb16", "_B")
             .replace("_imagenet", "_IN"))


def fig_ranking(sig: dict, outpath: str) -> None:
    stats = sig["model_stats"]
    order = sorted(stats, key=lambda m: stats[m]["observed"])
    obs = [stats[m]["observed"] for m in order]
    lo = [stats[m]["observed"] - stats[m]["ci95_low"] for m in order]
    hi = [stats[m]["ci95_high"] - stats[m]["observed"] for m in order]
    cols = [TCOL[TIER[m]] for m in order]
    labels = [m + ("  (mean-pool)" if m in MAE else "") for m in order]

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = list(range(len(order)))
    ax.barh(y, obs, color=cols, edgecolor="black", linewidth=0.4, zorder=2)
    ax.errorbar(obs, y, xerr=[lo, hi], fmt="none", ecolor="black",
                elinewidth=1.0, capsize=3, zorder=3)
    for i, o in enumerate(obs):
        ax.text(o + hi[i] + 0.004, i, f"{o:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.38, 0.50)
    ax.set_xlabel("f1_macro_pres (test) — IC95 bootstrap (n=%d)" % sig.get("n_bootstrap", 1000),
                  fontsize=9)
    ax.set_title("Classement des 9 foundation models gelés", fontsize=11, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    leg = [Patch(facecolor=TCOL[t], edgecolor="black", label=f"Palier {t}")
           for t in ["A", "B", "C", "D"]]
    ax.legend(handles=leg, loc="lower right", fontsize=8, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def fig_geometry(sig: dict, corr: dict, outpath: str) -> None:
    stats = sig["model_stats"]
    lat = corr["latent_values_used"]

    def panel(ax, xkey, xlabel, title):
        for m in stats:
            if m not in lat:
                continue
            x, yv = lat[m][xkey], stats[m]["observed"]
            ax.scatter(x, yv, s=90, c=TCOL[TIER[m]], marker="D" if m in MAE else "o",
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.annotate(_short(m), (x, yv), fontsize=6.8, xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("f1_macro_pres (test)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(ls=":", alpha=0.5)

    g = corr.get("global_rankme_normalized", {}).get("rankme_normalized", {})
    a = corr.get("global", {}).get("anisotropy", {})
    rho_n = g.get("spearman_r"); ci_n = g.get("spearman_ci95")
    rho_a = a.get("spearman_r"); ci_a = a.get("spearman_ci95")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    t1 = "RankMe norm. ↔ F1"
    if rho_n is not None:
        t1 += f"   (ρ={rho_n:.2f}, IC95 [{ci_n[0]:.2f}, {ci_n[1]:.2f}])"
    t2 = "Anisotropie ↔ F1"
    if rho_a is not None:
        t2 += f"   (ρ={rho_a:.2f}, IC95 [{ci_a[0]:.2f}, {ci_a[1]:.2f}])"
    panel(axes[0], "rankme_normalized", "RankMe normalisé  (rang effectif / dim)", t1)
    panel(axes[1], "anisotropy", "Anisotropie (cosinus moyen)", t2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sig", default="results/significance_matrix.json")
    ap.add_argument("--corr", default="results/correlations.json")
    ap.add_argument("--outdir", default="results/figures_support")
    args = ap.parse_args()

    with open(args.sig) as f:
        sig = json.load(f)
    with open(args.corr) as f:
        corr = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    fig_ranking(sig, os.path.join(args.outdir, "f1_ranking_ci.png"))
    fig_geometry(sig, corr, os.path.join(args.outdir, "geometry_vs_f1.png"))


if __name__ == "__main__":
    main()
