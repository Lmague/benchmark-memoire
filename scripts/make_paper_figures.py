#!/usr/bin/env python3
"""Figures dérivées des JSON de résultats pour les deux supports (mainstream / géométrie).

Génère (dans --outdir, défaut results/figures_paper/) :
  PDF 1 (mainstream) :
    - significance_holm.png : matrice de distinguishabilité après Holm (FWER, α=0.05),
      modèles ordonnés par F1 (les blocs non distinguables = les paliers).
  PDF 2 (géométrie) :
    - corr_forest.png            : Spearman ρ ± IC95 des 9 métriques géométriques ↔ F1.
    - geometry_extended_scatter.png : stable_rank / participation_ratio / alpha_spectral ↔ F1.

Lit uniquement des JSON (aucun embedding) :
  significance_matrix.json, significance_corrected.json, geometry_extended.json.

Usage :
  python scripts/make_paper_figures.py \\
      --sig results/significance_matrix.json \\
      --corrected results/significance_corrected.json \\
      --geom results/geometry_extended.json \\
      --outdir results/figures_paper
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

TIER = {"dinov3_vitl16_lvd": "A", "simdinov2_vitl16": "A", "simdinov2_vitb16": "A",
        "dinov3_vitb16_lvd": "A", "dinov3_vitl16_sat": "B", "vitb16_imagenet": "C",
        "scalemae_vitl16": "C", "satmae_vitl16": "D", "resnet50_imagenet": "D"}
TCOL = {"A": "#1b7837", "B": "#7fbf7b", "C": "#fdae61", "D": "#d73027"}
MAE = {"satmae_vitl16", "scalemae_vitl16"}


def _short(m):
    return (m.replace("_vitl16", "_L").replace("_vitb16", "_B").replace("_imagenet", "_IN"))


def fig_significance_holm(sig, corrected, outpath):
    stats = sig["model_stats"]
    order = sorted(stats, key=lambda m: stats[m]["observed"], reverse=True)
    n = len(order)
    idx = {m: i for i, m in enumerate(order)}
    M = np.zeros((n, n))
    for v in corrected["pairs"].values():
        a, b = v["model_a"], v["model_b"]
        sig_holm = 1.0 if v["reject_holm"] else 0.0
        M[idx[a], idx[b]] = sig_holm
        M[idx[b], idx[a]] = sig_holm
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = matplotlib.colors.ListedColormap(["#f0f0f0", "#4575b4"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, color="black", zorder=2))
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, "✓" if M[i, j] else "·", ha="center", va="center",
                        color="white" if M[i, j] else "#999", fontsize=10,
                        fontweight="bold" if M[i, j] else "normal")
    labels = [f"{_short(m)} ({TIER[m]})" for m in order]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Distinguishabilité après Holm (FWER, α=0.05)\n"
                 "✓ = paire statistiquement séparée  ·  modèles triés par F1",
                 fontsize=10, fontweight="bold")
    leg = [Patch(facecolor="#4575b4", label="distinguables (Holm)"),
           Patch(facecolor="#f0f0f0", label="non distinguables")]
    ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


def fig_corr_forest(geom, outpath):
    corr = geom["correlations_vs_f1"]
    labels_fr = {
        "stable_rank": "Rang stable", "participation_ratio": "Participation ratio",
        "alpha_spectral": "Exposant spectral α", "rankme_normalized": "RankMe normalisé",
        "rankme": "RankMe brut", "anisotropy": "Anisotropie",
        "knn_purity": "Pureté kNN", "intrinsic_dim_twonn": "Dim. intrinsèque (TwoNN)",
        "fisher_ratio": "Ratio de Fisher",
    }
    items = [(k, v) for k, v in corr.items() if v.get("spearman_r") is not None]
    items.sort(key=lambda kv: kv[1]["spearman_r"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (k, v) in enumerate(items):
        r = v["spearman_r"]; ci = v.get("spearman_ci95", [r, r])
        sig = (ci[0] > 0) or (ci[1] < 0)
        col = "#1b7837" if sig else "#999999"
        ax.plot([ci[0], ci[1]], [i, i], color=col, lw=2, zorder=2)
        ax.plot(r, i, "o", color=col, ms=8, zorder=3)
        ax.text(1.02, i, "✓" if sig else "", color=col, fontsize=12, va="center",
                fontweight="bold")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([labels_fr.get(k, k) for k, _ in items], fontsize=9)
    ax.set_xlim(-1.05, 1.12)
    ax.set_xlabel("Spearman ρ (métrique géométrique ↔ f1_macro_pres),  n=9,  IC95 bootstrap",
                  fontsize=9)
    ax.set_title("Quelles métriques géométriques prédisent le F1 ?\n"
                 "vert = IC95 ne traverse pas 0 (significatif)", fontsize=10,
                 fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def fig_geometry_extended_scatter(geom, outpath):
    metrics = geom["metrics"]
    f1 = geom["f1_values_used"]
    corr = geom["correlations_vs_f1"]
    keys = [("stable_rank", "Rang stable"),
            ("participation_ratio", "Participation ratio"),
            ("alpha_spectral", "Exposant spectral α")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, (mk, name) in zip(axes, keys):
        for m in metrics:
            if m not in f1:
                continue
            ax.scatter(metrics[m][mk], f1[m], s=85, c=TCOL[TIER[m]],
                       marker="D" if m in MAE else "o", edgecolor="black",
                       linewidth=0.5, zorder=3)
            ax.annotate(_short(m), (metrics[m][mk], f1[m]), fontsize=6.5,
                        xytext=(4, 4), textcoords="offset points")
        r = corr[mk]["spearman_r"]; ci = corr[mk].get("spearman_ci95", [r, r])
        ax.set_xlabel(name, fontsize=9)
        ax.set_title(f"{name} ↔ F1\nρ={r:+.2f}  IC95 [{ci[0]:+.2f}, {ci[1]:+.2f}]",
                     fontsize=9.5, fontweight="bold")
        ax.grid(ls=":", alpha=0.5)
    axes[0].set_ylabel("f1_macro_pres (test)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def fig_controlled_pair(pairs_json, outpath):
    """Comparaison IC95 de la paire contrôlée (même archi) gelé vs fine-tuné."""
    d = json.load(open(pairs_json))
    m = d["models"]
    pr = d["pairs"][0]
    a, b = pr["model_a"], pr["model_b"]          # a = gelé, b = fine-tuné
    rows = [(b, "fine-tuné complet", "#d73027"), (a, "gelé (linear probe)", "#1b7837")]
    fig, ax = plt.subplots(figsize=(7.6, 2.0))
    for i, (key, lab, col) in enumerate(rows):
        s = m[key]["f1_macro_pres"]
        obs, lo, hi = s["observed"], s["ci95_low"], s["ci95_high"]
        ax.errorbar(obs, i, xerr=[[obs - lo], [hi - obs]], fmt="o", color=col,
                    ms=9, capsize=5, elinewidth=1.6, zorder=3)
        ax.text(hi + 0.0008, i, f"{obs:.4f}", va="center", fontsize=9)
        ax.text(lo - 0.0008, i, f"{key}\n({lab})", va="center", ha="right",
                fontsize=8, color=col)
    p2 = 2 * min(pr["p_a_gt_b"], 1 - pr["p_a_gt_b"])
    ax.set_ylim(-0.6, 1.6); ax.set_yticks([])
    ax.set_xlim(0.448, 0.482)
    ax.set_xlabel("f1_macro_pres (test) — point = observé, barre = IC95 bootstrap", fontsize=9)
    ax.set_title(f"Paire contrôlée ViT-B/16 :  Δ = {pr['delta_observed_a_minus_b']:+.4f},  "
                 f"P(gelé > FT) = {pr['p_a_gt_b']:.3f}  (bilatéral p ≈ {p2:.3f})",
                 fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sig", default="results/significance_matrix.json")
    ap.add_argument("--corrected", default="results/significance_corrected.json")
    ap.add_argument("--geom", default="results/geometry_extended.json")
    ap.add_argument("--controlled", default="results/bootstrap_pairs_controlled.json")
    ap.add_argument("--outdir", default="results/figures_paper")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sig = json.load(open(args.sig))
    corrected = json.load(open(args.corrected))
    geom = json.load(open(args.geom))

    fig_significance_holm(sig, corrected, os.path.join(args.outdir, "significance_holm.png"))
    fig_corr_forest(geom, os.path.join(args.outdir, "corr_forest.png"))
    fig_geometry_extended_scatter(geom, os.path.join(args.outdir, "geometry_extended_scatter.png"))
    if os.path.exists(args.controlled):
        fig_controlled_pair(args.controlled, os.path.join(args.outdir, "controlled_pair.png"))


if __name__ == "__main__":
    main()
