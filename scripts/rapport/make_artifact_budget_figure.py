#!/usr/bin/env python3
"""Figure-signature du manuscrit « budget d'artefacts d'évaluation ».

Compare, sur une échelle commune de F1-macro, les artefacts du protocole
d'évaluation aux effets que la littérature rapporte habituellement.

Chaque valeur est sourcée. AUCUN nombre n'est inventé ici : ce script ne fait
que tracer des constantes déjà présentes dans les fichiers canoniques du dépôt.

    Artefacts
      A1 optimiseur du readout (lbfgs vs AdamW, MÊME modèle linéaire)
         -> results/rapport_data/mlp_probe_summary.md  (médiane delta_protocole)
      A2 capacité du readout (ReLU vs linéaire, optimiseur apparié)
         -> results/rapport_data/mlp_probe_summary.md  (médiane delta_nonlinearite)
      A3 nombre de threads BLAS (1 vs 3), solveur convergé
         -> AGENTS.md §4.8 / AGENT_MEMORY.md 2026-07-26 (0,4821 vs 0,4836)
      A4 basculement de best_C (LoRA r=8 seed0, écart de validation 1e-4)
         -> AGENT_MEMORY.md 2026-07-26 (+0,003 de F1 test)
      A5 basculement de best_C (QKV seed2)
         -> AGENT_MEMORY.md 2026-09-08 (0,4873 vs 0,4799)

    Effets
      E1 écart-type inter-seed (bruit de référence du projet)
         -> AGENTS.md §4.4
      E2..E4  régime affiné vs son propre gelé, à architecture égale
         -> paper_arctic_fm_benchmark/paper_arctic_fm_benchmark.tex §4.2
      E5 meilleur déployable vs meilleur gelé (non significatif, p=0,39)
         -> idem §4.1
      E6 étendue de l'ablation PEFT (13 bras x 3 seeds)
         -> idem §4.6 / Tableau ablation
      E7 NormTuning vs ancre LoRA r=8 (p=0,94)
         -> idem §4.6
      E8 gain de la fusion spatiale [tuile ; voisinage 512 px]
         -> idem §4.7
      E9 écart tirage aléatoire vs tirage spatial, à 1 % des données
         -> idem §4.4 (0,426 vs 0,099)

Usage :
    python scripts/rapport/make_artifact_budget_figure.py
Sortie :
    paper_arctic_fm_benchmark/figures/fig_artifact_budget.png
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper_arctic_fm_benchmark", "figures", "fig_artifact_budget.png",
)

# --------------------------------------------------------------------------
# Budget d'artefacts — amplitude en points de F1-macro
# --------------------------------------------------------------------------
ARTIFACTS = [
    ("A5  $C$ switch, single seed (QKV s2)", 0.0074),
    ("A1  Readout optimizer\n(lbfgs vs AdamW, same linear model)", 0.0061),
    ("A2  Readout capacity\n(ReLU vs linear, matched optimizer)", 0.0032),
    ("A4  $C$ switch, single seed (LoRA r8 s0)", 0.0030),
    ("A3  BLAS threads (1 vs 3)", 0.0015),
]

# --------------------------------------------------------------------------
# Effets rapportés
# --------------------------------------------------------------------------
EFFECTS = [
    ("E9  Random vs spatial split @1 % data", 0.3270),
    ("E8  Spatial context fusion\n(tile ; 512 px neighbourhood)", 0.0340),
    ("E2  LoRA r=8 vs its own frozen backbone", 0.0114),
    ("E3  MHSA vs its own frozen backbone", 0.0085),
    ("E1  Inter-seed s.d. (project noise floor)", 0.0080),
    ("E4  Full FT vs its own frozen backbone", 0.0072),
    ("E6  Spread of 13-arm PEFT ablation", 0.0068),
    ("E5  Best deployable vs best frozen\n(not significant, p = 0.39)", 0.0038),
    ("E7  NormTuning vs LoRA anchor (p = 0.94)", 0.0002),
]

SIGMA = 0.0080  # E1, ligne de référence

C_ART = "#c1272d"   # artefacts
C_EFF = "#2b6cb0"   # effets


def main() -> None:
    labels, vals, colors = [], [], []
    for name, v in ARTIFACTS:
        labels.append(name)
        vals.append(v)
        colors.append(C_ART)
    for name, v in EFFECTS:
        labels.append(name)
        vals.append(v)
        colors.append(C_EFF)

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    ax.barh(y, vals, color=colors, alpha=0.88, height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)

    for yy, v in zip(y, vals):
        ax.text(v * 1.06, yy, f"{v:.4f}", va="center", ha="left",
                fontsize=8, fontweight="bold", zorder=4)

    # seuil de bruit inter-seed
    ax.axvline(SIGMA, color="0.25", ls="--", lw=1.4, zorder=2)
    ax.text(0.0095, y[len(ARTIFACTS) - 1], "inter-seed s.d. = 0.0080",
            fontsize=8, color="0.25", style="italic", va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.2)
    ax.set_xlabel(r"$F1_{\mathrm{macro}}$ amplitude (11 classes, log scale)", fontsize=10)
    ax.grid(axis="x", which="both", ls=":", lw=0.6, alpha=0.45, zorder=0)

    # séparateur artefacts / effets
    n_art = len(ARTIFACTS)
    ax.axhline(y[n_art] + 0.5, color="0.6", lw=1.0)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_ART, alpha=0.88),
        plt.Rectangle((0, 0), 1, 1, color=C_EFF, alpha=0.88),
    ]
    ax.legend(handles, ["Evaluation artifact", "Reported effect"],
              loc="lower right", fontsize=9, frameon=True)

    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    fig.suptitle("Evaluation artifacts vs. reported effects",
                 fontsize=12, fontweight="bold", y=0.985)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    print("écrit :", OUT)


if __name__ == "__main__":
    main()
