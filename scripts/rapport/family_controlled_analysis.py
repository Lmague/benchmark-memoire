#!/usr/bin/env python3
"""Analyse approfondie de la famille géométrique contrôlée DINOv3-B.

Lit ``results/rapport_data/geometry_models.csv`` (protocole par-seed puis moyenne,
n fixe test = 17 598 — AUCUN recalcul), isole les 4 régimes DINOv3 ViT-B/16 :

  * ``dinov3_vitb16_lvd``          — backbone gelé (référence) ;
  * ``dinov3_vitb16_lvd_full``     — full fine-tuning ;
  * ``dinov3_vitb16_lvd_mhsa``     — MHSA seule ;
  * ``dinov3_vitb16_lvd_lora_r8``  — LoRA Q/V r=8.

Produit :
  1. ``rapport_bouguessa/tables/t_family_controlled.tex`` — tableau booktabs
     (lignes = 10 métriques, colonnes = Gelé | Full | MHSA | LoRA | Δ LoRA−gelé) ;
  2. ``rapport_bouguessa/figs/geom_family_controlled.png`` — figure 2 panneaux :
     (a) Δ vs gelé par régime (barres horizontales groupées, unité = σ de la
         famille, métriques triées par |Δ LoRA|) ;
     (b) écart de LoRA au centroïde (Full, MHSA), même unité σ — met en évidence
         les métriques où LoRA sort du sillon Full/MHSA.

Les Δ sont normalisés par l'écart-type inter-régime (σ_family) pour être
comparables entre métriques d'échelles différentes (silhouette ≈ 0,03 vs
logme ≈ 44 600). Les corrélations de Spearman (n = 4) sont descriptives.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "rapport_bouguessa", "figs"))
import vizstyle  # noqa: E402

CSV = os.path.join(BASE, "results", "rapport_data", "geometry_models.csv")
TABLE = os.path.join(BASE, "rapport_bouguessa", "tables", "t_family_controlled.tex")
FIG = os.path.join(BASE, "rapport_bouguessa", "figs", "geom_family_controlled.png")

# ── Identité des 4 régimes (ordre d'affichage) ──────────────────────────────
FROZEN = "dinov3_vitb16_lvd"
FULL = "dinov3_vitb16_lvd_full"
MHSA = "dinov3_vitb16_lvd_mhsa"
LORA = "dinov3_vitb16_lvd_lora_r8"
ORDER = [FROZEN, FULL, MHSA, LORA]

REGIME_LABEL = {
    FROZEN: "Gelé",
    FULL: "Full FT",
    MHSA: "MHSA",
    LORA: "LoRA r=8",
}
REGIME_COLOR = {
    FULL: vizstyle.SERIES[6],   # #4a3aa7 (violet) — cohérent vizstyle.MODEL_COLOR
    MHSA: vizstyle.SERIES[7],   # #e34948 (rouge)
    LORA: vizstyle.SERIES[0],   # #2a78d6 (bleu)
}

# F1 canoniques (registry.py CANONICAL_F1) — non recalculées ici.
F1 = {
    FROZEN: (0.4712, None),
    FULL: (0.4786, 0.0027),
    MHSA: (0.4800, 0.0020),
    LORA: (0.4835, 0.0011),
}

# Sélection raisonnée de 10 métriques pour la table (et la figure).
METRICS = [
    "eff_rank_sigma2",      # rang effectif σ² (centré) — « RankMe » des rapports
    "stable_rank",
    "aniso_cosine",
    "fisher_ratio",
    "knn_purity",
    "silhouette",
    "nc1_canonical",        # tr(Σ_W)/tr(Σ_B) — bas = bon
    "alpha_spectral",       # exposant du spectre (loi de puissance)
    "intrinsic_dim_twonn",
    "logme_test",
]
METRIC_LABEL = {
    "eff_rank_sigma2": "Rang effectif $\\sigma^2$",
    "stable_rank": "Rang stable",
    "aniso_cosine": "Anisotropie (cos)",
    "fisher_ratio": "Rapport de Fisher",
    "knn_purity": "Pureté kNN",
    "silhouette": "Silhouette",
    "nc1_canonical": "NC1 (bas = bon)",
    "alpha_spectral": "Exposant $\\alpha$",
    "intrinsic_dim_twonn": "Dim. intrinsèque (2NN)",
    "logme_test": "LogME (test)",
}
DECIMALS = {
    "eff_rank_sigma2": 1,
    "stable_rank": 2,
    "aniso_cosine": 3,
    "fisher_ratio": 3,
    "knn_purity": 3,
    "silhouette": 4,
    "nc1_canonical": 2,
    "alpha_spectral": 2,
    "intrinsic_dim_twonn": 2,
    "logme_test": 1,
}
THOUSANDS = {"logme_test"}   # séparateur de milliers


def fr(x: float | None, ndec: int, thousands: bool = False) -> str:
    """Format nombre français (virgule décimale, séparateur de milliers optionnel)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{ndec}f}"
    ip, dp = s.split(".")
    if thousands:
        ip = f"{int(ip):,}".replace(",", "\\,")
    return f"{ip},{dp}"


def fmt_val(metric: str, val: float | None) -> str:
    return fr(val, DECIMALS[metric], thousands=(metric in THOUSANDS))


def fmt_pm(metric: str, val: float | None, std: float | None) -> str:
    """Valeur ± std (std affiché seulement si non nul)."""
    base = fmt_val(metric, val)
    if std is None or std == 0.0 or (isinstance(std, float) and not np.isfinite(std)):
        return base
    # std arrondi au même nombre de décimales que la valeur
    return base + r"\,\tiny{$\pm$\,%s}" % fr(std, DECIMALS[metric],
                                             thousands=(metric in THOUSANDS))


def fmt_delta(d: float, ndec: int, thousands: bool = False) -> str:
    """Δ signé au format français (+/-)."""
    sign = "+" if d >= 0 else "-"
    if thousands:
        body = f"{abs(d):,.{ndec}f}".replace(",", "\\,").replace(".", ",")
    else:
        body = f"{abs(d):.{ndec}f}".replace(".", ",")
    return sign + body


def main() -> None:
    df = pd.read_csv(CSV)
    # Régimes DINOv3-B présents dans le CSV
    fam = df[df["model"].isin(ORDER)].set_index("model")
    missing = [m for m in ORDER if m not in fam.index]
    if missing:
        raise SystemExit(f"[ERROR] régimes absents du CSV : {missing}")

    # Valeurs (moyennes) + std pour les métriques retenues
    values: dict[str, dict[str, float]] = {}
    stds: dict[str, dict[str, float]] = {}
    for m in METRICS:
        values[m] = {r: float(fam.loc[r, m]) for r in ORDER}
        stds[m] = {r: float(fam.loc[r, m + "_std"]) for r in ORDER}

    # ── Δ vs gelé + normalisation par σ de la famille ────────────────────────
    # σ_family = écart-type des 4 régimes pour chaque métrique (unité comparable).
    delta = {}          # Δ_σ (regime − gelé) / σ_family
    sigma_fam = {}
    for m in METRICS:
        arr = np.array([values[m][r] for r in ORDER], dtype=float)
        sigma_fam[m] = float(arr.std(ddof=0))
        s = sigma_fam[m] if sigma_fam[m] > 1e-12 else 1.0
        delta[m] = {r: (values[m][r] - values[m][FROZEN]) / s for r in (FULL, MHSA, LORA)}

    # ── Écart de LoRA au centroïde (Full, MHSA), en σ ─────────────────────────
    lora_dev = {}
    for m in METRICS:
        s = sigma_fam[m] if sigma_fam[m] > 1e-12 else 1.0
        centroid = 0.5 * (values[m][FULL] + values[m][MHSA])
        lora_dev[m] = (values[m][LORA] - centroid) / s

    # Identification de la métrique où LoRA s'écarte le plus de Full/MHSA
    dev_rank = sorted(METRICS, key=lambda m: abs(lora_dev[m]), reverse=True)
    top_dev = dev_rank[0]

    # ── Corrélations de Spearman métrique ↔ F1 (n = 4, descriptif) ───────────
    f1_vec = np.array([F1[r][0] for r in ORDER], dtype=float)
    print("\nCorrélations de Spearman (n = 4, descriptif) :")
    spear = {}
    for m in METRICS:
        rho, p = stats.spearmanr([values[m][r] for r in ORDER], f1_vec)
        spear[m] = (rho, p)
        print(f"  {METRIC_LABEL[m]:<26} ρ = {rho:+5.2f}   p = {p:.3f}")

    print("\nΔ vs gelé (en unités de σ_family) :")
    print(f"  {'métrique':<26} {'Full':>7} {'MHSA':>7} {'LoRA':>7}  {'LoRA−centroïde':>15}")
    for m in dev_rank:
        print(f"  {METRIC_LABEL[m]:<26} {delta[m][FULL]:+7.2f} {delta[m][MHSA]:+7.2f} "
              f"{delta[m][LORA]:+7.2f}  {lora_dev[m]:+15.2f}")
    print(f"\n→ Métrique où LoRA s'écarte le plus de Full/MHSA : "
          f"{METRIC_LABEL[top_dev]}  (LoRA−centroïde = {lora_dev[top_dev]:+.2f} σ)")

    # ══════════════════════════════════════════════════════════════════════════
    # TABLE LaTeX
    # ══════════════════════════════════════════════════════════════════════════
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(
        r"\caption{Famille géométrique contrôlée DINOv3 ViT-B/16 (768~dim.)~: backbone "
        r"gelé et ses trois régimes d'affinage, même pré-entraînement, même test "
        r"($n=17\,598$). $\pm$ = écart-type inter-seed (gelé~: $n=1$ seed). "
        r"$\Delta$ = LoRA $-$ gelé. F1 macro stable dans une bande de 0,012 alors que "
        r"la géométrie diverge fortement.}\label{tab:family_controlled}")
    lines.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Métrique & Gelé & Full FT & MHSA & LoRA r=8 & $\Delta$ LoRA $-$ gelé \\")
    lines.append(r"\midrule")
    # Ligne F1 de contexte (ancrage performance)
    f1_cells = [fr(F1[r][0], 4) if F1[r][1] is None else fmt_pm_simple(F1[r][0], F1[r][1])
                for r in ORDER]
    f1_delta = F1[LORA][0] - F1[FROZEN][0]
    lines.append("F1 macro & " + " & ".join(f1_cells) + f" & {fmt_delta(f1_delta, 4)} \\\\")
    lines.append(r"\addlinespace")
    for m in METRICS:
        cells = [fmt_pm(m, values[m][FROZEN], stds[m][FROZEN]),
                 fmt_pm(m, values[m][FULL], stds[m][FULL]),
                 fmt_pm(m, values[m][MHSA], stds[m][MHSA]),
                 fmt_pm(m, values[m][LORA], stds[m][LORA])]
        d = values[m][LORA] - values[m][FROZEN]
        d_str = fmt_delta(d, DECIMALS[m], thousands=(m in THOUSANDS))
        lines.append(f"{METRIC_LABEL[m]} & " + " & ".join(cells) + f" & {d_str} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{flushleft}\footnotesize")
    lines.append(r"Source~: \texttt{results/rapport\_data/geometry\_models.csv} "
                 r"(embeddings test, protocole par-seed puis moyenne). "
                 r"NC1 = $\mathrm{tr}(\Sigma_W)/\mathrm{tr}(\Sigma_B)$ (bas = meilleur). "
                 r"LogME (test)~: plus haut = meilleur. "
                 r"F1 canoniques~: \texttt{registry.py} \texttt{CANONICAL\_F1}.")
    lines.append(r"\end{flushleft}")
    lines.append(r"\end{table}")
    with open(TABLE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  [TABLE] {TABLE}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 panneaux
    # ══════════════════════════════════════════════════════════════════════════
    vizstyle.setup()
    import matplotlib.pyplot as plt

    # Tri des métriques par |Δ_σ LoRA| (croissant → barres horizontales du haut vers le bas)
    order_a = sorted(METRICS, key=lambda m: abs(delta[m][LORA]))
    # Panneau (b) : tri par écart LoRA−centroïde (croissant)
    order_b = sorted(METRICS, key=lambda m: lora_dev[m])

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.2, 5.6))

    # ── (a) Δ vs gelé par régime (barres horizontales groupées) ───────────────
    y = np.arange(len(order_a))
    bar_h = 0.26
    regimes = [FULL, MHSA, LORA]
    for i, r in enumerate(regimes):
        vals = [delta[m][r] for m in order_a]
        axa.barh(y + (i - 1) * bar_h, vals, height=bar_h,
                 color=REGIME_COLOR[r], label=REGIME_LABEL[r], zorder=3)
    axa.axvline(0, color=vizstyle.RULE, lw=1.1, zorder=2)
    axa.set_yticks(y)
    axa.set_yticklabels([METRIC_LABEL[m] for m in order_a], fontsize=7.5)
    axa.set_xlabel("Δ vs gelé (unités de $\\sigma_{\\mathrm{famille}}$)")
    axa.set_title("(a) Déplacement géométrique vs backbone gelé", fontsize=10)
    axa.legend(loc="lower right", ncol=1, frameon=False)
    vizstyle.clean(axa)

    # ── (b) écart de LoRA au centroïde (Full, MHSA) ───────────────────────────
    yb = np.arange(len(order_b))
    vals_b = [lora_dev[m] for m in order_b]
    colors_b = [vizstyle.SERIES[0] if v >= 0 else vizstyle.SERIES[7]
                for v in vals_b]
    axb.barh(yb, vals_b, height=0.62, color=colors_b, zorder=3)
    axb.axvline(0, color=vizstyle.RULE, lw=1.1, zorder=2)
    axb.set_yticks(yb)
    axb.set_yticklabels([METRIC_LABEL[m] for m in order_b], fontsize=7.5)
    axb.set_xlabel("LoRA − centroïde(Full, MHSA) (unités de $\\sigma_{\\mathrm{famille}}$)")
    axb.set_title("(b) Où LoRA sort du sillon Full/MHSA", fontsize=10)
    axb.annotate("bleu = LoRA au-dessus du centroïde\nrouge = LoRA en dessous",
                 xy=(0.03, 0.03), xycoords="axes fraction", fontsize=6.6,
                 color=vizstyle.INK_SOFT, va="bottom")
    vizstyle.clean(axb)

    fig.tight_layout()
    vizstyle.savefig(fig, FIG)

    # ── Résumé final (pour le rapport parent) ────────────────────────────────
    print("\n══ RÉSUMÉ ══")
    print(f"Δ le plus frappant LoRA vs Full/MHSA : {METRIC_LABEL[top_dev]} "
          f"(LoRA−centroïde = {lora_dev[top_dev]:+.2f} σ_famille)")
    print(f"  valeurs : gelé={fmt_val(top_dev, values[top_dev][FROZEN])}, "
          f"Full={fmt_val(top_dev, values[top_dev][FULL])}, "
          f"MHSA={fmt_val(top_dev, values[top_dev][MHSA])}, "
          f"LoRA={fmt_val(top_dev, values[top_dev][LORA])}")


def fmt_pm_simple(val: float, std: float) -> str:
    s = fr(val, 4)
    return s + r"\,\tiny{$\pm$\,%s}" % fr(std, 4)


if __name__ == "__main__":
    main()
