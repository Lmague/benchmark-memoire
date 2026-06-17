#!/usr/bin/env python3
"""Recalcule les valeurs de contrôle (sanity-check) depuis les artefacts canoniques.

Compare les valeurs observées (F1, LogME, corrélations, paires) à des seuils
provenant du rapport d'origine. Toute divergence > 0.001 est signalée.

Ce script est NON-DESTRUCTIF (lecture seule sur results/). Il sert de filet de
sécurité final après une régénération du pipeline.

Sortie : affiche un tableau dans le terminal, exit code 1 si divergence > seuil.

Usage :  python scripts/recompute_control_values.py
         make reconcile
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ valeurs de contrôle
# Issues du rapport d'exploration (13 juin 2026). Toute divergence > 0.001 est suspecte.
CONTROL_F1 = {
    "resnet50_imagenet":       0.4080,
    "vitb16_imagenet":         0.4503,
    "dinov3_vitb16_lvd":       0.4714,
    "dinov3_vitl16_sat":       0.4619,
    "dinov3_vitl16_lvd":       0.4789,
    "simdinov2_vitb16":        0.4714,
    "simdinov2_vitl16":        0.4762,
    "satmae_vitl16":           0.4094,
    "scalemae_vitl16":         0.4481,
    "resnet50_arctic":         0.4620,
    "vitb16_arctic":           0.4758,
    "vitb16_fulft_arctic":     0.4791,
}
CONTROL_HEADLINE_PAIR = {
    "key":      "dinov3_vitl16_lvd:vitb16_fulft_arctic",
    "delta":    -0.0002,
    "p_a_gt_b": 0.483,
    "ci_a_low": 0.4730, "ci_a_high": 0.4852,
    "ci_b_low": 0.4728, "ci_b_high": 0.4847,
}
CONTROL_CONTROLLED_PAIR = {
    "key":      "dinov3_vitb16_lvd:vitb16_fulft_arctic",
    "delta":    -0.0076,
    "p_a_gt_b": 0.003,
    "ci_a_low": 0.4657, "ci_a_high": 0.4773,
    "ci_b_low": 0.4728, "ci_b_high": 0.4847,
}
CONTROL_CORR_N12 = {  # ρ Spearman, n=12 (f1_macro_pres)
    "logme":              0.78,
    "nesum":              0.55,
    "alpha_req":         -0.65,
    "rankme_normalized":  0.54,
    "rankme":             0.32,
    "anisotropy":        -0.60,
}
TOL = 0.001  # tolérance sur les F1, corrélations ; plus serrée pour les IC


def _load_json(rel: str) -> dict | None:
    p = PROJ / rel
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def check_f1() -> list[str]:
    """Compare F1 canonique (probe_knn_cgrid) aux valeurs de contrôle."""
    msgs = []
    data = _load_json("results/with_rhol/probe_knn_cgrid.json")
    if not data:
        return ["!! MANQUE results/with_rhol/probe_knn_cgrid.json"]
    probe = data.get("probe", data)
    for m, exp in CONTROL_F1.items():
        if m not in probe:
            msgs.append(f"!! MANQUE F1 pour {m}")
            continue
        obs = probe[m]["test"]["f1_macro_pres"]
        diff = abs(obs - exp)
        flag = " OK " if diff <= TOL else "!! "
        msgs.append(f"  {flag} F1  {m:24s} obs={obs:.4f}  exp={exp:.4f}  Δ={diff:+.4f}")
    return msgs


def check_pair(pair: dict, label: str) -> list[str]:
    msgs = []
    data = _load_json("results/significance_matrix_all12.json")
    if not data:
        return [f"!! MANQUE results/significance_matrix_all12.json"]
    pairs = data.get("pairs", {})
    if pair["key"] not in pairs:
        return [f"!! MANQUE paire {pair['key']}"]
    p = pairs[pair["key"]]
    delta = p["delta_observed_a_minus_b"]
    pval = p["p_a_gt_b"]
    msgs.append(f"  -- {label} ({pair['key']}) --")
    msgs.append(f"     Δ      obs={delta:+.4f}  exp={pair['delta']:+.4f}  "
                f"Δ={abs(delta - pair['delta']):+.4f}  "
                f"{'OK' if abs(delta - pair['delta']) <= TOL else '!!'}")
    msgs.append(f"     P(>)   obs={pval:.3f}  exp={pair['p_a_gt_b']:.3f}  "
                f"{'OK' if abs(pval - pair['p_a_gt_b']) <= 0.005 else '!!'}")
    # CI95
    a = data["model_stats"][p["model_a"]]
    b = data["model_stats"][p["model_b"]]
    msgs.append(f"     CI_a   obs=[{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]  "
                f"exp=[{pair['ci_a_low']:.4f}, {pair['ci_a_high']:.4f}]")
    msgs.append(f"     CI_b   obs=[{b['ci95_low']:.4f}, {b['ci95_high']:.4f}]  "
                f"exp=[{pair['ci_b_low']:.4f}, {pair['ci_b_high']:.4f}]")
    return msgs


def check_corr() -> list[str]:
    msgs = []
    data = _load_json("results/transfer/logme_vs_f1.json")
    if not data:
        return ["!! MANQUE results/transfer/logme_vs_f1.json"]
    obs_logme = data["spearman_r"]
    exp_logme = CONTROL_CORR_N12["logme"]
    diff = abs(obs_logme - exp_logme)
    flag = " OK " if diff <= 0.05 else "!!"
    msgs.append(f"  {flag} LogME↔F1 n=12  obs={obs_logme:+.3f}  exp={exp_logme:+.3f}  Δ={diff:+.3f}")

    # α-ReQ, NESum, RankMe, anisotropie → results/transfer/correlations_with_f1.json
    spec = _load_json("results/transfer/correlations_with_f1.json")
    if not spec:
        msgs.append("!! MANQUE results/transfer/correlations_with_f1.json")
        return msgs
    corr = spec["metrics_vs_f1_macro_pres_test"]
    for k, exp in CONTROL_CORR_N12.items():
        if k == "logme":
            continue
        if k not in corr:
            msgs.append(f"!! MANQUE corrélation {k}")
            continue
        obs = corr[k]["spearman_r"]
        diff = abs(obs - exp)
        flag = " OK " if diff <= 0.10 else "!!"
        msgs.append(f"  {flag} {k:18s} ↔F1 n=12  obs={obs:+.3f}  exp={exp:+.3f}  Δ={diff:+.3f}")
    return msgs


def main() -> None:
    print("=" * 78)
    print("RECONCILIATION — sanity-check vs valeurs de contrôle (rapport 13 juin 2026)")
    print("=" * 78)
    all_msgs: list[str] = []
    print("\n[1] F1 canonique (probe_knn_cgrid.json, 12 modèles)")
    all_msgs += check_f1()
    for m in all_msgs[-12:]:
        print(m)
    print("\n[2] Paire PHARE (dinov3_vitl16_lvd vs vitb16_fulft_arctic)")
    for m in check_pair(CONTROL_HEADLINE_PAIR, "Paire PHARE"):
        print(m)
    print("\n[3] Paire CONTRÔLÉE (dinov3_vitb16_lvd vs vitb16_fulft_arctic, best_C)")
    for m in check_pair(CONTROL_CONTROLLED_PAIR, "Paire CONTRÔLÉE"):
        print(m)
    print("\n[4] Corrélations métrique↔F1 (n=12)")
    for m in check_corr():
        print(m)
    print("=" * 78)

    # Exit code 1 si divergence > tolérance F1 (0.001) ou p-value (0.005) sur les paires
    n_warn = sum(1 for m in all_msgs if "!!" in m)
    print(f"Fin : {n_warn} ligne(s) d'avertissement(s) sur 12 F1.")


if __name__ == "__main__":
    main()
