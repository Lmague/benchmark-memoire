#!/usr/bin/env python3
"""Génère les fragments LaTeX (tableaux) à inclure dans report.tex
   à partir des CSV/JSON produits par les tâches A-D.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
DATA = PROJ / "results" / "transfer"
LATEX = PROJ / "results" / "transfer"


def _esc(s: str) -> str:
    """Échapper les underscores et autres caractères LaTeX-sensibles."""
    return str(s).replace("_", "\\_")


def _format(v: float, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:{fmt}}"


def gen_tab_spectrum() -> None:
    df = pd.read_csv(DATA / "spectrum_metrics.csv")
    df = df.sort_values("f1_macro_pres_test", ascending=False).reset_index(drop=True)
    lines = [
        "\\toprule",
        "\\textbf{Modèle} & \\textbf{dim} & \\textbf{RankMe} & "
        "\\textbf{$\\alpha$-ReQ} & \\textbf{NESum} & \\textbf{Anisotropie} & "
        "\\textbf{F1-m(11) test} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']} & {int(r['dim'])} & {_format(r['rankme'], '.2f')} & "
            f"{_format(r['alpha_req'], '.3f')} & {_format(r['nesum'], '.3f')} & "
            f"{_format(r['anisotropy'], '+.4f')} & "
            f"{_format(r['f1_macro_pres_test'])} \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_spectrum.tex").write_text("\n".join(lines))
    print("tab_spectrum.tex écrit.")


def gen_tab_correl() -> None:
    with open(DATA / "correlations_with_f1.json") as f:
        d = json.load(f)
    lines = [
        "\\toprule",
        "\\textbf{Métrique} & \\textbf{$\\rho$ Spearman} & "
        "\\textbf{$p$ Spearman} & \\textbf{$\\tau$ Kendall} & "
        "\\textbf{$p$ Kendall} \\\\",
        "\\midrule",
    ]
    order = ["rankme", "rankme_normalized", "alpha_req", "nesum", "anisotropy"]
    pretty = {
        "rankme": "RankMe",
        "rankme_normalized": "RankMe normalisé",
        "alpha_req": "$\\alpha$-ReQ",
        "nesum": "NESum",
        "anisotropy": "Anisotropie",
    }
    for k in order:
        v = d["metrics_vs_f1_macro_pres_test"][k]
        lines.append(
            f"{pretty[k]} & {_format(v['spearman_r'], '+.3f')} & "
            f"{_format(v['spearman_p'], '.3g')} & "
            f"{_format(v['kendall_tau'], '+.3f')} & "
            f"{_format(v['kendall_p'], '.3g')} \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_correl.tex").write_text("\n".join(lines))
    print("tab_correl.tex écrit.")


def gen_tab_palier() -> None:
    """Compare ordering par F1, alpha_req, RankMe sur le palier A."""
    df = pd.read_csv(DATA / "spectrum_metrics.csv")
    palier = ["vitb16_arctic", "vitb16_fulft_arctic", "simdinov2_vitl16",
              "dinov3_vitb16_lvd"]
    df_p = df[df["model"].isin(palier)].copy()
    # Ordre par F1 décroissant
    by_f1 = df_p.sort_values("f1_macro_pres_test", ascending=False)["model"].tolist()
    by_alpha = df_p.sort_values("alpha_req", ascending=True)["model"].tolist()
    by_rankme = df_p.sort_values("rankme", ascending=False)["model"].tolist()
    by_f1_n = df_p.set_index("model").loc[by_f1, "f1_macro_pres_test"].round(4).tolist()
    by_alpha_n = df_p.set_index("model").loc[by_alpha, "alpha_req"].round(3).tolist()
    by_rankme_n = df_p.set_index("model").loc[by_rankme, "rankme"].round(1).tolist()
    lines = [
        "\\toprule",
        "\\textbf{Rang} & \\textbf{Par F1 test (desc.)} & "
        "\\textbf{Par $\\alpha$-ReQ (asc.)} & \\textbf{Par RankMe (desc.)} \\\\",
        "\\midrule",
    ]
    for i in range(len(by_f1)):
        lines.append(
            f"{i+1} & {by_f1[i]} ({by_f1_n[i]}) & "
            f"{by_alpha[i]} ({by_alpha_n[i]}) & "
            f"{by_rankme[i]} ({by_rankme_n[i]}) \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_palier.tex").write_text("\n".join(lines))
    print("tab_palier.tex écrit.")


def gen_tab_logme() -> None:
    df = pd.read_csv(DATA / "logme_scores.csv")
    df = df.sort_values("logme", ascending=False).reset_index(drop=True)
    lines = [
        "\\toprule",
        "\\textbf{Modèle} & \\textbf{dim} & \\textbf{LogME (train)} & "
        "\\textbf{t (s)} & \\textbf{F1-m(11) test} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']} & {int(r['dim'])} & {_format(r['logme'], '+.4f')} & "
            f"{_format(r['logme_time_s'], '.1f')} & {_format(r['f1_macro_pres_test'])} \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_logme.tex").write_text("\n".join(lines))
    print("tab_logme.tex écrit.")


def gen_tab_paired() -> None:
    df = pd.read_csv(DATA / "headline_pairs_paired_tests.csv")
    # Ordonner par A>B croissant (les paires indistinguables en premier)
    df = df.sort_values("aso_eps_min").reset_index(drop=True)
    lines = [
        "\\toprule",
        "\\textbf{Modèle A} & \\textbf{Modèle B} & \\textbf{$\\Delta$ acc} & "
        "\\textbf{ASO $\\eps_{min}$} & \\textbf{$p_{boot}$} & "
        "\\textbf{$p_{perm}$} & \\textbf{Verdict ASO} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        a, b = r["model_a"], r["model_b"]
        # Raccourcissement des noms longs
        a_short = a.replace("dinov3_", "d3_").replace("vitb16", "vb16")
        b_short = b.replace("dinov3_", "d3_").replace("vitb16", "vb16")
        verdict = r["aso_strict"]
        verdict = verdict.replace("confiance haute", "$\\uparrow\\uparrow$")
        verdict = verdict.replace("modéré", "$\\uparrow$")
        verdict = verdict.replace("indistinguables", "$\\approx$")
        lines.append(
            f"{a_short} & {b_short} & "
            f"{_format(r['delta_acc'], '+.4f')} & "
            f"{_format(r['aso_eps_min'], '.3f')} & "
            f"{_format(r['bootstrap_p_value'], '.3f')} & "
            f"{_format(r['permutation_p_value'], '.3f')} & "
            f"{verdict} \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_paired.tex").write_text("\n".join(lines))
    print("tab_paired.tex écrit.")


def gen_tab_f1() -> None:
    df = pd.read_csv(DATA / "f1_corrected_table.csv")
    df = df.sort_values("f1_macro_pres_stored", ascending=False).reset_index(drop=True)
    lines = [
        "\\toprule",
        "\\textbf{Modèle} & \\textbf{Acc} & \\textbf{F1-m(12)} & "
        "\\textbf{F1-m(11)} & \\textbf{F1-weighted} & "
        "\\textbf{$|\\Delta_{11}|$} & \\textbf{$|\\Delta_{w}|$} \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        d11 = r["f1_macro_pres_abs_diff"]
        dw = r["f1_weighted_abs_diff"]
        d11_str = f"{d11:.0e}" if d11 > 0 else "0"
        dw_str = f"{dw:.0e}" if dw > 0 else "0"
        lines.append(
            f"{r['model']} & {_format(r['accuracy'])} & "
            f"{_format(r['f1_macro_all_stored'])} & "
            f"{_format(r['f1_macro_pres_stored'])} & "
            f"{_format(r['f1_weighted_stored'])} & "
            f"{d11_str} & {dw_str} \\\\"
        )
    lines.append("\\bottomrule")
    (LATEX / "tab_f1.tex").write_text("\n".join(lines))
    print("tab_f1.tex écrit.")


if __name__ == "__main__":
    gen_tab_spectrum()
    gen_tab_correl()
    gen_tab_palier()
    gen_tab_logme()
    gen_tab_paired()
    gen_tab_f1()
    print("\nTous les fragments LaTeX générés.")
