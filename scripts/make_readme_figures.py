#!/usr/bin/env python3
"""Regenerate the README figures in English, using the report's house style.

Outputs, all written to figures/:
    fig_readme_ranking.png   all canonical models plus the frozen context control
    fig_readme_context.png   frozen, distillation and fusion, side by side
    fig_readme_latent.png    spectra, ranking power per metric, probe dimension
    fig_readme_datacurve.png random draw vs spatially blocked draw

Canonical sources only:
    results/all_models_canonical_merged.json
    results/rapport_data/geometry_models.csv
    results/rapport_data/spectra_cache.npz
    results/datacurve_lora/{results_agg,frozen_probe_results_agg}.csv
    lora_spatial_v2/results_spatial_summary.csv

Run from anywhere: python3 scripts/make_readme_figures.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "rapport_bouguessa" / "figs"))
import vizstyle as V  # noqa: E402

V.setup()

TYPE_COLOR = {
    "frozen": V.SERIES[0],
    "ft_fresh": V.SERIES[1],
    "ft_old": V.SERIES[2],
    "scratch": V.SERIES[4],
    "frozen_ctx": "#7d3c98",
}
TYPE_LABEL = {
    "frozen": "Frozen",
    "ft_fresh": "Fine-tuned, SSL init",
    "ft_old": "Fine-tuned, ImageNet init",
    "scratch": "From scratch",
    "frozen_ctx": "Frozen, fused readout",
}

PRETTY = {
    "ctxdistill_dB_tL": "Context fusion (neighbourhood at inference)",
    "ctxdistill_dB_tSL_r8a16": "Context fusion, SimDINOv2-B r8a16",
    "ctxdistill_dB_tSL_r2a4": "Context fusion, SimDINOv2-B r2a4",
    "ctxdistill_dA_tL": "Context distillation (tile only)",
    "ctxdistill_dA_tEMA": "Context distillation, EMA teacher",
    "dinov3_vith16plus_lvd": "DINOv3 ViT-H+/16",
    "dinov3_vitl16_lvd": "DINOv3 ViT-L/16",
    "dinov3_vitb16_lvd": "DINOv3 ViT-B/16",
    "dinov3_vits16_lvd": "DINOv3 ViT-S/16",
    "dinov3_vitl16_sat": "DINOv3 ViT-L/16, satellite",
    "simdinov2_vitl16": "SimDINOv2 ViT-L/16",
    "simdinov2_vitb16": "SimDINOv2 ViT-B/16",
    "satmae_vitl16": "SatMAE ViT-L/16",
    "scalemae_vitl16": "Scale-MAE ViT-L/16",
    "resnet50_imagenet": "ResNet-50, ImageNet",
    "vitb16_imagenet": "ViT-B/16, ImageNet",
    "dinov3_vitb16_lvd_lora_r8": "DINOv3-B, LoRA r=8",
    "dinov3_vitl16_lvd_lora": "DINOv3-L, LoRA r=8",
    "dinov3_vits16_lvd_lora_r8": "DINOv3-S, LoRA r=8",
    "dinov3_vitb16_lvd_mhsa": "DINOv3-B, attention only",
    "dinov3_vitb16_lvd_full": "DINOv3-B, full fine-tuning",
    "simdinov2_vitl16_lora": "SimDINOv2-L, LoRA r=8",
    "simdinov2_vitb16_lora": "SimDINOv2-B, LoRA r=8",
    "simdinov2_vitb16_norm_tuning": "SimDINOv2-B, norms and head only",
    "simdinov2_vitb16_full": "SimDINOv2-B, full fine-tuning",
    "simdinov2_vitb16_mhsa": "SimDINOv2-B, attention only",
    "simdinov2_vitb16_lora_r2": "SimDINOv2-B, LoRA r=2",
    "simdinov2_vitb16_lora_r4": "SimDINOv2-B, LoRA r=4",
    "simdinov2_vitb16_lora_r16": "SimDINOv2-B, LoRA r=16",
    "simdinov2_vitb16_lora_r32": "SimDINOv2-B, LoRA r=32",
    "simdinov2_vitb16_lora_r8_qkv": "SimDINOv2-B, LoRA Q+K+V",
    "simdinov2_vitb16_lora_r8_b05": "SimDINOv2-B, LoRA blocks 0-5",
    "simdinov2_vitb16_lora_r8_b611": "SimDINOv2-B, LoRA blocks 6-11",
    "simdinov2_vitb16_lora_r8_b911": "SimDINOv2-B, LoRA blocks 9-11",
    "simdinov2_vitb16_lora_r8_s2": "SimDINOv2-B, LoRA scaling 2",
    "simdinov2_vitb16_lora_r16_s2": "SimDINOv2-B, LoRA r=16 scaling 2",
    "simdinov2_vitb16_lora_r8_rslora": "SimDINOv2-B, LoRA rsLoRA",
    "simdinov2_vitb16_lora_r16_rslora": "SimDINOv2-B, LoRA r=16 rsLoRA",
    "vitb16_full_old": "ViT-B/16 ImageNet, full fine-tuning",
    "vitb16_mhsa_old": "ViT-B/16 ImageNet, attention only",
    "resnet50_fulft_sota": "ResNet-50, fine-tuned",
    "vitb16_scratch_old": "ViT-B/16, from scratch",
}


def pretty(model: str, display: str | None = None) -> str:
    return PRETTY.get(model) or display or model


def load_registry() -> list[dict]:
    entries = json.loads((ROOT / "results/all_models_canonical_merged.json").read_text())["models"]
    out = []
    for m in entries:
        if "explora" in m["model"]:
            continue
        kind = m.get("type", "frozen")
        if "scratch" in m["model"]:
            kind = "scratch"
        out.append({
            "model": m["model"],
            "display": m.get("display"),
            "type": kind,
            "f1": m["f1_linear_probe"],
            "std": m.get("f1_std"),
            "seeds": m.get("f1_seeds"),
            "dim": m.get("dim"),
        })
    return out


# --------------------------------------------------------------------------
# 1. Ranking, single panel with a dedicated value column
# --------------------------------------------------------------------------
def ranking() -> None:
    rows = load_registry()
    rows.append({
        "model": "frozen_simdinov2_ctx512",
        "display": "SimDINOv2-B, fused readout, frozen",
        "type": "frozen_ctx",
        "f1": 0.5059,
        "std": None,
        "seeds": None,
        "dim": 1536,
    })
    rows.sort(key=lambda r: r["f1"])

    fig, ax = plt.subplots(figsize=(9.6, max(7.4, 0.29 * len(rows) + 1.3)))
    xmin, xtext = 0.345, 0.517

    for i, r in enumerate(rows):
        colour = TYPE_COLOR[r["type"]]
        if r["std"]:
            ax.errorbar(r["f1"], i, xerr=r["std"], color=colour, lw=1.6, capsize=3, zorder=3)
        if r["seeds"]:
            ax.plot(r["seeds"], [i] * len(r["seeds"]), ls="none", marker="|",
                    ms=8, color=colour, alpha=0.75, zorder=4)
        marker = "D" if r["type"] == "frozen_ctx" else "o"
        ax.plot(r["f1"], i, marker=marker, ms=8 if marker == "D" else 7,
                color=colour, zorder=5)
        text = f"{r['f1']:.4f}"
        text += f" +/- {r['std']:.4f}" if r["std"] else "   (1 seed)"
        ax.text(xtext, i, text, va="center", ha="left", fontsize=7.4,
                color=V.INK_SOFT)

    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{pretty(r['model'], r['display'])}  ({r['dim']}d)" for r in rows],
                       fontsize=8)
    ax.set_ylim(-0.7, len(rows) - 0.35)
    ax.set_xlabel("macro F1 over the 11 classes present in the test split")
    ax.set_xlim(xmin, 0.56)
    ax.set_xticks([0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525])
    V.clean(ax, y_only=False)
    ax.grid(axis="y", color=V.GRID, lw=0.5, ls=":")
    ax.axvline(0.4805, color=V.RULE, ls="--", lw=1.0, zorder=1)

    handles = [Line2D([], [], ls="none", marker="o", color=TYPE_COLOR[t], label=TYPE_LABEL[t])
               for t in ("frozen", "ft_fresh", "ft_old", "scratch")]
    handles.append(Line2D([], [], ls="none", marker="D", color=TYPE_COLOR["frozen_ctx"],
                          label=TYPE_LABEL["frozen_ctx"]))
    handles.append(Line2D([], [], color=V.INK_SOFT, marker="|", ls="-",
                          label="+/- inter-seed SD, ticks are individual seeds"))
    handles.append(Line2D([], [], color=V.RULE, ls="--", lw=1.0,
                          label="best frozen model (0.4805)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7.6, bbox_to_anchor=(0.55, -0.05))
    ax.set_title("Ranking of the 42 canonical models and the frozen context control",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    V.savefig(fig, FIG / "fig_readme_ranking.png")


# --------------------------------------------------------------------------
# 2. Spatial context
# --------------------------------------------------------------------------
def context() -> None:
    rows = [
        ("DINOv3 ViT-B/16, frozen", 0.4712, 0.0, TYPE_COLOR["frozen"]),
        ("DINOv3 ViT-B/16, context distillation (tile only)", 0.4870, 0.0013, V.SERIES[1]),
        ("DINOv3 ViT-B/16, context fusion (neighbourhood)", 0.5080, 0.0015, V.SERIES[3]),
        ("SimDINOv2 ViT-B/16, frozen", 0.4723, 0.0, TYPE_COLOR["frozen"]),
        ("SimDINOv2 ViT-B/16, context fusion (neighbourhood)", 0.5057, 0.0072, V.SERIES[3]),
        ("SimDINOv2 ViT-B/16, fused readout, frozen", 0.5059, 0.0, TYPE_COLOR["frozen_ctx"]),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 3.3))
    for i, (label, value, std, colour) in enumerate(rows):
        if std:
            ax.errorbar(value, i, xerr=std, color=colour, lw=1.6, capsize=3, zorder=3)
        ax.plot(value, i, marker="o", ms=7, color=colour, zorder=5)
        ax.text(value + (std or 0) + 0.0015, i, f"{value:.4f}", va="center",
                fontsize=7.8, color=V.INK_SOFT)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0.465, 0.518)
    ax.set_xlabel("macro F1")
    V.clean(ax, y_only=False)
    ax.grid(axis="y", color=V.GRID, lw=0.5, ls=":")
    ax.set_title("Adding the 512 px neighbourhood to a frozen backbone",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    V.savefig(fig, FIG / "fig_readme_context.png")


# --------------------------------------------------------------------------
# 3. Latent space: spectra, ranking power, probe dimension
# --------------------------------------------------------------------------
def pair_share(metric: list[float], f1: list[float], higher_is_better: bool) -> float:
    ok = total = 0
    for i in range(len(f1)):
        for j in range(i + 1, len(f1)):
            if f1[i] == f1[j]:
                continue
            total += 1
            prod = (metric[i] - metric[j]) * (f1[i] - f1[j])
            if (prod > 0) if higher_is_better else (prod < 0):
                ok += 1
    return ok / total


def latent() -> None:
    geo_path = ROOT / "results/rapport_data/geometry_models.csv"
    f1_by_model = {m["model"]: m["f1"] for m in load_registry()}

    rows = []
    with geo_path.open() as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            if model not in f1_by_model or f1_by_model[model] < 0.4689:
                continue
            rows.append(row)

    f1 = [f1_by_model[r["model"]] for r in rows]

    separability = [
        ("Silhouette", "silhouette", True),
        ("Fisher ratio", "fisher_ratio", True),
        ("Calinski-Harabasz", "chi", True),
        ("Davies-Bouldin", "dbi", False),
        ("NC1", "nc1_canonical", False),
    ]
    spectral = [
        ("RankMe", "rankme_sigma_centered", True),
        ("Effective rank", "eff_rank_sigma2", True),
        ("Participation ratio", "participation_ratio", True),
        ("Intrinsic dimension", "intrinsic_dim_twonn", True),
    ]

    names, shares, colours = [], [], []
    for label, key, higher in separability:
        names.append(label)
        shares.append(pair_share([float(r[key]) for r in rows], f1, higher) * 100)
        colours.append(V.SERIES[0])
    for label, key, higher in spectral:
        names.append(label)
        shares.append(pair_share([float(r[key]) for r in rows], f1, higher) * 100)
        colours.append(V.SERIES[1])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.9))

    spectra = np.load(ROOT / "results/rapport_data/spectra_cache.npz")
    curves = [("dinov3_vitb16_lvd", "DINOv3-B frozen", V.SERIES[0]),
              ("dinov3_vitl16_lvd", "DINOv3-L frozen", V.SERIES[3]),
              ("dinov3_vitb16_lvd_lora_r8", "DINOv3-B LoRA r=8", V.SERIES[1]),
              ("scalemae_vitl16", "Scale-MAE", V.SERIES[4]),
              ("vitb16_scratch_old", "from scratch", "#8a8985")]
    for key, label, colour in curves:
        eig = np.asarray(spectra[key], dtype=float)
        eig = eig / eig[0]
        rank = np.arange(1, len(eig) + 1)
        ax1.plot(np.log10(rank), np.log10(eig), lw=1.6, color=colour, label=label)
    ax1.set_xlabel("eigenvalue rank (log)")
    ax1.set_ylabel("normalised eigenvalue (log)")
    ax1.legend(fontsize=6.8, frameon=False, loc="lower left")
    ax1.set_title("Broken backbones collapse early", loc="left", fontweight="bold", fontsize=9.5)
    V.clean(ax1, y_only=False)

    y = np.arange(len(names))
    ax2.barh(y, shares, color=colours, height=0.65)
    ax2.axvline(50, color=V.RULE, ls="--", lw=1.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("model pairs ordered like F1 (%)")
    for yi, value in zip(y, shares):
        ax2.text(value + 1.5, yi, f"{value:.0f}", va="center", fontsize=7.2,
                 color=V.INK_SOFT)
    ax2.set_title("Separability ranks, spectra do not", loc="left", fontweight="bold", fontsize=9.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=V.SERIES[0]),
               plt.Rectangle((0, 0), 1, 1, color=V.SERIES[1])]
    ax2.legend(handles, ["Class separability", "Spectral"], fontsize=7,
               frameon=False, loc="lower right")

    V.clean(ax2, y_only=False)
    ax2.grid(axis="y", visible=False)

    dim_k = [10, 20, 50, 100, 200, 400, 768]
    dim_f1 = [0.4081, 0.4464, 0.4629, 0.4672, 0.4674, 0.4690, 0.4715]
    ax3.plot(dim_k, dim_f1, "o-", color=V.SERIES[0], lw=1.8, ms=5)
    ax3.set_xscale("log")
    ax3.set_xlabel("principal components kept")
    ax3.set_ylabel("macro F1")
    ax3.set_ylim(0.39, 0.48)
    ax3.axvline(100, color=V.RULE, ls=":", lw=1.0)
    ax3.annotate("100 dims keep 99.1%\nof the full probe", xy=(100, 0.4672),
                 xytext=(12, 0.448), fontsize=7.2,
                 arrowprops={"arrowstyle": "->", "lw": 0.8, "color": V.INK_SOFT})
    ax3.set_title("The task uses about 100 dimensions", loc="left",
                  fontweight="bold", fontsize=9.5)
    V.clean(ax3, y_only=False)

    fig.tight_layout()
    V.savefig(fig, FIG / "fig_readme_latent.png")

    for name, share in zip(names, shares):
        print(f"  {name:24} {share:5.1f}%")


# --------------------------------------------------------------------------
# 4. Data curves
# --------------------------------------------------------------------------
def datacurve() -> None:
    def read_agg(name: str):
        x, mean, std = [], [], []
        with (ROOT / "results/datacurve_lora" / name).open() as fh:
            for row in csv.DictReader(fh):
                x.append(float(row["fraction"]))
                mean.append(float(row["f1_pres_mean"]))
                std.append(float(row["f1_pres_std"]))
        return x, mean, std

    rx, rm, rs = read_agg("results_agg.csv")
    fx, fm, fs = read_agg("frozen_probe_results_agg.csv")

    spatial = defaultdict(list)
    with (ROOT / "lora_spatial_v2/results_spatial_summary.csv").open() as fh:
        for row in csv.DictReader(fh):
            spatial[float(row["fraction_cible"])].append(float(row["f1_macro_pres_test"]))
    sx = sorted(spatial)
    sm = [float(np.mean(spatial[k])) for k in sx]
    ss = [float(np.std(spatial[k], ddof=1)) for k in sx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.6))

    ax1.errorbar(rx, rm, yerr=rs, marker="o", ms=4, lw=1.8, capsize=2,
                 color=V.SERIES[1], label="LoRA r=8, random tiles")
    ax1.errorbar(fx, fm, yerr=fs, marker="o", ms=4, lw=1.8, capsize=2,
                 color=V.SERIES[0], label="frozen probe, random tiles")
    ax1.axhline(0.4712, color=V.RULE, ls="--", lw=1.0)
    ax1.text(0.006, 0.474, "DINOv3-B frozen, full data", fontsize=6.8, color=V.RULE)
    ax1.set_xscale("log")
    ax1.set_xlabel("fraction of training tiles")
    ax1.set_ylabel("macro F1")
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    ax1.set_title("Fine-tuning needs a lot of data", loc="left",
                  fontweight="bold", fontsize=9.5)
    V.clean(ax1, y_only=False)

    ax2.errorbar(rx, rm, yerr=rs, marker="o", ms=4, lw=1.8, capsize=2,
                 color="#8a8985", label="random tiles")
    ax2.errorbar(sx, sm, yerr=ss, marker="s", ms=4, lw=1.8, capsize=2,
                 color=V.SERIES[1], label="blocked by orthomosaic")
    ax2.set_xscale("log")
    ax2.set_xlabel("fraction of training tiles")
    ax2.set_ylabel("macro F1")
    ax2.annotate("1% of the data:\n0.426 vs 0.099", xy=(0.01, 0.099), xytext=(0.021, 0.235),
                 fontsize=7.2,
                 arrowprops={"arrowstyle": "->", "lw": 0.8, "color": V.INK_SOFT})
    ax2.legend(fontsize=7, frameon=False, loc="lower right")
    ax2.set_title("The random draw flatters the result", loc="left",
                  fontweight="bold", fontsize=9.5)
    V.clean(ax2, y_only=False)

    fig.tight_layout()
    V.savefig(fig, FIG / "fig_readme_datacurve.png")


if __name__ == "__main__":
    ranking()
    context()
    latent()
    datacurve()
