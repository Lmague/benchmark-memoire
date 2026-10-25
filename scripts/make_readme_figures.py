#!/usr/bin/env python3
"""Regenerate the README figures in English.

Outputs, all written to figures/:
    fig_readme_ranking.png   all 42 canonical models, two panels
    fig_readme_context.png   spatial context, frozen vs distillation vs fusion
    fig_readme_latent.png    probe dimension and class separability
    fig_readme_datacurve.png random draw vs spatially blocked draw

Data sources are the canonical artifacts only:
    results/all_models_canonical_merged.json
    results/rapport_data/geometry_models.csv
    results/rapport_data/correlations_geometry.csv
    results/datacurve_lora/{results_agg,frozen_probe_results_agg}.csv
    lora_spatial_v2/results_spatial_summary.csv

Run from anywhere: python3 scripts/make_readme_figures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FROZEN = "#4C72B0"
FT_SSL = "#DD8452"
FT_IN = "#55A868"
ACCENT = "#C44E52"
GREY = "#8C8C8C"

TYPE_COLOR = {"frozen": FROZEN, "ft_fresh": FT_SSL, "ft_old": FT_IN}
TYPE_LABEL = {
    "frozen": "Frozen",
    "ft_fresh": "Fine-tuned, SSL init",
    "ft_old": "Fine-tuned, ImageNet init",
}

PRETTY = {
    "ctxdistill_dB_tL": "Context fusion (neighbourhood at inference)",
    "ctxdistill_dB_tSL_r8a16": "Context fusion, SimDINOv2-B r8a16",
    "ctxdistill_dB_tSL_r2a4": "Context fusion, SimDINOv2-B r2a4",
    "ctxdistill_dA_tL": "Context distillation (tile only)",
    "ctxdistill_dA_tEMA": "Context distillation with EMA teacher",
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
    if model in PRETTY:
        return PRETTY[model]
    if display:
        return display
    return model


# --------------------------------------------------------------------------
# 1. Ranking, two panels of 21 models each
# --------------------------------------------------------------------------
def ranking() -> None:
    entries = json.loads((ROOT / "results/all_models_canonical_merged.json").read_text())
    models = [m for m in entries["models"] if "explora" not in m["model"]]
    models.sort(key=lambda m: m["f1_linear_probe"])

    half = (len(models) + 1) // 2
    chunks = [models[half:], models[:half]]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.6), sharex=True)
    lo, hi = 0.34, 0.53

    for ax, chunk in zip(axes, chunks):
        y = np.arange(len(chunk))
        vals = [m["f1_linear_probe"] for m in chunk]
        errs = [m.get("f1_std") or 0.0 for m in chunk]
        cols = [TYPE_COLOR[m["type"]] for m in chunk]
        ax.barh(y, vals, xerr=errs, color=cols, height=0.72,
                error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#333333"})
        ax.set_yticks(y)
        ax.set_yticklabels([pretty(m["model"], m.get("display")) for m in chunk], fontsize=7.4)
        for yi, v, e in zip(y, vals, errs):
            ax.text(v + e + 0.003, yi, f"{v:.4f}", va="center", fontsize=6.6)
        ax.axvline(0.4805, color=ACCENT, lw=1.0, ls="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_xlabel("macro F1, 11 classes")
        ax.grid(axis="y", visible=False)

    axes[0].text(0.482, len(chunks[0]) - 0.4, "best frozen", color=ACCENT,
                 fontsize=7.5, va="bottom")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COLOR[k]) for k in ("frozen", "ft_fresh", "ft_old")]
    fig.legend(handles, [TYPE_LABEL[k] for k in ("frozen", "ft_fresh", "ft_old")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("42 canonical configurations, same linear probe on the same test split", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "fig_readme_ranking.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. Spatial context
# --------------------------------------------------------------------------
def context() -> None:
    rows = [
        ("DINOv3-B, frozen", 0.4712, FROZEN),
        ("DINOv3-B + context distillation\n(tile only)", 0.4870, FT_SSL),
        ("DINOv3-B + context fusion\n(neighbourhood at inference)", 0.5080, ACCENT),
        ("SimDINOv2-B, frozen", 0.4723, FROZEN),
        ("SimDINOv2-B + context fusion\n(neighbourhood at inference)", 0.5057, ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=[r[2] for r in rows], height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v + 0.003, yi, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlim(0.44, 0.52)
    ax.set_xlabel("macro F1, 11 classes")
    ax.grid(axis="y", visible=False)
    ax.set_title("Adding the 512 px neighbourhood to a frozen backbone")
    fig.tight_layout()
    fig.savefig(FIG / "fig_readme_context.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Latent space: probe dimension and class separability
# --------------------------------------------------------------------------
def latent() -> None:
    dim_k = [10, 20, 50, 100, 200, 400, 768]
    dim_f1 = [0.4081, 0.4464, 0.4629, 0.4672, 0.4674, 0.4690, 0.4715]

    geo_path = ROOT / "results/rapport_data/geometry_models.csv"
    reg = json.loads((ROOT / "results/all_models_canonical_merged.json").read_text())["models"]
    f1_by_model = {m["model"]: m["f1_linear_probe"] for m in reg}

    sil, f1, kinds = [], [], []
    with geo_path.open() as fh:
        for row in csv.DictReader(fh):
            model = row["model"]
            if model not in f1_by_model or not row["silhouette"]:
                continue
            sil.append(float(row["silhouette"]))
            f1.append(f1_by_model[model])
            kinds.append(row["kind"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))

    ax1.plot(dim_k, dim_f1, "o-", color=ACCENT, lw=1.6, ms=4)
    ax1.set_xscale("log")
    ax1.set_xlabel("principal components kept")
    ax1.set_ylabel("macro F1")
    ax1.set_ylim(0.39, 0.48)
    ax1.axvline(100, color=GREY, ls=":", lw=1.0)
    ax1.annotate("100 dims reach 99.1%\nof the full probe", xy=(100, 0.4672),
                 xytext=(14, 0.452), fontsize=7.5,
                 arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})
    ax1.set_title("The task uses about 100 dimensions")

    colour = {"frozen": FROZEN, "ft_fresh": FT_SSL, "ft_old": FT_IN}
    for kind in ("frozen", "ft_fresh", "ft_old"):
        xs = [s for s, k in zip(sil, kinds) if k == kind]
        ys = [v for v, k in zip(f1, kinds) if k == kind]
        if xs:
            ax2.scatter(xs, ys, s=26, color=colour[kind], label=TYPE_LABEL[kind],
                        edgecolor="white", linewidth=0.4, zorder=3)
    ax2.set_xlabel("silhouette on frozen embeddings")
    ax2.set_ylabel("macro F1")
    ax2.set_title("Class separability ranks models (Spearman $\\rho$ = 0.38)")
    ax2.legend(fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIG / "fig_readme_latent.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Data curves: random draw vs spatially blocked draw
# --------------------------------------------------------------------------
def datacurve() -> None:
    def read_agg(name: str) -> tuple[list[float], list[float], list[float]]:
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))

    for x, m, s, lab, c in (
        (rx, rm, rs, "LoRA r=8, random tiles", ACCENT),
        (fx, fm, fs, "frozen probe, random tiles", FROZEN),
    ):
        ax1.errorbar(x, m, yerr=s, marker="o", ms=3.5, lw=1.5, capsize=2, color=c, label=lab)
    ax1.set_xscale("log")
    ax1.set_xlabel("fraction of training tiles")
    ax1.set_ylabel("macro F1")
    ax1.axhline(0.4712, color=GREY, ls="--", lw=0.9)
    ax1.text(0.006, 0.474, "DINOv3-B frozen, 100%", fontsize=7, color=GREY)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    ax1.set_title("Random draw: fine-tuning needs a lot of data")

    ax2.errorbar(rx, rm, yerr=rs, marker="o", ms=3.5, lw=1.5, capsize=2,
                 color=GREY, label="random tiles")
    ax2.errorbar(sx, sm, yerr=ss, marker="s", ms=3.5, lw=1.5, capsize=2,
                 color=ACCENT, label="blocked by orthomosaic")
    ax2.set_xscale("log")
    ax2.set_xlabel("fraction of training tiles")
    ax2.set_ylabel("macro F1")
    ax2.annotate("1% of the data:\n0.426 vs 0.099", xy=(0.01, 0.099), xytext=(0.02, 0.24),
                 fontsize=7.5, arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#333333"})
    ax2.legend(fontsize=7, frameon=False, loc="lower right")
    ax2.set_title("The random draw flatters the result")

    fig.tight_layout()
    fig.savefig(FIG / "fig_readme_datacurve.png")
    plt.close(fig)


if __name__ == "__main__":
    ranking()
    context()
    latent()
    datacurve()
    for name in ("ranking", "context", "latent", "datacurve"):
        path = FIG / f"fig_readme_{name}.png"
        print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB")
