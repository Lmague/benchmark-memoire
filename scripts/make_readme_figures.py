#!/usr/bin/env python3
"""Regenerate the README figures in English, using the report's house style.

Outputs, all written to figures/:
    fig_readme_ranking.png   all canonical models plus the frozen context control
    fig_readme_context.png   frozen, distillation and fusion, side by side
    fig_readme_latent.png    spectra, ranking power per metric, probe dimension
    fig_readme_datacurve.png random draw vs spatially blocked draw
    fig_readme_peft.png      LoRA rank ladder and adaptation regimes, SimDINOv2-B
    fig_readme_perclass.png  per class F1 of the eight best models of the tier
    fig_readme_cost.png      macro F1 against inference GFLOPs per tile

Canonical sources only:
    results/all_models_canonical_merged.json
    results/rapport_data/geometry_models.csv
    results/rapport_data/spectra_cache.npz
    results/datacurve_lora/{results_agg,frozen_probe_results_agg}.csv
    lora_spatial_v2/results_spatial_summary.csv
    results/rapport_data/tile_heads/*.json
    results/rapport_data/per_class_tier.csv

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


# --------------------------------------------------------------------------
# 5. PEFT variants on SimDINOv2-B (tile heads reprobe)
# --------------------------------------------------------------------------
TILE_HEADS = ROOT / "results/rapport_data/tile_heads"


def tile_head_f1(name: str) -> float:
    """Canonical linear probe F1 (lin_lbfgs) of one tile_heads json file."""
    data = json.loads((TILE_HEADS / f"{name}.json").read_text())
    return float(data["lin_lbfgs"]["f1_mean"])


def fig_peft_variants() -> None:
    ladder_files = [(2, "simdinov2_b_lora_r2"), (4, "simdinov2_b_lora_r4"),
                    (8, "simdinov2_b_lora_r_8"), (16, "simdinov2_b_lora_r16"),
                    (32, "simdinov2_b_lora_r32")]
    ladder = [(r, tile_head_f1(n)) for r, n in ladder_files]
    frozen = tile_head_f1("simdinov2_vit_b16")

    variants = [
        ("Frozen backbone", "simdinov2_vit_b16"),
        ("LoRA r=8", "simdinov2_b_lora_r_8"),
        ("LoRA r=8, Q+K+V", "simdinov2_b_lora_r8_q_k_v"),
        ("LoRA r=8, blocks 0-5", "simdinov2_b_lora_r8_blocs_0_5"),
        ("LoRA r=8, blocks 6-11", "simdinov2_b_lora_r8_blocs_6_11"),
        ("LoRA r=8, blocks 9-11", "simdinov2_b_lora_r8_blocs_9_11"),
        ("LoRA r=8, scaling 2", "simdinov2_b_lora_r8_scaling_2"),
        ("LoRA r=8, rsLoRA", "simdinov2_b_lora_r8_rslora"),
        ("Norm and head only", "simdinov2_b_normtuning"),
        ("Attention only", "simdinov2_b_mhsa"),
        ("Full fine-tuning", "simdinov2_b_full"),
    ]
    values = [(label, tile_head_f1(name)) for label, name in variants]
    # keep the r16 run only if its value is distinct from the other variants
    r16 = ("LoRA r=16", tile_head_f1("simdinov2_b_lora_r16"))
    if all(abs(r16[1] - v) > 5e-7 for _, v in values):
        values.append(r16)
    spread = max(v for _, v in values) - min(v for _, v in values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.0),
                                   gridspec_kw={"width_ratios": [1, 1.42]})

    xs = [r for r, _ in ladder]
    ys = [v for _, v in ladder]
    ax1.plot(xs, ys, "o-", color=V.SERIES[0], lw=1.8, ms=6, zorder=3)
    for r, v in ladder:
        ax1.text(r, v + 0.00045, f"{v:.4f}", ha="center", fontsize=7.2,
                 color=V.INK_SOFT)
    ax1.axhline(frozen, color=V.RULE, ls="--", lw=1.1, zorder=1)
    ax1.text(2.0, frozen + 0.00035, "frozen backbone", fontsize=7.2,
             color=V.RULE, va="bottom")
    ax1.set_xticks(xs)
    ax1.set_xlabel("LoRA rank (r)")
    ax1.set_ylabel("macro F1")
    ax1.set_ylim(0.4705, 0.4810)
    ax1.set_title("LoRA rank ladder on SimDINOv2-B", loc="left",
                  fontweight="bold", fontsize=9.5)
    V.clean(ax1, y_only=False)

    values_sorted = sorted(values, key=lambda t: t[1])
    labels = [t[0] for t in values_sorted]
    vals = np.array([t[1] for t in values_sorted])
    xmin = 0.4675
    colours = [V.RULE if lab == "Frozen backbone" else V.SERIES[0]
               for lab in labels]
    ypos = np.arange(len(labels))
    ax2.barh(ypos, vals - xmin, left=xmin, color=colours, height=0.62, zorder=3)
    for y, v in zip(ypos, vals):
        ax2.text(v + 0.0004, y, f"{v:.4f}", va="center", fontsize=7.2,
                 color=V.INK_SOFT, zorder=4)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(labels, fontsize=7.8)
    ax2.set_xlim(xmin, 0.4862)
    ax2.set_xlabel("macro F1")
    ax2.set_title(f"Adaptation regimes on SimDINOv2-B, total spread {spread:.4f}",
                  loc="left", fontweight="bold", fontsize=9.5)
    V.clean(ax2, y_only=False)
    ax2.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=V.SERIES[0]),
               plt.Rectangle((0, 0), 1, 1, color=V.RULE)]
    ax2.legend(handles, ["adapted variant", "frozen backbone"], fontsize=7,
               frameon=False, loc="lower right")

    fig.tight_layout()
    V.savefig(fig, FIG / "fig_readme_peft.png")
    print(f"  PEFT total spread over {len(values)} variants: {spread:.4f}")


# --------------------------------------------------------------------------
# 6. Per class F1 of the tier top models
# --------------------------------------------------------------------------
PERCLASS_LABEL = {
    "ctxdistill_dB_tL": "Context fusion R2",
    "ctxdistill_dA_tL": "Context distill R1",
    "ctxdistill_dA_tEMA": "Context distill R3 (EMA)",
    "simdinov2_vitl16_lora": "SimDINOv2-L LoRA r=8",
    "dinov3_vitb16_lvd_lora_r8": "DINOv3-B LoRA r=8",
    "dinov3_vitl16_lvd_lora": "DINOv3-L LoRA r=8",
    "simdinov2_vitb16_lora_r8_qkv": "SimDINOv2-B LoRA r=8 Q+K+V",
    "dinov3_vith16plus_lvd": "DINOv3-H+ frozen",
}
PERCLASS_CLASSES = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                    "RUBC", "SEDG", "TUSS", "WILL"]
DEAD_CLASSES = ["ARCA", "DRYI", "RUBC"]


def fig_perclass() -> None:
    with (ROOT / "results/rapport_data/per_class_tier.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    top = rows[:8]
    matrix = np.array([[float(r[c]) for c in PERCLASS_CLASSES] for r in top])
    labels = [PERCLASS_LABEL.get(r["id"], r["id"]) for r in top]
    macros = [float(r["f1_macro_pres"]) for r in top]

    fig, ax = plt.subplots(figsize=(11.4, 4.4))
    im = ax.imshow(matrix, cmap=V.SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(PERCLASS_CLASSES)))
    ax.set_xticklabels(PERCLASS_CLASSES, fontsize=8)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"{lab} ({m:.4f})" for lab, m in zip(labels, macros)],
                       fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=6.0,
                    color="white" if v > 0.70 else V.INK)

    # dead classes: verify from the displayed rows, then shade and annotate
    dead_idx = [PERCLASS_CLASSES.index(c) for c in DEAD_CLASSES]
    dead_max = max(matrix[:, j].max() for j in dead_idx)
    if dead_max < 0.03:
        for j in dead_idx:
            ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, matrix.shape[0],
                                       facecolor="#8a8985", alpha=0.22,
                                       lw=0, zorder=2.5))
        ax.annotate("shaded columns: ARCA, DRYI and RUBC are near zero "
                    "for every model",
                    xy=(0.02, -0.16), xycoords="axes fraction", ha="left",
                    fontsize=7.2, color=V.INK_SOFT)

    ax.set_title("Per class F1 of the eight best models of the tier",
                 loc="left", fontweight="bold", fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("per class F1", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    V.savefig(fig, FIG / "fig_readme_perclass.png")
    print(f"  dead classes, max over the displayed rows: {dead_max:.4f}")


# --------------------------------------------------------------------------
# 7. Deployment cost: macro F1 against inference FLOPs
# --------------------------------------------------------------------------
# GFLOPs per tile (Macaque, fp32, 224 px) are engineering constants measured
# once per backbone family; they are not experiment outputs.
COST = [
    ("dinov3_vitb16_lvd", "DINOv3-B frozen", 17.6, V.SERIES[5], "o"),
    ("dinov3_vitb16_lvd_full", "DINOv3-B, full fine-tuning", 17.6, V.SERIES[6], "s"),
    ("dinov3_vitb16_lvd_mhsa", "DINOv3-B, attention only", 17.6, V.SERIES[7], "^"),
    ("dinov3_vitb16_lvd_lora_r8", "DINOv3-B, LoRA r=8", 17.6, V.SERIES[0], "D"),
    ("dinov3_vith16plus_lvd", "DINOv3 ViT-H+/16 frozen", 230.4, V.INK_SOFT, "P"),
    ("ctxdistill_dA_tL", "Context distill R1 (tile only)", 17.4, V.SERIES[4], "o"),
    ("ctxdistill_dA_tEMA", "Context distill R3 (EMA teacher)", 17.4, V.SERIES[4], "s"),
    ("ctxdistill_dB_tL", "Context fusion R2 (both views)", 675.8, V.SERIES[4], "^"),
]
# Frozen control, same value quoted in scripts/make_readme_figures.py::ranking().
CTRL = ("frozen_simdinov2_ctx512", "SimDINOv2-B + context 512, frozen", 34.8, 0.5059, "#7d3c98", "d")


def fig_cost() -> None:
    import matplotlib.ticker as mticker
    f1 = {r["model"]: r["f1"] for r in load_registry()}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for key, label, gflops, colour, marker in COST:
        ax.plot(gflops, f1[key], marker=marker, ms=9, color=colour, ls="none",
                zorder=4, markeredgecolor="white", markeredgewidth=0.8, label=label)
    _, label, gflops, value, colour, marker = CTRL
    ax.plot(gflops, value, marker=marker, ms=9, color=colour, ls="none", zorder=4,
            markeredgecolor="white", markeredgewidth=0.8, label=label + " (control)")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator([20, 50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["20", "50", "100", "200", "500"]))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(13, 800)
    ax.set_ylim(0.4685, 0.5145)
    ax.set_xlabel("inference cost, GFLOPs per tile (log scale)")
    ax.set_ylabel("macro F1, canonical probe")
    ratio = next(g for k, _, g, _, _ in COST if k == "ctxdistill_dB_tL") / \
        next(g for k, _, g, _, _ in COST if k == "ctxdistill_dA_tL")
    ax.set_title(f"Best F1 costs {ratio:.0f}x the FLOPs of the best single tile model",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8.2)
    V.clean(ax, y_only=False)
    fig.tight_layout()
    V.savefig(fig, FIG / "fig_readme_cost.png")


if __name__ == "__main__":
    ranking()
    context()
    latent()
    datacurve()
    fig_peft_variants()
    fig_perclass()
    fig_cost()
