#!/usr/bin/env python3
"""Figures de la section « adaptation paramétrique sur SimDINOv2-B » (Stage A).

Toutes les valeurs viennent de results/simb_stageA_probe_CANONICAL.json
(F1 canonique, reprobe mono-thread) — jamais tapées.
  - figs/peft_scaling_ladder.png  : F1 vs scaling à r fixé (r=8 : 1/2/2,83 ;
                                    r=16 : 1/2/4) — le scaling dégrade partout
  - figs/peft_rank_ladder.png     : F1 vs rang à scaling=1 (r=2/4/8/16/32)
  - figs/peft_position.png        : b611 / b05 / b911 / tous + NormTuning
Style commun via vizstyle.
"""
from __future__ import annotations
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizstyle as V
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(FIG))
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, PURPLE, RED = V.SERIES
GRAY = "#8a8a8a"

d = {m["model"]: m for m in json.load(open(os.path.join(
    ROOT, "results", "simb_stageA_probe_CANONICAL.json")))["models"]}

ANCHOR = float({m["model"]: m for m in json.load(open(os.path.join(
    ROOT, "results", "all_models_canonical_merged.json")))["models"]
    }["simdinov2_vitb16_lora"]["f1_linear_probe"])
# Ancre = canonique SimDINOv2-B LoRA r8a8 tous blocs (campagne 3models).


def bar(ax, labels, means, stds, colors, ylim, title, ylabel="F1 macro (canonique)"):
    x = range(len(labels))
    b = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", lw=0.6,
               capsize=3, width=0.58, error_kw=dict(lw=0.9))
    for i, (v, s) in enumerate(zip(means, stds)):
        ax.text(i, v + s + 0.0009, f"{v:.4f}", ha="center", fontsize=7)
    ax.axhline(ANCHOR, color=GRAY, ls=":", lw=1.0)
    ax.text(len(labels) - 0.5, ANCHOR + 0.0004, "ancre r8a8", ha="right",
            fontsize=7, color=GRAY)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel(ylabel, fontsize=8.5); ax.set_title(title, fontsize=9.5)
    ax.set_ylim(*ylim); ax.grid(axis="y", alpha=0.25, lw=0.5)


# (a) échelle de scaling à rang fixé
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=True)
for ax, r, keys, scal in zip(
        axes, (8, 16),
        (["simdinov2_vitb16_lora_r8", "simdinov2_vitb16_lora_r8_s2",
          "simdinov2_vitb16_lora_r8_rslora"],
         ["simdinov2_vitb16_lora_r16", "simdinov2_vitb16_lora_r16_s2",
          "simdinov2_vitb16_lora_r16_rslora"]),
        (["s=1", "s=2", "s=√8"], ["s=1", "s=2", "s=√16"])):
    bar(ax, scal, [d[k]["f1_linear_probe"] for k in keys],
        [d[k]["f1_std"] for k in keys], [BLUE, ORANGE, RED], (0.470, 0.484),
        f"(a{r//8}) r={r} — plus de scaling = pire" if r == 8 else f"(b) r={r}")
axes[0].set_title("(a) r=8 — plus de scaling = pire", fontsize=9.5)
axes[1].set_title("(b) r=16 — idem (rsLoRA infirmé)", fontsize=9.5)
fig.suptitle("Échelle de scaling à rang fixé — le scaling dégrade partout",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "peft_scaling_ladder.png"), dpi=220); plt.close(fig)

# (b) échelle de rang à scaling=1
fig, ax = plt.subplots(figsize=(6.6, 3.1))
_merged = {m["model"]: m for m in json.load(open(os.path.join(
    ROOT, "results", "all_models_canonical_merged.json")))["models"]}
_anchor = _merged["simdinov2_vitb16_lora"]  # r8a8 tous blocs Q/V, canonique 3models
keys = ["simdinov2_vitb16_lora_r2", "simdinov2_vitb16_lora_r4",
        "simdinov2_vitb16_lora_r16", "simdinov2_vitb16_lora_r32"]
bar(ax, ["r=2", "r=4", "r=8\n(ancre)", "r=16", "r=32"],
    [d["simdinov2_vitb16_lora_r2"]["f1_linear_probe"],
     d["simdinov2_vitb16_lora_r4"]["f1_linear_probe"],
     _anchor["f1_linear_probe"],
     d["simdinov2_vitb16_lora_r16"]["f1_linear_probe"],
     d["simdinov2_vitb16_lora_r32"]["f1_linear_probe"]],
    [d["simdinov2_vitb16_lora_r2"]["f1_std"],
     d["simdinov2_vitb16_lora_r4"]["f1_std"],
     _anchor["f1_std"],
     d["simdinov2_vitb16_lora_r16"]["f1_std"],
     d["simdinov2_vitb16_lora_r32"]["f1_std"]],
    [GREEN, GREEN, GRAY, ORANGE, RED], (0.471, 0.482),
    "Échelle de rang à scaling=1 — pente douce, pas de falaise")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "peft_rank_ladder.png"), dpi=220); plt.close(fig)

# (c) position + type + NormTuning
fig, ax = plt.subplots(figsize=(7.4, 3.1))
keys = ["simdinov2_vitb16_lora_r8_b911", "simdinov2_vitb16_lora_r8_b611",
        "simdinov2_vitb16_lora_r8_qkv", "simdinov2_vitb16_lora_r8_b05",
        "simdinov2_vitb16_norm_tuning"]
bar(ax, ["b911\n(3 derniers)", "b611\n(6 hauts)", "Q+K+V\n(tous)",
         "b05\n(6 bas)", "NormTuning\n(normes+tête)"],
    [d[k]["f1_linear_probe"] for k in keys],
    [d[k]["f1_std"] for k in keys],
    [GREEN, BLUE, PURPLE, ORANGE, MAGENTA], (0.473, 0.484),
    "Position, type et PEFT minimal — b911 au même niveau que tout, "
    "NormTuning = ancre")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "peft_position.png"), dpi=220); plt.close(fig)

print("figures écrites : peft_scaling_ladder.png peft_rank_ladder.png peft_position.png")
