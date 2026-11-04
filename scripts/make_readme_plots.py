"""Figures of the repository README, one message per figure.

Every number is read from a result file at run time, nothing is typed in by hand.

Sources (relative to the repository root):
    results/baseline_clean_2026-10-08/{per_seed.csv,statistics/pairwise_effects.csv}
    results/tile_context_2026-11-01/{canonical_probe_arms,recipe_ablation,recipe_comparisons,
        r3_vs_r2_bootstrap,gnn_canonical,strata,neighbourhood_scale}.json
    results/rapport_data/{loo_triage,dim_probe_summary}.csv
    results/datacurve_lora/{results_agg,frozen_probe_results_agg}.csv
    lora_spatial_v2/results_spatial_summary.csv

Output: figures/readme/*.png

    python scripts/make_readme_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TC = RES / "tile_context_2026-11-01"
OUT = ROOT / "figures" / "readme"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "frozen": "#4C72B0",
    "lora": "#DD8452",
    "attn": "#55A868",
    "full": "#8172B3",
    "ctxd": "#937860",
    "nbr": "#C44E52",
    "scratch": "#8C8C8C",
    "ink": "#2B2B2B",
    "grey": "#8A8A8A",
    "light": "#D9D9D9",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelcolor": COL["ink"],
    "axes.edgecolor": "#555555",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "xtick.color": COL["ink"],
    "ytick.color": COL["ink"],
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
})


def jload(path: Path):
    return json.loads(path.read_text())


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# --------------------------------------------------------------------------- clean baseline helpers
BASE = RES / "baseline_clean_2026-10-08"

LABELS = {
    "dinov3_vith16plus_lvd": "DINOv3 ViT-H+/16, frozen",
    "dinov3_vitl16_lvd": "DINOv3 ViT-L/16, frozen",
    "dinov3_vitl16_sat": "DINOv3 ViT-L/16 satellite, frozen",
    "dinov3_vitb16_lvd": "DINOv3 ViT-B/16, frozen",
    "dinov3_vits16_lvd": "DINOv3 ViT-S/16, frozen",
    "simdinov2_vitl16": "SimDINOv2 ViT-L/16, frozen",
    "simdinov2_vitb16": "SimDINOv2 ViT-B/16, frozen",
    "vitb16_imagenet": "ViT-B/16 ImageNet, frozen",
    "scalemae_vitl16": "Scale-MAE ViT-L/16, frozen",
    "satmae_vitl16": "SatMAE ViT-L/16, frozen",
    "resnet50_imagenet": "ResNet-50 ImageNet, frozen",
    "dinov3_vitl16_lvd_lora": "DINOv3-L, LoRA r=8",
    "dinov3_vitb16_lvd_lora_r8": "DINOv3-B, LoRA r=8",
    "dinov3_vits16_lvd_lora_r8": "DINOv3-S, LoRA r=8",
    "simdinov2_vitl16_lora": "SimDINOv2-L, LoRA r=8",
    "dinov3_vitb16_lvd_mhsa": "DINOv3-B, attention only",
    "dinov3_vitb16_lvd_full": "DINOv3-B, full fine-tuning",
    "vitb16_full_old": "ViT-B/16 ImageNet, full fine-tuning",
    "vitb16_mhsa_old": "ViT-B/16 ImageNet, attention only",
    "resnet50_fulft_sota": "ResNet-50 ImageNet, full fine-tuning",
    "vitb16_scratch_old": "ViT-B/16 from scratch",
    "ctxdistill_dA_tL": "Context distillation, tile only at inference",
    "ctxdistill_dA_tEMA": "Context distillation, EMA teacher, tile only",
    "ctxdistill_dB_tL": "Context fusion, fine-tuned",
    "ctxdistill_dB_tSL_r8a16": "Context fusion, SimDINOv2-B LoRA r8a16",
    "ctxdistill_dB_tSL_r2a4": "Context fusion, SimDINOv2-B LoRA r2a4",
    "simdinov2_vitb16_lora_r8_b911": "SimDINOv2-B, LoRA r=8, blocks 9-11",
    "simdinov2_vitb16_lora_r8_b611": "SimDINOv2-B, LoRA r=8, blocks 6-11",
    "simdinov2_vitb16_lora_r2": "SimDINOv2-B, LoRA r=2",
    "simdinov2_vitb16_lora_r8_qkv": "SimDINOv2-B, LoRA r=8, Q+K+V",
    "simdinov2_vitb16_lora_r4": "SimDINOv2-B, LoRA r=4",
    "simdinov2_vitb16_norm_tuning": "SimDINOv2-B, norm layers + head only",
    "simdinov2_vitb16_full": "SimDINOv2-B, full fine-tuning",
    "simdinov2_vitb16_lora": "SimDINOv2-B, LoRA r=8",
    "simdinov2_vitb16_lora_r8_b05": "SimDINOv2-B, LoRA r=8, blocks 0-5",
    "simdinov2_vitb16_lora_r16": "SimDINOv2-B, LoRA r=16",
    "simdinov2_vitb16_lora_r8_s2": "SimDINOv2-B, LoRA r=8, scaling 2",
    "simdinov2_vitb16_lora_r8_rslora": "SimDINOv2-B, LoRA r=8, rsLoRA",
    "simdinov2_vitb16_lora_r32": "SimDINOv2-B, LoRA r=32",
    "simdinov2_vitb16_lora_r16_s2": "SimDINOv2-B, LoRA r=16, scaling 2",
    "simdinov2_vitb16_lora_r16_rslora": "SimDINOv2-B, LoRA r=16, rsLoRA",
    "simdinov2_vitb16_mhsa": "SimDINOv2-B, attention only (MHSA)",
}
FROZEN_KEYS = {k for k, v in LABELS.items() if v.endswith("frozen")}


def baseline_scores() -> pd.DataFrame:
    ps = pd.read_csv(BASE / "per_seed.csv")
    g = ps.groupby("key")["f1_macro_pres"]
    out = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count(),
                        "seeds": g.apply(list)})
    return out


def baseline_vs(ref: str) -> pd.DataFrame:
    """Paired hierarchical (orthomosaic, then tile) differences of every model against `ref`."""
    p = pd.read_csv(BASE / "statistics" / "pairwise_effects.csv")
    rows = []
    for _, r in p.iterrows():
        if r.model_a == ref:
            s, other = -1, r.model_b
        elif r.model_b == ref:
            s, other = 1, r.model_a
        else:
            continue
        lo, hi = ((r.ci95_diff_low, r.ci95_diff_high) if s == 1
                  else (-r.ci95_diff_high, -r.ci95_diff_low))
        rows.append((other, s * r.effect_delta_f1, lo, hi, r.bh_q_hier, bool(r.bh_reject_hier)))
    return pd.DataFrame(rows, columns=["key", "delta", "lo", "hi", "q", "reject"]).set_index("key")


def kind_of(key: str) -> str:
    if key in FROZEN_KEYS:
        return "frozen"
    if key == "vitb16_scratch_old":
        return "scratch"
    return "ft"


KIND_COL = {"frozen": COL["frozen"], "ft": COL["lora"], "scratch": COL["scratch"]}
UPPER_BOUND = {"ctxdistill_dB_tL", "ctxdistill_dB_tSL_r8a16", "ctxdistill_dB_tSL_r2a4"}


# --------------------------------------------------------------------------- 1. leaderboard
def fig_leaderboard():
    ref = "dinov3_vith16plus_lvd"
    sc = baseline_scores()
    vs = baseline_vs(ref)
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    abl = jload(TC / "recipe_ablation.json")
    f = lambda a: arms[a]["metrics"]["f1_macro_pres"]

    tier_a_keys = list(vs.index) + [ref]
    ranked = sorted(tier_a_keys, key=lambda k: -sc.loc[k, "mean"])
    top = ranked[:18]
    refs = ["dinov3_vitl16_sat", "vitb16_imagenet", "scalemae_vitl16", "satmae_vitl16",
            "resnet50_imagenet", "vitb16_scratch_old"]
    same_backbone = "simdinov2_vitb16"
    if same_backbone not in top:
        refs = [same_backbone] + refs
    tier_a = top + sorted([k for k in refs if k not in top], key=lambda k: -sc.loc[k, "mean"])

    recipe = lambda name, v: dict(label=name, mean=v, sd=None, kind="recipe", key=None)
    ctx_ft = lambda k: dict(label=LABELS[k] + (" †" if k in UPPER_BOUND else ""),
                            mean=sc.loc[k, "mean"], sd=sc.loc[k, "sd"], kind="ft", key=None)
    tier_c = [recipe("3x3 mean + mean field", abl["R3"]["mf"]["test"]),
              recipe("3x3 mean + ICM", abl["R3"]["icm"]["test"]),
              recipe("Single tile + mean field", abl["R0"]["mf"]["test"])]
    tier_b = sorted([recipe("3x3 mean of 9 tiles", f("R3")),
                     recipe("Tile + 512 px context †", f("R2")),
                     ctx_ft("ctxdistill_dB_tL"), ctx_ft("ctxdistill_dB_tSL_r8a16"),
                     ctx_ft("ctxdistill_dB_tSL_r2a4")], key=lambda r: -r["mean"])
    tier_a_rows = [dict(label=LABELS[k], mean=sc.loc[k, "mean"],
                        sd=sc.loc[k, "sd"] if sc.loc[k, "n"] > 1 else None,
                        kind=kind_of(k), key=k) for k in tier_a]

    tiers = [("Smoothing of the predicted map, on frozen SimDINOv2-B/16 (no training)", tier_c),
             ("Neighbour tiles as input", tier_b),
             ("Single tile only", tier_a_rows)]
    ys, y = [], 0.0
    layout = []
    for title, rows in tiers:
        layout.append(("hdr", title, y))
        y -= 1.0
        for r in rows:
            layout.append(("row", r, y))
            y -= 1.0
        y -= 0.5

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 10.2), sharey=True,
                               gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.12})
    best_frozen = sc.loc[ref, "mean"]
    a.axvline(best_frozen, color=COL["grey"], ls="--", lw=1)
    b.axvline(0, color=COL["grey"], ls="--", lw=1)
    yticks, ylabels = [], []
    rows_y = [y for kind, _, y in layout if kind == "row"]
    for k, yy in enumerate(rows_y):
        if k % 2 == 0:
            a.axhspan(yy - 0.5, yy + 0.5, color="#F5F5F5", lw=0, zorder=0)
            b.axhspan(yy - 0.5, yy + 0.5, color="#F5F5F5", lw=0, zorder=0)
    for kind, item, y in layout:
        if kind == "hdr":
            a.text(0.402, y, item, fontsize=9, fontweight="bold", color=COL["ink"], va="center", zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
            continue
        r = item
        c = COL["frozen"] if r["kind"] in ("frozen", "recipe") else KIND_COL[r["kind"]]
        mk = "D" if r["kind"] == "recipe" else "o"
        yticks.append(y)
        ylabels.append(r["label"])
        x = r["mean"]
        if r["sd"]:
            a.errorbar([x], [y], xerr=[r["sd"]], fmt="none", ecolor=c, elinewidth=1.2, alpha=0.7, zorder=2)
        if x < 0.40:
            a.text(0.4015, y, f"{x:.4f} (off scale)", fontsize=8, va="center", color=c)
        else:
            a.scatter([x], [y], s=46, color=c, marker=mk, edgecolor="white", linewidth=0.7, zorder=3)
            a.text(1.015, y, f"{x:.4f}", transform=a.get_yaxis_transform(), fontsize=8.5,
                   va="center", ha="left", color=COL["ink"], family="DejaVu Sans Mono")
        if r["key"] is not None:
            if r["key"] == ref:
                b.scatter([0], [y], s=40, color=COL["ink"], marker="|", zorder=3)
                b.text(0.004, y, "reference", fontsize=8, va="center", color=COL["grey"])
            else:
                d = vs.loc[r["key"]]
                cc = c if d["reject"] else "white"
                b.hlines(y, d["lo"], d["hi"], color=c, lw=1.4, alpha=0.8, zorder=2)
                b.scatter([d["delta"]], [y], s=42, facecolor=cc, edgecolor=c, linewidth=1.3, zorder=3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COL["frozen"], label="frozen"),
               Line2D([], [], marker="o", ls="", color=COL["lora"], label="fine-tuned"),
               Line2D([], [], marker="o", ls="", color=COL["scratch"], label="from scratch"),
               Line2D([], [], marker="D", ls="", color=COL["frozen"], label="frozen SimDINOv2-B + recipe")]
    a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.065), ncol=4, fontsize=8.5, frameon=False)
    a.set_yticks(yticks)
    a.set_yticklabels(ylabels, fontsize=8.5)
    a.set_xlim(0.40, 0.55)
    a.set_ylim(min(y for _, _, y in layout) - 0.8, 0.9)
    a.grid(axis="y", visible=False)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    b.set_xlim(-0.17, 0.065)
    a.set_xlabel("macro F1, 11 classes, test split")
    b.set_xlabel("difference vs DINOv3 ViT-H+/16 (95% CI over 9 test maps)")
    a.set_title("Named ranking, three levels of input", fontsize=11.5)
    b.set_title("Which models separate from the best frozen one?", fontsize=11.5)
    n_rej = int(vs["reject"].sum())
    fig.text(0.125, 0.012,
             f"Left: clean baseline, mean of 3 seeds with ±SD bars (frozen models: one run). Diamonds: frozen SimDINOv2-B/16 "
             f"from the tile-context run, one seed.\nRight: paired bootstrap over maps then tiles, filled = survives "
             f"Benjamini-Hochberg at 5%. {n_rej} of {len(vs)} single-tile models do. Context models are outside this test family.\n"
             f"† non-deployable upper bound: the model reads the tile and a 512 px context at inference.",
             fontsize=8, color=COL["grey"], va="top")
    save(fig, "01_leaderboard.png")


# --------------------------------------------------------------------------- 2. adaptation regimes
def fig_adaptation():
    sc = baseline_scores()
    keys = [k for k in LABELS if k.startswith("simdinov2_vitb16_") and k != "simdinov2_vitb16"]
    rows = sorted(keys, key=lambda k: sc.loc[k, "mean"])
    frozen = sc.loc["simdinov2_vitb16", "mean"]
    means = [sc.loc[k, "mean"] for k in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.axvline(frozen, color=COL["frozen"], ls="--", lw=1.2, zorder=1)
    ax.text(frozen, len(rows) - 0.35, f" frozen {frozen:.4f}", color=COL["frozen"], fontsize=8.5,
            ha="left", va="bottom")
    for i, k in enumerate(rows):
        c = COL["attn"] if ("mhsa" in k or "norm" in k) else COL["full"] if k.endswith("_full") else COL["lora"]
        seeds = sc.loc[k, "seeds"]
        ax.scatter(seeds, [i] * len(seeds), s=16, color=c, alpha=0.45, zorder=2)
        ax.scatter([sc.loc[k, "mean"]], [i], s=60, color=c, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[k].replace("SimDINOv2-B, ", "") for k in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) + 0.2)
    ax.set_xlabel("macro F1, 11 classes, test split (small dots: seeds, large dot: mean)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"SimDINOv2-B/16: {len(rows)} adaptation regimes span {max(means) - min(means):.3f} F1")
    save(fig, "02_adaptation_regimes.png")


# --------------------------------------------------------------------------- 3. context ladder
def fig_context():
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    abl = jload(TC / "recipe_ablation.json")
    cmp_ = jload(TC / "recipe_comparisons.json")
    r32 = jload(TC / "r3_vs_r2_bootstrap.json")["pairs"]

    f = lambda a: arms[a]["metrics"]["f1_macro_pres"]
    rows = [
        ("Single tile", f("R0"), "feat", None),
        ("512 px context only", f("R1"), "feat", None),
        ("Tile + 512 px context", f("R2"), "feat", ("vs single tile", r32["R2-R0"])),
        ("3x3 mean of 9 tiles", f("R3"), "feat", ("vs tile + 512 px", r32["R3-R2"])),
        ("Single tile + mean field", abl["R0"]["mf"]["test"], "smooth",
         ("vs single tile", cmp_["R0MF_R0base"])),
        ("3x3 mean + ICM", abl["R3"]["icm"]["test"], "smooth", ("vs 3x3 mean", cmp_["ICM_base"])),
        ("3x3 mean + mean field", abl["R3"]["mf"]["test"], "smooth", ("vs 3x3 mean", cmp_["MF_base"])),
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.set_xlim(0.455, 0.56)
    for i, (lab, v, kind, d) in enumerate(rows):
        y = n - 1 - i
        c = COL["frozen"] if kind == "feat" else COL["nbr"]
        ax.hlines(y, 0.455, v, color=c, lw=1.2, alpha=0.35, zorder=1)
        ax.scatter([v], [y], s=70, color=c, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(v + 0.0012, y + 0.28, f"{v:.4f}", fontsize=8.5, color=c, ha="left", va="bottom")
        if d is not None:
            ref, s = d
            lo, hi = s["ci95"]
            sig = lo > 0 or hi < 0
            txt = f"{s['delta']:+.4f} {ref}   [{lo:+.3f}, {hi:+.3f}]"
            ax.text(1.02, y, txt, transform=ax.get_yaxis_transform(), fontsize=8.5, va="center",
                    color=COL["ink"] if sig else COL["grey"], fontweight="bold" if sig else "normal")
    ax.axhline(n - 4.5, color=COL["light"], lw=1.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.3)
    ax.set_xlabel("macro F1, 11 classes, test split")
    ax.grid(axis="y", visible=False)
    ax.text(1.02, n - 0.35, "paired Δ  [95% CI over 9 test maps]", transform=ax.get_yaxis_transform(),
            fontsize=8.5, color=COL["grey"], va="bottom")
    ax.text(0.559, n - 4.5 + 0.1, "features", fontsize=8.5, color=COL["frozen"], va="bottom", ha="right")
    ax.text(0.559, n - 4.5 - 0.1, "decoding on top", fontsize=8.5, color=COL["nbr"], va="top", ha="right")
    ax.set_title("Neighbour tiles, then smoothing (frozen SimDINOv2-B/16)")
    save(fig, "03_context_ladder.png")


# --------------------------------------------------------------------------- 4. where smoothing acts
def fig_smoothing():
    st = jload(TC / "strata.json")["strata"]
    sc = jload(TC / "neighbourhood_scale.json")
    names = ["8 of 8\nsame label", "6 to 7", "3 to 5", "0 to 2\n(boundary)"]
    net = [s["flips"]["net"] for s in st]
    share = [s["part"] * 100 for s in st]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.2),
                               gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.5})
    cols = [COL["attn"] if v > 0 else COL["nbr"] for v in net]
    a.bar(range(4), net, color=cols, width=0.62)
    for i, (v, s) in enumerate(zip(net, share)):
        a.text(i, v + (8 if v > 0 else -8), f"{v:+d}", ha="center", va="bottom" if v > 0 else "top",
               fontsize=9.5, fontweight="bold", color=COL["ink"])
        a.text(i, -0.02, f"{s:.1f}% of test", transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=8, color=COL["grey"])
    a.axhline(0, color=COL["ink"], lw=0.8)
    a.set_xticks(range(4))
    a.set_xticklabels(names, fontsize=8.5)
    a.tick_params(axis="x", pad=14)
    a.set_ylim(-110, 350)
    a.grid(axis="x", visible=False)
    a.set_ylabel("net tiles corrected by mean field")
    a.set_xlabel("neighbours (of 8) sharing the tile's true label", labelpad=6)
    a.set_title(f"Net {sum(net)} tiles corrected on top of the 3x3 mean")

    grid = sc["grid"]
    lab = {"+-112": "±112 px", "+-224": "±224 px", "+-336": "±336 px", "112+224": "±112 and ±224",
           "112+224+336": "all three rings"}
    y = np.arange(len(grid))[::-1]
    b.axvline(sc["base_test"], color=COL["frozen"], ls="--", lw=1.1)
    b.text(sc["base_test"] + 0.001, len(grid) - 0.45, f"no smoothing {sc['base_test']:.4f}",
           color=COL["frozen"], fontsize=8.5, va="bottom")
    for yy, g in zip(y, grid):
        b.scatter([g["test"]], [yy], s=62, color=COL["nbr"], edgecolor="white", linewidth=0.7, zorder=3)
        b.hlines(yy, sc["base_test"], g["test"], color=COL["nbr"], alpha=0.3, lw=1.2)
        b.text(g["test"] + 0.002, yy, f"{g['test']:.4f}  ({g['voisins_moyens']:.1f} nbrs)", va="center", fontsize=8.5)
    b.set_yticks(y)
    b.set_yticklabels([lab[g["scale"]] for g in grid], fontsize=9)
    b.set_xlim(0.45, 0.60)
    b.set_ylim(-0.6, len(grid) - 0.1)
    b.grid(axis="y", visible=False)
    b.set_xlabel("macro F1, single tile + mean field")
    b.set_title("Smoothing radius: nearest tiles only")
    save(fig, "04_smoothing.png")


# --------------------------------------------------------------------------- 5. learned aggregation
def fig_gnn():
    arms = jload(TC / "canonical_probe_arms.json")["records"]
    g = jload(TC / "gnn_canonical.json")["records"]
    seeds = lambda a: [s["metrics"]["f1_macro_pres"] for s in arms[a]["per_seed"]]
    params = lambda p: f"{p / 1000:.0f}k params" if p else "no parameters"

    groups = [
        ("Graph of annotated tiles only", g["R3"]["f1"], [
            ("3x3 mean", g["R3"]["f1"], None, 0, True),
            ("Attention over the 3x3 window", g["L1"]["f1"], seeds("L1"), g["L1"]["n_params"], False),
            ("Attention, neighbour positions shuffled", g["L1p"]["f1"], seeds("L1p"), g["L1p"]["n_params"], False),
            ("CNN on the 3x3 window", g["L2"]["f1"], seeds("L2"), g["L2"]["n_params"], False),
        ]),
        ("Graph with unannotated windows added", g["MEAN9"]["f1"], [
            ("Uniform mean over the graph", g["MEAN9"]["f1"], None, 0, True),
            ("GATv2, 2 layers", g["GNN2"]["f1"], g["GNN2"]["f1_std"], g["GNN2"]["n_params"], False),
            ("GATv2, 3 layers", g["GNN3"]["f1"], g["GNN3"]["f1_std"], g["GNN3"]["n_params"], False),
            ("GATv2, edges shuffled", g["GNN2rw"]["f1"], g["GNN2rw"]["f1_std"], g["GNN2rw"]["n_params"], False),
        ]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True, gridspec_kw={"hspace": 0.38})
    for ax, (title, ref, rows) in zip(axes, groups):
        ax.axvline(ref, color=COL["grey"], ls="--", lw=1)
        rows = sorted(rows, key=lambda r: r[1])
        n = len(rows)
        for i, (lab, v, spread, p, is_ref) in enumerate(rows):
            y = n - 1 - i
            c = COL["frozen"] if is_ref else COL["nbr"]
            if isinstance(spread, list):
                ax.scatter(spread, [y] * len(spread), s=16, color=c, alpha=0.45, zorder=2)
            elif spread:
                ax.errorbar([v], [y], xerr=spread, fmt="none", ecolor=c, elinewidth=1.4, capsize=3, zorder=2)
            ax.scatter([v], [y], s=70, color=c, edgecolor="white", linewidth=0.8, zorder=3,
                       marker="D" if is_ref else "o")
            ax.text(0.537, y, f"{v:.4f}   {params(p)}", va="center", fontsize=8.5, color=COL["ink"])
        ax.set_yticks(range(n))
        ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_xlim(0.455, 0.585)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=10, color=COL["ink"])
    axes[1].set_xlabel("macro F1, 11 classes, test split")
    axes[1].text(0.0, -0.32, "small dots: seeds, bars: ±SD, dashed line: the fixed-rule reference",
                 transform=axes[1].transAxes, fontsize=8, color=COL["grey"], va="top")
    fig.suptitle("Learned aggregation does not beat a plain mean", x=0.125, ha="left", y=0.98,
                 fontsize=11.5, fontweight="bold")
    save(fig, "05_learned_aggregation.png")


# --------------------------------------------------------------------------- 6. latent geometry
def fig_latent():
    d = pd.read_csv(RES / "rapport_data" / "loo_triage.csv")
    nm = {"fisher_ratio": "Fisher ratio", "silhouette": "Silhouette", "chi": "Calinski-Harabasz",
          "nc1_canonical": "NC1", "eff_rank_sigma2": "Effective rank", "stable_rank": "Stable rank",
          "participation_ratio": "Participation ratio", "rankme_sigma_raw": "RankMe",
          "knn_purity": "k-NN purity", "aniso_cosine": "Anisotropy",
          "intrinsic_dim_twonn": "Intrinsic dim. (TwoNN)", "dbi": "Davies-Bouldin",
          "logme_train": "LogME (train)", "logme_test": "LogME (test)"}
    fam = {"séparabilité": ("Class separation", COL["frozen"]), "voisinage": ("Neighbourhood", "#7FA7D6"),
           "spectrale": ("Spectral", COL["lora"]), "transférabilité": ("Transferability", COL["grey"])}
    d = d.sort_values("pair_acc_loo")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(d))
    a.barh(y, d["pair_acc_loo"] * 100, color=[fam[f][1] for f in d["family"]], height=0.66)
    a.axvline(50, color=COL["ink"], lw=1, ls="--")
    for yy, v in zip(y, d["pair_acc_loo"] * 100):
        a.text(v + 0.8, yy, f"{v:.1f}", va="center", fontsize=8.5)
    a.set_yticks(y)
    a.set_yticklabels([nm[m] for m in d["metric"]], fontsize=9)
    a.set_xlim(25, 86)
    a.set_xlabel("pairs of models ordered correctly, leave-one-model-out (%)")
    a.grid(axis="y", visible=False)
    a.text(50.6, len(d) - 0.45, "chance", fontsize=8.5, va="center", color=COL["ink"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in fam.values()]
    a.legend(handles, [n for n, _ in fam.values()], loc="lower right", fontsize=8.5)
    a.set_title("Which embedding scores rank models by F1?")

    dp = pd.read_csv(RES / "rapport_data" / "dim_probe_summary.csv")
    ks = [10, 20, 50, 100, 200, 400]
    pretty = {"dinov3_vitb16_lvd": "DINOv3-B", "dinov3_vitl16_lvd": "DINOv3-L",
              "dinov3_vitl16_sat": "DINOv3-L satellite", "simdinov2_vitb16": "SimDINOv2-B",
              "vitb16_imagenet": "ViT-B ImageNet"}
    pal = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#DD8452"]
    for c, (_, r) in zip(pal, dp.iterrows()):
        b.plot(ks, [r[f"f1_k{k}"] / r["f1_full"] * 100 for k in ks], marker="o", ms=4.5, lw=1.5,
               color=c, label=pretty[r["model"]])
    b.axhline(100, color=COL["grey"], lw=1, ls="--")
    b.set_xscale("log")
    b.set_xticks(ks)
    b.set_xticklabels([str(k) for k in ks])
    b.minorticks_off()
    b.set_ylim(65, 104)
    b.set_xlabel("principal components kept before the probe")
    b.set_ylabel("% of full-embedding F1")
    b.legend(fontsize=8.5, loc="lower right")
    b.set_title("Most of the F1 sits in ~100 dimensions")
    save(fig, "06_latent_geometry.png")


# --------------------------------------------------------------------------- 7. data curve
def fig_datacurve():
    r = pd.read_csv(RES / "datacurve_lora" / "results_agg.csv")
    fz = pd.read_csv(RES / "datacurve_lora" / "frozen_probe_results_agg.csv")
    sp = pd.read_csv(ROOT / "lora_spatial_v2" / "results_spatial_summary.csv")
    sg = sp.groupby("fraction_cible")["f1_macro_pres_test"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(r["fraction"] * 100, r["f1_pres_mean"], marker="o", ms=5, color=COL["lora"], lw=1.8,
            label="LoRA r=8, random subsets")
    ax.fill_between(r["fraction"] * 100, r["f1_pres_mean"] - r["f1_pres_std"],
                    r["f1_pres_mean"] + r["f1_pres_std"], color=COL["lora"], alpha=0.15, lw=0)
    ax.plot(fz["fraction"] * 100, fz["f1_pres_mean"], marker="s", ms=4.5, color=COL["frozen"], lw=1.8,
            label="Frozen probe, random subsets")
    ax.plot(sg["fraction_cible"] * 100, sg["mean"], marker="^", ms=6, color=COL["nbr"], lw=1.8,
            label="LoRA r=8, spatially blocked subsets")
    ax.scatter(sp["fraction_cible"] * 100, sp["f1_macro_pres_test"], s=13, color=COL["nbr"], alpha=0.4, zorder=2)
    ax.set_xscale("log")
    ticks = [0.5, 1, 5, 10, 25, 50, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("share of the 49,281 training tiles")
    ax.set_ylabel("macro F1, 11 classes, test split")
    ax.set_ylim(0, 0.52)
    ax.legend(loc="lower right", fontsize=9)
    comp = pd.read_csv(ROOT / "spatial_datacurve" / "class_composition.csv")
    c1 = comp[comp["fraction_cible"] == 0.01]["n_classes_8cls_presentes"]
    ax.annotate(f"blocked 1% subsets hold only {c1.min()} to {c1.max()}\nof the 8 main classes",
                xy=(1, sg.loc[sg["fraction_cible"] == 0.01, "mean"].iloc[0]), xytext=(1.5, 0.25),
                fontsize=8.5, color=COL["nbr"], arrowprops=dict(arrowstyle="-", color=COL["nbr"], lw=0.8))
    ax.set_title("Label efficiency depends on how the subset is drawn")
    save(fig, "07_data_curve.png")



# --------------------------------------------------------------------------- 8-class view
DROP8 = ["ARCA", "DRYI", "RUBC"]
KEEP8 = [c for c in ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF", "RUBC", "SEDG", "TUSS", "WILL"]
         if c not in DROP8]


def readme_scores_8cls() -> pd.DataFrame:
    """Macro F1 over the 8 kept classes, same 11-class classifier, full test split.

    Read from the committed per-class table, so a clone can rebuild it. The convention
    matches f1_macro_8cls_test in the run metrics: predictions on the 321 ARCA/DRYI/RUBC
    test tiles count as errors, and the three classes are not averaged.
    """
    pc = pd.read_csv(BASE / "per_class_f1.csv")
    pc = pc[pc["class"].isin(KEEP8)]
    per_probe = (pc.groupby(["key", "probe_id"])["f1_test"].mean()
                   .reset_index().rename(columns={"f1_test": "f8"}))
    seeds = pd.read_csv(BASE / "per_seed.csv")[["id", "f1_macro_pres"]]
    per_probe = per_probe.merge(seeds, left_on="probe_id", right_on="id", how="left")
    assert per_probe["f1_macro_pres"].notna().all(), "unmatched probe id"
    out = per_probe.groupby("key").agg(
        f11=("f1_macro_pres", "mean"), f8=("f8", "mean"),
        sd8=("f8", lambda v: v.std(ddof=1) if len(v) > 1 else np.nan),
        n=("f8", "size")).reset_index()
    out = out[out["key"].isin(LABELS)]
    assert len(out) == 42, len(out)
    return out.set_index("key")


def arms_scores() -> pd.DataFrame:
    """11-class and 8-class macro F1 of the context, decoding and aggregation arms.

    Read from the committed table results/tile_context_2026-11-01/arms_8cls.csv, built from
    the saved predictions with the same 8-class convention as readme_scores_8cls().
    """
    f = TC / "arms_8cls.csv"
    if not f.exists():
        return pd.DataFrame(columns=["label", "f11", "f8"])
    return pd.read_csv(f)


def fig_classes8():
    sc = readme_scores_8cls()
    from scipy.stats import spearmanr, kendalltau
    rho = spearmanr(sc.f11, sc.f8)
    r11 = sc.f11.rank(ascending=False)
    r8 = sc.f8.rank(ascending=False)
    move = (r8 - r11)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.28})
    frozen = [k for k in sc.index if k in FROZEN_KEYS]
    ft = [k for k in sc.index if k not in FROZEN_KEYS]
    a.scatter(sc.loc[frozen, "f11"], sc.loc[frozen, "f8"], s=46, color=COL["frozen"],
              edgecolor="white", linewidth=0.7, label="frozen", zorder=3)
    a.scatter(sc.loc[ft, "f11"], sc.loc[ft, "f8"], s=46, color=COL["lora"],
              edgecolor="white", linewidth=0.7, label="fine-tuned", zorder=3)
    a.scatter([sc.loc["dinov3_vith16plus_lvd", "f11"]], [sc.loc["dinov3_vith16plus_lvd", "f8"]],
              s=120, facecolor="none", edgecolor=COL["ink"], linewidth=1.1, zorder=4,
              label="best frozen (DINOv3-H+)")
    labels = {"ctxdistill_dB_tL": (-7, 8, False), "dinov3_vitb16_lvd_full": (34, -30, True),
              "vitb16_scratch_old": (6, 0, False)}
    for k, (dx, dy, arrow) in labels.items():
        a.annotate(LABELS[k], (sc.loc[k, "f11"], sc.loc[k, "f8"]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=8, color=COL["ink"],
                   ha="left" if dx > 0 else "right",
                   arrowprops=dict(arrowstyle="-", lw=0.8, color=COL["grey"]) if arrow else None)
    arms = arms_scores()
    if len(arms):
        a.scatter(arms.f11, arms.f8, s=58, marker="D", color=COL["nbr"], edgecolor="white",
                  linewidth=0.7, zorder=3, label="probe-arms, decoding, aggregation")

    a.set_xlim(0.345, 0.548)
    a.set_xlabel("macro F1, 11 classes")
    a.set_ylabel("macro F1, 8 classes")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=8.5)
    a.text(0.98, 0.03, f"Spearman {rho.statistic:.3f}, Kendall {kendalltau(sc.f11, sc.f8).statistic:.3f}.\n"
           f"Median gain {(sc.f8 - sc.f11).median():+.3f}.", transform=a.transAxes,
           fontsize=8.5, va="bottom", ha="right", color=COL["grey"])
    a.set_title("8 classes instead of 11")

    buckets = [0, 1, 2]
    counts = [int((move.abs() == k).sum()) for k in buckets] + [int((move.abs() >= 3).sum())]
    b.bar(range(4), counts, color=[COL["grey"]] * 3 + [COL["nbr"]], width=0.62)
    for i, c in enumerate(counts):
        b.text(i, c + 0.4, str(c), ha="center", va="bottom", fontsize=10, fontweight="bold",
               color=COL["nbr"] if i == 3 else COL["ink"])
    b.set_xticks(range(4))
    b.set_xticklabels(["0", "1", "2", "3 or more"])
    b.set_xlabel("ranks moved between the two schemas")
    b.set_ylabel("number of models")
    b.set_ylim(0, max(counts) * 1.28)
    b.grid(axis="x", visible=False)
    big = move[move.abs() >= 3].sort_values(key=lambda x: -x.abs())
    txt = "\n".join(f"{LABELS[k]}: {int(v):+d}" for k, v in big.items())
    b.text(0.98, 0.72, txt, transform=b.transAxes, ha="right", va="top", fontsize=8.5, color=COL["ink"])
    b.set_title("Ranking is preserved for 40 of 42")
    save(fig, "08_classes_8.png")



# --------------------------------------------------------------------------- 9. multi-backbone
MB_LABELS = {
    "simdinov2_vitb16": "SimDINOv2-B, frozen",
    "dinov3_vitb16_lvd": "DINOv3-B, frozen",
    "dinov3_vitl16_lvd": "DINOv3-L, frozen",
    "dinov3_vith16plus_lvd": "DINOv3-H+, frozen",
    "simdinov2_vitl16": "SimDINOv2-L, frozen",
    "dinov3_vitl16_lvd_lora_s0": "DINOv3-L + LoRA r=8",
    "simdinov2_vitb16_lora_r8a16_s0": "SimDINOv2-B + LoRA r8a16",
}


def fig_multibackbone():
    f = TC / "multibackbone_smoothing.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d["label"] = d["backbone"].map(MB_LABELS).fillna(d["backbone"])
    d = d.sort_values("R3_MF")
    ps = pd.read_csv(BASE / "per_seed.csv").groupby("key")["f1_macro_pres"].mean()
    ref = max(ps[k] for k in LABELS if k not in UPPER_BOUND)
    y = np.arange(len(d))

    fig, (a, b) = plt.subplots(1, 2, figsize=(12.8, 5.4), sharey=True,
                               gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.08})
    a.axvline(ref, color=COL["grey"], ls="--", lw=1)
    series = [("R0", "single tile", "o", COL["frozen"], 34),
              ("R0_MF", "single tile + mean field", "o", "#7FA7D6", 34),
              ("R3", "3x3 mean", "D", COL["lora"], 38),
              ("R3_MF", "3x3 mean + mean field", "D", COL["nbr"], 44)]
    for col, lab, mk, c, sz in series:
        a.scatter(d[col], y, marker=mk, s=sz, color=c, edgecolor="white", linewidth=0.6,
                  label=lab, zorder=3)
    a.set_yticks(y)
    a.set_yticklabels(d["label"], fontsize=9)
    a.set_ylim(-0.6, len(d) - 0.25)
    xlo = min(d[["R0", "R0_MF", "R3", "R3_MF"]].min()) - 0.008
    xhi = max(d[["R0", "R0_MF", "R3", "R3_MF"]].max()) + 0.012
    a.set_xlim(xlo, xhi)
    a.set_xlabel("macro F1, 11 classes, test split")
    a.grid(axis="y", visible=False)
    h, l = a.get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=COL["grey"], ls="--", lw=1))
    l.append(f"best single-tile model ({ref:.4f})")
    a.set_title("Context and smoothing on 7 backbones")

    b.axvline(0, color=COL["grey"], ls="--", lw=1)
    b.errorbar(d["d_mf_R0"], y - 0.13, xerr=[d["d_mf_R0"] - d["lo_mf_R0"], d["hi_mf_R0"] - d["d_mf_R0"]],
               fmt="o", ms=5, color="#4C72B0", ecolor="#7FA7D6", elinewidth=1.3, capsize=2,
               label="mean field on a single tile", zorder=3)
    if "d_mf_R3" in d:
        b.errorbar(d["d_mf_R3"], y + 0.13, xerr=[d["d_mf_R3"] - d["lo_mf_R3"], d["hi_mf_R3"] - d["d_mf_R3"]],
                   fmt="D", ms=5, color=COL["nbr"], ecolor=COL["nbr"], elinewidth=1.3, capsize=2,
                   label="mean field on the 3x3 mean", zorder=3)
    blo = min(d[["lo_mf_R0", "lo_mf_R3"]].min()) - 0.004
    bhi = max(d[["hi_mf_R0", "hi_mf_R3"]].max()) + 0.004
    b.set_xlim(blo, bhi)
    b.set_ylim(-0.6, len(d) - 0.25)
    b.grid(axis="y", visible=False)
    b.set_xlabel("paired delta vs the same features without smoothing")
    b.set_title("Smoothing gain, 95% CI over 9 maps")
    h2, l2 = b.get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, 0.07))
    fig.subplots_adjust(bottom=0.3)
    fig.text(0.5, 0.015, "Extended grids: beta 0.1 to 16, 1 to 12 iterations, chosen on validation. "
             "The two fine-tuned rows use seed 0 only.", fontsize=8, color=COL["grey"],
             va="bottom", ha="center")
    save(fig, "09_multibackbone.png")

if __name__ == "__main__":
    fig_leaderboard()
    fig_adaptation()
    fig_context()
    fig_smoothing()
    fig_gnn()
    fig_latent()
    fig_datacurve()
    fig_classes8()
    fig_multibackbone()
