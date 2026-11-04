#!/usr/bin/env python3
"""Regenerate the README figures for tile context, graph aggregation and smoothing.

Outputs, all written to figures/:
    fig_readme_recipe.png     pipeline schematic and the readout ladder
    fig_readme_gnn.png        learned aggregation vs plain averaging, canonical split
    fig_readme_smoothing.png  before/after map, neighbourhood scale, strata diagnostic

Sources (canonical split, frozen SimDINOv2 ViT-B/16, 17,598 test tiles):
    research/gnn_grid_2026-10-31/out/decoding/ablation.json
    research/gnn_grid_2026-10-31/out/decoding/cmp_*.json
    research/gnn_grid_2026-10-31/out/decoding/scales.json
    research/gnn_grid_2026-10-31/out/decoding/strata.json
    research/gnn_grid_2026-10-31/out/gnn/*/metrics.json
    research/gnn_grid_2026-10-31/out/gnn/cmp_*.json
    research/gnn_grid_2026-10-31/out/canon/summary_canonical.json
    research/gnn_grid_2026-10-31/out/canonical_split.json
    research/gnn_grid_2026-10-31/out/decoding/preds/{R3_base,R3_MF}.npy

The copies under results/tile_context_2026-11-01/ hold the scalar results only.
Run from anywhere: python3 scripts/make_readme_context_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "research/gnn_grid_2026-10-31/out"
RES = ROOT / "results/tile_context_2026-11-01"

sys.path.insert(0, str(ROOT / "rapport_bouguessa" / "figs"))
import vizstyle as V  # noqa: E402

V.setup()

INK, SOFT, RULE = V.INK, V.INK_SOFT, V.RULE
BLUE, ORANGE, PURPLE = V.SERIES[0], V.SERIES[1], V.SERIES[6]
GREEN, RED, GREY = V.SERIES[2], V.SERIES[7], "#8a8985"

CLASS_NAMES = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
               "RUBC", "SEDG", "TUSS", "WILL"]
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                "#4a3aa7", "#e34948", "#7f7f7f", "#17becf", "#bcbd22"]


def load(name: str) -> dict:
    return json.loads((RES / name).read_text())


def box(ax, x, y, w, h, title, detail, colour):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor="white", edgecolor=colour, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=8.4, color=INK, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h * 0.28, detail, ha="center", va="center",
            fontsize=7.2, color=SOFT, zorder=3)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10,
                                 color=SOFT, lw=1.1, zorder=1))


def recipe() -> None:
    ab = load("recipe_ablation.json")
    arms = load("canonical_probe_arms.json")["records"]
    cmp_ = load("recipe_comparisons.json")

    r0 = ab["R0"]["base_test"]
    r1 = arms["R1"]["metrics"]["f1_macro_pres"]
    r2 = arms["R2"]["metrics"]["f1_macro_pres"]
    r3 = ab["R3"]["base_test"]
    r0_mf = ab["R0"]["mf"]["test"]
    r3_icm = ab["R3"]["icm"]["test"]
    r3_mf = ab["R3"]["mf"]["test"]

    fig = plt.figure(figsize=(10.4, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.82, 2.2], hspace=0.18)
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")

    # 3x3 neighbourhood icon
    cell = 0.052
    x0, y0 = 0.018, 0.16
    for i in range(3):
        for j in range(3):
            centre = (i == 1 and j == 1)
            ax0.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell,
                                    facecolor="#cde2fb" if centre else "#eef1f5",
                                    edgecolor=BLUE if centre else "#c9c8c4", lw=0.9))
    ax0.text(x0 + 1.5 * cell, y0 - 0.10, "9 tiles, 224 px\n112 px stride",
             ha="center", va="top", fontsize=7.2, color=SOFT)

    bw, gap = 0.135, 0.032
    yb, bh = 0.24, 0.46
    xs = [0.215, 0.385, 0.555, 0.725]
    labels = [
        ("mean", "9 x 768-d vectors\n1 vector per tile"),
        ("linear probe", "multinomial lbfgs\nshared C grid"),
        ("probabilities", "one 11-class row\nper tile"),
        ("mean-field", "5 sparse iterations\n0 parameters"),
    ]
    colours = [BLUE, PURPLE, PURPLE, ORANGE]
    for x, (title, detail), colour in zip(xs, labels, colours):
        box(ax0, x, yb, bw, bh, title, detail, colour)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax0, a + bw + 0.004, b - 0.004, yb + bh / 2)
    arrow(ax0, x0 + 3 * cell + 0.008, xs[0] - 0.006, yb + bh / 2)
    ax0.text(xs[-1] + bw + 0.012, yb + bh * 0.62, f"{r3_mf:.4f}", fontsize=12,
             color=ORANGE, fontweight="bold", va="center")
    ax0.text(xs[-1] + bw + 0.012, yb + bh * 0.28, "macro F1", fontsize=7.2,
             color=SOFT, va="center")

    ax = fig.add_subplot(gs[1])
    rows = [
        ("Single tile (R0)", r0, GREY, ""),
        ("512 px context only (R1)", r1, GREY, ""),
        ("Tile + 512 px fusion (R2)", r2, PURPLE, ""),
        ("3x3 mean, 9 tiles (R3)", r3, BLUE,
         "paired test: p = 0.20"),
        ("3x3 mean + ICM", r3_icm, ORANGE, f"vs R3: +{r3_icm - r3:.4f}, p = 0.011"),
        ("Single tile + mean field", r0_mf, ORANGE,
         f"vs R0: +{r0_mf - r0:.4f}, p = 0.001"),
        ("3x3 mean + mean field", r3_mf, ORANGE,
         f"vs R3: +{r3_mf - r3:.4f}, p = 0.010"),
    ]
    ypos = np.arange(len(rows))
    xmin = 0.460
    ax.barh(ypos, [v - xmin for _, v, _, _ in rows], left=xmin,
            color=[c for _, _, c, _ in rows], height=0.62, zorder=3)
    for y, (label, value, colour, delta) in zip(ypos, rows):
        ax.text(value + 0.0012, y, f"{value:.4f}", va="center", fontsize=8.0,
                color=INK, fontweight="bold", zorder=4)
        if delta:
            ax.text(0.548, y, delta, va="center", fontsize=7.2, color=SOFT)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(xmin, 0.594)
    ax.set_xlabel("macro F1 over the 11 classes, 17,598 test tiles")
    V.clean(ax, y_only=False)
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=GREY, label="single view"),
               Patch(facecolor=PURPLE, label="tile + 512 px fusion"),
               Patch(facecolor=BLUE, label="mean of 9 tiles"),
               Patch(facecolor=ORANGE, label="mean-field or ICM smoothing")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              fontsize=7.4, ncol=4)
    ax.set_title("The neighbourhood helps, the decoding step does most of the work",
                 loc="left", fontsize=10.5, fontweight="bold")

    fig.savefig(FIG / "fig_readme_recipe.png", dpi=V.DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  recipe ladder: R0 {r0:.4f} -> R3 {r3:.4f} -> R3+MF {r3_mf:.4f}")


def gnn() -> None:
    rec = load("gnn_canonical.json")["records"]
    order = [
        ("R0", "Single tile", GREY),
        ("R2", "Tile + 512 px fusion", PURPLE),
        ("R3", "3x3 mean, 9 tiles", BLUE),
        ("L1", "Attention, 1 hop", ORANGE),
        ("L1p", "Attention, shuffled positions", ORANGE),
        ("L2", "CNN, same budget", ORANGE),
        ("GNN2", "GATv2, 2 layers", GREEN),
        ("GNN3", "GATv2, 3 layers", GREEN),
        ("MEAN9", "Uniform mean, full graph", GREEN),
        ("GNN2rw", "GATv2, edges shuffled", GREEN),
    ]
    vals = [rec[k]["f1"] for k, _, _ in order]
    ypos = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    xmin = 0.455
    ax.barh(ypos, [v - xmin for v in vals], left=xmin,
            color=[c for _, _, c in order], height=0.62, zorder=3)
    for y, (key, label, _), value in zip(ypos, order, vals):
        params = rec[key].get("n_params") or 0
        ax.text(value + 0.0012, y, f"{value:.4f}", va="center", fontsize=8.0,
                color=INK, fontweight="bold")
        ax.text(xmin + 0.0010, y, f"{params / 1000:.0f}k" if params else "0",
                va="center", fontsize=6.8, color="white" if value - xmin > 0.03 else SOFT)

    ax.set_yticks(ypos)
    ax.set_yticklabels([label for _, label, _ in order], fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(xmin, 0.536)
    ax.set_xlabel("macro F1, canonical split")
    V.clean(ax, y_only=False)
    ax.grid(axis="y", visible=False)
    handles = [Patch(facecolor=GREY, label="no learned aggregation"),
               Patch(facecolor=PURPLE, label="fusion with 512 px context"),
               Patch(facecolor=BLUE, label="plain mean of 9 tiles"),
               Patch(facecolor=ORANGE, label="learned on the 3x3 grid"),
               Patch(facecolor=GREEN, label="full graph, 120,118 nodes")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=7.2, ncol=3)
    ax.text(0.0, -0.29,
            "Params are printed inside each bar. GATv2 vs uniform mean: -0.0033 (p = 0.64) and "
            "+0.0009 (p = 0.82).\nGATv2 vs 3x3 mean: -0.0314 (p = 0.0004). "
            "Shuffling neighbour positions does not hurt attention.",
            transform=ax.transAxes, fontsize=7.4, color=SOFT, va="top")
    ax.set_title("Learned aggregation does not beat a plain average",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIG / "fig_readme_gnn.png", dpi=V.DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  GNN arms: " + ", ".join(f"{k} {rec[k]['f1']:.4f}" for k, _, _ in order))


def smoothing_map() -> None:
    split = json.loads((OUT / "canonical_split.json").read_text())
    test = [t for t in split["tiles"] if t["split"] == "test"]
    ortho = "20230724_sedge_mosstundra22_m3m"
    rows = sorted({t["tile_row_off"] for t in test if t["ortho"] == ortho})
    cols = sorted({t["tile_col_off"] for t in test if t["ortho"] == ortho})
    ri = {v: i for i, v in enumerate(rows)}
    ci = {v: i for i, v in enumerate(cols)}

    idx = [i for i, t in enumerate(test) if t["ortho"] == ortho]
    sub = [test[i] for i in idx]
    y = np.array([t["label"] for t in sub])
    base = np.load(OUT / "decoding/preds/R3_base.npy")[idx]
    smooth = np.load(OUT / "decoding/preds/R3_MF.npy")[idx]

    def grid(values, fill=np.nan):
        g = np.full((len(rows), len(cols)), fill, dtype=float)
        for t, v in zip(sub, values):
            g[ri[t["tile_row_off"]], ci[t["tile_col_off"]]] = v
        return g

    gb, gm = grid(base), grid(smooth)
    change = np.full(gb.shape, np.nan)
    for t, b, m, truth in zip(sub, base, smooth, y):
        r, c = ri[t["tile_row_off"]], ci[t["tile_col_off"]]
        change[r, c] = 2 if (b != truth and m == truth) else 1 if (b == truth and m != truth) else 0

    extent = (cols[0], cols[-1] + 224, rows[-1] + 224, rows[0])
    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 11.5), 11)

    fig = plt.figure(figsize=(11.6, 8.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.55, 1.0], hspace=0.34, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    for ax, data, title in ((ax1, gb, "3x3 mean, argmax"),
                            (ax2, gm, "3x3 mean + mean field")):
        ax.imshow(data, cmap=cmap, norm=norm, extent=extent, interpolation="nearest",
                  aspect="equal")
        ax.set_title(title, loc="left", fontsize=9.2, fontweight="bold")
        ax.set_xlabel("column offset, px", fontsize=7.6)
        ax.set_ylabel("row offset, px", fontsize=7.6)
        ax.tick_params(labelsize=6.6)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    cmap_ch = ListedColormap(["#eeedea", RED, GREEN])
    ax3.imshow(change, cmap=cmap_ch, vmin=-0.5, vmax=2.5, extent=extent,
               interpolation="nearest", aspect="equal")
    net = int((change == 2).sum() - (change == 1).sum())
    ax3.set_title(f"Changed tiles: +{(change == 2).sum()} right, "
                  f"-{(change == 1).sum()} wrong", loc="left", fontsize=9.2,
                  fontweight="bold")
    ax3.set_xlabel("column offset, px", fontsize=7.6)
    ax3.set_ylabel("row offset, px", fontsize=7.6)
    ax3.tick_params(labelsize=6.6)
    ax3.grid(False)
    for s in ax3.spines.values():
        s.set_visible(False)
    ax3.legend(handles=[Patch(facecolor=GREEN, label="wrong to right"),
                        Patch(facecolor=RED, label="right to wrong"),
                        Patch(facecolor="#eeedea", label="unchanged")],
               loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
               fontsize=7.0, frameon=False)
    fig.text(0.5, 0.955, f"One test orthomosaic: {ortho}, {len(sub):,} tiles, "
             f"net {net:+d} corrected", ha="center", fontsize=9.4, color=INK,
             fontweight="bold")

    # neighbourhood scale, R0 probabilities smoothed at each scale
    sc = load("neighbourhood_scale.json")
    ax4 = fig.add_subplot(gs[1, 0:3])
    labels = ["+/-112", "+/-224", "+/-336", "112+224", "112+224\n+336"]
    vals = [g["test"] for g in sc["grid"]]
    xpos = np.arange(len(vals))
    ax4.plot(xpos, vals, "o-", color=ORANGE, lw=1.8, ms=6, zorder=3)
    for x, v in zip(xpos, vals):
        ax4.text(x, v + 0.0030, f"{v:.4f}", ha="center", fontsize=7.4, color=SOFT)
    ax4.axhline(sc["base_test"], color=RULE, ls="--", lw=1.0, zorder=1)
    ax4.text(0.05, sc["base_test"] + 0.0012, "single tile, no smoothing",
             fontsize=6.8, color=RULE)
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(labels, fontsize=7.2)
    ax4.set_ylim(0.468, 0.539)
    ax4.set_ylabel("macro F1")
    ax4.set_xlabel("neighbourhood radius, px")
    ax4.set_title("Only the closest ring helps", loc="left", fontsize=9.2,
                  fontweight="bold")
    V.clean(ax4, y_only=False)

    # strata diagnostic
    st = load("strata.json")["strata"]
    ax5 = fig.add_subplot(gs[1, 3:6])
    names = ["interior\n8/8", "6 to 7", "mixed\n3 to 5", "boundary\n0 to 2"]
    nets = [s["flips"]["net"] for s in st]
    shares = [s["part"] for s in st]
    xpos = np.arange(len(st))
    ax5.bar(xpos, nets, color=[GREEN if n > 0 else RED for n in nets], width=0.62,
            zorder=3)
    for x, n, share in zip(xpos, nets, shares):
        ax5.text(x, n + (16 if n > 0 else -22), f"{n:+d}", ha="center",
                 fontsize=8.0, color=INK, fontweight="bold")
        ax5.text(x, -118, f"{share * 100:.0f}% of\ntest tiles", ha="center",
                 fontsize=6.6, color=SOFT)
    ax5.axhline(0, color=RULE, lw=0.9)
    ax5.set_xticks(xpos)
    ax5.set_xticklabels(names, fontsize=7.2)
    ax5.set_ylim(-140, 345)
    ax5.set_ylabel("net tiles corrected")
    ax5.set_title("Gain comes from mixed neighbourhoods", loc="left",
                  fontsize=9.2, fontweight="bold")
    V.clean(ax5, y_only=False)

    handles = [Patch(facecolor=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=11, fontsize=6.8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.savefig(FIG / "fig_readme_smoothing.png", dpi=V.DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  map: {ortho}, {len(sub)} tiles, net {net:+d}")


def artifact_budget() -> None:
    """README variant of the manuscript artifact-budget figure.

    The artifact amplitudes and effects E1 to E9 are imported from
    scripts/rapport/make_artifact_budget_figure.py, so the two figures cannot drift.
    E10 to E12 are the tile-context effects measured on the canonical split.
    """
    import importlib.util

    path = ROOT / "scripts/rapport/make_artifact_budget_figure.py"
    spec = importlib.util.spec_from_file_location("artifact_budget", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ab = load("recipe_ablation.json")
    r0 = ab["R0"]["base_test"]
    r3 = ab["R3"]["base_test"]
    effects = list(mod.EFFECTS)
    effects += [
        ("E10  3x3 mean vs single tile", r3 - r0),
        ("E11  3x3 mean + mean field\nvs 3x3 mean (paired +0.0152)", ab["R3"]["mf"]["test"] - r3),
        ("E12  Single tile + mean field\nvs single tile (paired +0.0578)", ab["R0"]["mf"]["test"] - r0),
    ]

    labels = [n for n, _ in mod.ARTIFACTS] + [n for n, _ in effects]
    vals = [v for _, v in mod.ARTIFACTS] + [v for _, v in effects]
    colors = ["#c1272d"] * len(mod.ARTIFACTS) + ["#2b6cb0"] * len(effects)
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    ax.barh(y, vals, color=colors, alpha=0.88, height=0.68, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.3)
    for yy, v in zip(y, vals):
        ax.text(v * 1.07, yy, f"{v:.4f}", va="center", ha="left", fontsize=7.8,
                fontweight="bold", zorder=4)
    ax.axvline(0.0080, color="#3f3f3f", ls="--", lw=1.3, zorder=2)
    ax.text(0.0080, y.max() + 1.05, "inter-seed s.d. = 0.0080", fontsize=7.8,
            color="#3f3f3f", style="italic", va="center", ha="center")
    ax.axhline(y[len(mod.ARTIFACTS)] + 0.5, color="#9a9a9a", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.2)
    ax.set_xlabel("macro F1 amplitude, 11 classes (log scale)")
    ax.grid(axis="x", which="both", ls=":", lw=0.6, alpha=0.45, zorder=0)
    handles = [Patch(facecolor="#c1272d", alpha=0.88),
               Patch(facecolor="#2b6cb0", alpha=0.88)]
    ax.legend(handles, ["Evaluation artifact", "Reported effect"],
              loc="lower right", fontsize=8.4, frameon=True)
    ax.set_ylim(y.min() - 0.6, y.max() + 1.6)
    ax.set_title("Evaluation artifacts against the effects measured on this benchmark",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig_readme_artifact_budget.png", dpi=V.DPI,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  artifact budget: {len(mod.ARTIFACTS)} artifacts, {len(effects)} effects")


if __name__ == "__main__":
    recipe()
    gnn()
    smoothing_map()
    artifact_budget()
