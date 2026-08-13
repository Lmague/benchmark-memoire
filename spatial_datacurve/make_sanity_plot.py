#!/usr/bin/env python3
"""Plot sanity-check de la courbe de données spatiale v2.

Lit spatial_datacurve/manifest.json + les manifest par split, produit
spatial_datacurve/sanity_check.png :

  (a) volume réel par (fraction, seed) vs cible ;
  (b) décomposition du train en orthos / blocs (seed 0) ;
  (c) composition de classes (schéma 8-class, seed 0) par fraction.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "spatial_datacurve")
MANIFEST = os.path.join(OUT, "manifest.json")
OUT_PNG = os.path.join(OUT, "sanity_check.png")

ORTHO_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#A0522D", "#6B8E23",
    "#CD5C5C", "#4682B4", "#DAA520",
]
CLASS_8_NAMES = ["ALDE", "BIRC", "LICH", "MOSS", "PETF", "SEDG", "TUSS", "WILL"]


def main() -> None:
    with open(MANIFEST) as f:
        m = json.load(f)
    niveaux = m["niveaux"]

    tags = [n["tag"] for n in niveaux]
    n_levels = len(niveaux)
    n_seeds = len(niveaux[0]["seeds"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # ---- (a) volume réel par fraction / seed --------------------------------
    ax = axes[0]
    x = np.arange(n_levels)
    width = 0.26
    real = [[s["fraction_reelle"] * 100 for s in n["seeds"]] for n in niveaux]
    for si in range(n_seeds):
        ax.bar(x + (si - 1) * width, [r[si] for r in real], width,
               label=f"seed {si}")
    ax.axhline(0, color="k", lw=0.5)
    # cibles
    ax.plot(x, [n["cible"] * 100 for n in niveaux], "k--", lw=1.2,
            label="cible")
    for xi, n in zip(x, niveaux):
        fs = [s["fraction_reelle"] * 100 for s in n["seeds"]]
        ax.annotate(f"{min(fs):.0f}-{max(fs):.0f}%", (xi, max(fs) + 2.5),
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("Volume de train (% du split v3)")
    ax.set_title("(a) Fraction réelle atteinte par (fraction, seed)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 115)

    # ---- (b) décomposition orthos / blocs (seed 0) --------------------------
    ax = axes[1]
    ortho_names = sorted({o for n in niveaux
                          for s in n["seeds"][:1] for o in s["orthos"]},
                         key=lambda o: -sum(
                             n["seeds"][0]["n_tiles"] if o in n["seeds"][0]["orthos"] else 0
                             for n in niveaux))
    color_of = {o: ORTHO_COLORS[i % len(ORTHO_COLORS)]
                for i, o in enumerate(ortho_names)}
    bottom = np.zeros(n_levels)
    labelled: set[str] = set()
    for i, n in enumerate(niveaux):
        sel = n["seeds"][0]
        for entry in sel["selection"]:
            o, part = entry["ortho"], entry["n_tiles"]
            short = o.replace("20230724_", "").replace("20230725_", "").replace("20230728_", "").replace("_m3m", "")
            ax.bar(i, part, bottom=bottom[i], color=color_of[o],
                   label=short if o not in labelled else None,
                   edgecolor="white", lw=0.4)
            labelled.add(o)
            bottom[i] += part
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("Tuiles")
    ax.set_title("(b) Décomposition du train par unité spatiale (seed 0)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=6.5, ncol=2)

    # ---- (c) composition 8-class par fraction (seed 0) ----------------------
    ax = axes[2]
    comps = []
    for n in niveaux:
        sp = os.path.join(OUT, "splits", f"{n['tag']}_seed0", "manifest.json")
        with open(sp) as f:
            sm = json.load(f)
        comps.append([sm["classes_8cls"].get(c, 0) for c in CLASS_8_NAMES])
    arr = np.array(comps, dtype=float)
    arr = arr / arr.sum(axis=1, keepdims=True)
    bottom = np.zeros(n_levels)
    for ci, cname in enumerate(CLASS_8_NAMES):
        ax.bar(x, arr[:, ci], bottom=bottom, label=cname, lw=0.3,
               edgecolor="white")
        bottom += arr[:, ci]
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.set_ylabel("Proportion (schéma 8-class)")
    ax.set_title("(c) Composition de classes par fraction (seed 0)")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Courbe de données spatiale v2 — sanity check",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=140)
    print(f"plot écrit : {OUT_PNG}")


if __name__ == "__main__":
    main()
