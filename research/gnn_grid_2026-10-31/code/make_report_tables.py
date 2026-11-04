#!/usr/bin/env python3
"""Construit les tableaux du REPORT et applique mécaniquement les critères G1/G2/G3.

Lit out/arms/*.json + out/bootstrap.json + out/grid_stats.json. N'invente rien : toute
valeur affichée provient d'un fichier. Sortie : out/tables.md et out/decision.json.

Usage : python code/make_report_tables.py --out-dir out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

DESC = {
    "R0": "tuile seule (sonde)",
    "R1": "contexte 512 seul (sonde)",
    "R2": "fusion [tuile ; ctx512] (sonde) — comparateur",
    "R3": "3×3 mean-pool (sonde)",
    "L1": "3×3 attention apprise + pos. emb",
    "L2": "3×3 CNN (contrôle)",
    "L1p": "3×3 attention, positions permutées",
    "L1b": "3×3 attention, CE pondérée",
    "L3": "5×5 attention",
    "L4": "7×7 attention",
}


def fmt(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(root, "out"))
    a = ap.parse_args()
    arms = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "arms", "*.json"))):
        r = json.load(open(p))
        arms[r["arm"]] = r
    boot = {}
    bp = os.path.join(a.out_dir, "bootstrap.json")
    if os.path.exists(bp):
        boot = json.load(open(bp))
    grid = {}
    gp = os.path.join(a.out_dir, "grid_stats.json")
    if os.path.exists(gp):
        grid = json.load(open(gp))

    r2 = arms.get("R2", {}).get("metrics", {}).get("f1_macro_pres")
    rows = []
    for name in ["R0", "R1", "R2", "R3"] + [x for x in arms if x.startswith("R4p")] + \
                ["L1", "L2", "L1p", "L1b", "L3", "L4"]:
        if name not in arms:
            continue
        m = arms[name]["metrics"]
        ci = boot.get("ci", {}).get(name, {}).get("ci95")
        delta = (m["f1_macro_pres"] - r2) if r2 is not None else None
        sd = m.get("f1_macro_pres_std")
        rows.append(
            f"| {name} | {DESC.get(name, name)} | {fmt(m['f1_macro_pres'])}"
            + (f" ± {sd:.4f}" if sd else "")
            + f" | {fmt(m['f1_macro_all'])} | {fmt(m['accuracy'])} | "
            + (f"{delta:+.4f}" if delta is not None else "—") + " | "
            + (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—") + " |")

    # --- agrégat R4 (descriptif, moyenne des réplicats — pas un estimateur de décision)
    r4s = [arms[k]["metrics"]["f1_macro_pres"] for k in arms if k.startswith("R4p")]
    r4line = (f"| R4 (moy. {len(r4s)} rép.) | 3×3 appariement rompu | "
              f"{sum(r4s) / len(r4s):.4f} | — | — | — | — |") if r4s else None

    md = ["| bras | description | F1-macro présentes | F1-macro 11 | exactitude | "
          "Δ vs R2 | IC 95 % hiérarchique |",
          "|---|---|---:|---:|---:|---:|---|"] + rows
    if r4line:
        md.append(r4line)

    # --- décision mécanique (critères pré-enregistrés §9)
    dec = {}
    l1 = arms.get("L1", {}).get("metrics", {}).get("f1_macro_pres")
    l2 = arms.get("L2", {}).get("metrics", {}).get("f1_macro_pres")
    g1 = (l1 is not None and r2 is not None and (l1 - r2) > 0.010)
    g2 = (l1 is not None and l2 is not None and (l1 - l2) > 0.0)
    pd_ = boot.get("paired", {}).get("L1-R2", {})
    g3 = bool(pd_.get("ci95") and pd_["ci95"][0] > 0)
    dec.update({"L1": l1, "L2": l2, "R2": r2, "delta_L1_R2": None if None in (l1, r2) else l1 - r2,
                "delta_L1_L2": None if None in (l1, l2) else l1 - l2,
                "G1_delta_gt_0.010": g1, "G2_L1_gt_L2": g2, "G3_ci_lower_gt_0": g3,
                "paired_L1_R2": pd_, "verdict": "GO" if (g1 and g2) else "NO-GO"})
    if not (g1 and g2) and (g1 or g2):
        dec["verdict"] = "NO-GO (cas ambigu : un seul critère satisfait — règle §9)"

    md += ["", "## Décision mécanique (critères pré-enregistrés)", ""]
    md += [f"- G1 `L1 − R2 > +0,010` : **{g1}** "
           f"(Δ = {dec['delta_L1_R2']:+.4f})" if dec["delta_L1_R2"] is not None else "- G1 : n/d"]
    md += [f"- G2 `L1 − L2 > 0` : **{g2}** "
           f"(Δ = {dec['delta_L1_L2']:+.4f})" if dec["delta_L1_L2"] is not None else "- G2 : n/d"]
    md += [f"- G3 IC 95 % apparié `L1 − R2` exclut 0 : **{g3}** "
           f"({pd_.get('ci95')})" if pd_ else "- G3 : n/d"]
    md += ["", f"**Verdict : {dec['verdict']}**", ""]
    if grid:
        md += ["## Complétude de la grille", ""]
        for k, v in grid.items():
            md += [f"- {k}: {v['mean_present']:.2f} voisins présents en moyenne, "
                   f"{100 * v['frac_all_present']:.1f} % des tuiles complètes "
                   f"(histogramme {v['hist_present']})"]

    with open(os.path.join(a.out_dir, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    json.dump(dec, open(os.path.join(a.out_dir, "decision.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
