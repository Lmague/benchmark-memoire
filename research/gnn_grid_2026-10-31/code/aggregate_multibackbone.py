#!/usr/bin/env python3
"""Aggregate the multi-backbone context/smoothing runs into one CSV.

Reads, for every backbone directory under ``--root``:
    <name>/ablation.json          point estimates for R0, R3, +MF, +ICM
    <name>/cmp_R0_base_R0_MF.json paired bootstrap of the smoothing gain
    <name>/cmp_R3_base_R3_MF.json same on top of the 3x3 mean
    <name>/cmp_R0_base_R3_base.json context gain
and writes one row per backbone.

Usage:
    python aggregate_multibackbone.py --root $SCRATCH/context_multibackbone/out \
        --out results/tile_context_2026-11-01/multibackbone_smoothing.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def load(path: str):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        abl = os.path.join(d, "ablation.json")
        if not os.path.exists(abl):
            continue
        r = load(abl)
        row = {"backbone": name,
               "R0": r["R0"]["base_test"], "R0_MF": r["R0"]["mf"]["test"],
               "R0_ICM": r["R0"]["icm"]["test"],
               "R3": r["R3"]["base_test"], "R3_MF": r["R3"]["mf"]["test"],
               "R3_ICM": r["R3"]["icm"]["test"],
               "mf_beta_R0": r["R0"]["mf"]["beta"], "mf_iters_R0": r["R0"]["mf"]["iters"],
               "mf_beta_R3": r["R3"]["mf"]["beta"], "mf_iters_R3": r["R3"]["mf"]["iters"]}
        for tag, f in (("mf_R0", "cmp_R0_base_R0_MF.json"),
                       ("mf_R3", "cmp_R3_base_R3_MF.json"),
                       ("ctx", "cmp_R0_base_R3_base.json")):
            p = os.path.join(d, f)
            if os.path.exists(p):
                c = load(p)
                row[f"d_{tag}"] = c["delta"]
                row[f"lo_{tag}"], row[f"hi_{tag}"] = c["ci95"]
                row[f"p_{tag}"] = c["p"]
        rows.append(row)

    if not rows:
        raise SystemExit(f"[aggregate] no ablation.json under {a.root}")
    keys = sorted({k for r in rows for k in r})
    first = ["backbone", "R0", "R0_MF", "R3", "R3_MF", "R3_ICM"]
    keys = first + [k for k in keys if k not in first]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {a.out} ({len(rows)} backbones)")


if __name__ == "__main__":
    main()
