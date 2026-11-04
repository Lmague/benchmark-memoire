#!/usr/bin/env python3
"""Paired comparison of a fine-tuned (PEFT) run against its own frozen backbone.

For each pair, compares on the same test tiles and the same 9 maps:
  - single tile (R0),
  - 3x3 mean + mean field (R3_MF),
  - the smoothing gain itself, (R3_MF - R3_base), PEFT minus frozen.
with a hierarchical bootstrap over maps, as in compare_arms.py.

Usage:
    python compare_peft_pairs.py --root $SCRATCH/context_multibackbone/out \
        --out results/tile_context_2026-11-01/peft_pairs.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

PAIRS = [
    ("dinov3_vitl16_lvd", "dinov3_vitl16_lvd_lora_s0", "DINOv3-L"),
    ("simdinov2_vitb16", "simdinov2_vitb16_lora_r8a16_s0", "SimDINOv2-B"),
    ("simdinov2_vitl16", "simdinov2_vitl16_lora_s0", "SimDINOv2-L"),
]
K = 11


def f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    return float(f[~np.isnan(f)].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    split = a.split or os.path.join(os.path.dirname(here), "out/canonical_split.json")
    tiles = json.load(open(split))["tiles"]
    te = [t for t in tiles if t["split"] == "test"]
    y = np.array([t["label"] for t in te])
    ortho = np.array([t["ortho"] for t in te])
    names = sorted(set(ortho))
    groups = {o: np.where(ortho == o)[0] for o in names}

    def pred(bb: str, arm: str) -> np.ndarray:
        return np.load(os.path.join(a.root, bb, "preds", f"{arm}.npy"))

    def boot(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float, float]:
        rng = np.random.RandomState(a.seed)
        d = np.empty(a.boot)
        for i in range(a.boot):
            pick = rng.choice(names, size=len(names), replace=True)
            idx = np.concatenate([groups[o][rng.randint(0, len(groups[o]), len(groups[o]))] for o in pick])
            d[i] = f1(y[idx], pa[idx]) - f1(y[idx], pb[idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        return float(np.mean(d)), float(lo), float(hi), float(p)

    rows = []
    for frozen, peft, label in PAIRS:
        try:
            pr, fr = pred(peft, "R0_base"), pred(frozen, "R0_base")
            pm, fm = pred(peft, "R3_MF"), pred(frozen, "R3_MF")
            pb, fb = pred(peft, "R3_base"), pred(frozen, "R3_base")
        except FileNotFoundError:
            print(f"[skip] {label}: missing predictions")
            continue
        _, l0, h0, p0 = boot(pr, fr)
        _, lm, hm, pm_p = boot(pm, fm)
        d0 = f1(y, pr) - f1(y, fr)
        dm = f1(y, pm) - f1(y, fm)
        # difference of smoothing gains, (PEFT MF - PEFT base) - (frozen MF - frozen base)
        rng = np.random.RandomState(a.seed)
        d = np.empty(max(a.boot // 2, 1000))
        for i in range(len(d)):
            pick = rng.choice(names, size=len(names), replace=True)
            idx = np.concatenate([groups[o][rng.randint(0, len(groups[o]), len(groups[o]))] for o in pick])
            d[i] = (f1(y[idx], pm[idx]) - f1(y[idx], pb[idx])) - (f1(y[idx], fm[idx]) - f1(y[idx], fb[idx]))
        rows.append({"pair": label, "frozen": frozen, "peft": peft,
                     "frozen_R0": f1(y, fr), "peft_R0": f1(y, pr),
                     "d_R0": d0, "lo_R0": l0, "hi_R0": h0, "p_R0": p0,
                     "frozen_R3_MF": f1(y, fm), "peft_R3_MF": f1(y, pm),
                     "d_R3_MF": dm, "lo_R3_MF": lm, "hi_R3_MF": hm, "p_R3_MF": pm_p,
                     "d_gain": (f1(y, pm) - f1(y, pb)) - (f1(y, fm) - f1(y, fb)),
                     "lo_gain": float(np.percentile(d, 2.5)),
                     "hi_gain": float(np.percentile(d, 97.5))})

    if not rows:
        raise SystemExit(f"[peft] no pairs found under {a.root}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[peft] wrote {a.out} ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
