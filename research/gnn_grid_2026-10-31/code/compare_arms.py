#!/usr/bin/env python3
"""Comparaison appariée de deux bras quelconques, bootstrap hiérarchique.

Protocole identique à out/canon/bootstrap.json : orthos avec remise, puis tuiles avec
remise dans chaque ortho ; 10 000 réplicats ; F1 macro. Un bras multi-seeds est agrégé
par moyenne des F1 (jamais de vote majoritaire).

Usage :
  compare_arms.py --preds-dir out/canon/gnn/preds --arm-a GNN2 --arm-b MEAN9
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

K = 11
KEEP8 = [0, 2, 4, 5, 6, 8, 9, 10]
REMAP8 = {c: i for i, c in enumerate(KEEP8)}


def f1_macro(y: np.ndarray, p: np.ndarray, k: int = K) -> float:
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    return float(f[~np.isnan(f)].mean())


def load_preds(d: str, arm: str) -> np.ndarray:
    a = np.load(os.path.join(d, f"{arm}.npy"))
    return a.reshape(1, -1) if a.ndim == 1 else a


def arm_f1(y: np.ndarray, preds: np.ndarray, k: int = K) -> float:
    return float(np.mean([f1_macro(y, p, k) for p in preds]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tiles = [t for t in json.load(open(a.split))["tiles"] if t["split"] == "test"]
    y = np.array([t["label"] for t in tiles], dtype=np.int64)
    ortho = np.array([t["ortho"] for t in tiles])
    A = load_preds(a.preds_dir, a.arm_a)
    B = load_preds(a.preds_dir, a.arm_b)
    assert A.shape[1] == B.shape[1] == len(y), (A.shape, B.shape, len(y))

    fa, fb = arm_f1(y, A), arm_f1(y, B)
    print(f"{a.arm_a:8s} F1(11cls)={fa:.4f}  seeds={[round(x,4) for x in [f1_macro(y,p) for p in A]]}")
    print(f"{a.arm_b:8s} F1(11cls)={fb:.4f}  seeds={[round(x,4) for x in [f1_macro(y,p) for p in B]]}")
    print(f"delta observe = {fa - fb:+.4f}")

    # 8 classes — convention identique à add_8cls_metrics.py : prédiction d'une classe
    # exclue (ARCA/DRYI/RUBC) repliée sur la classe 0, F1 macro sur les 8 classes.
    lut = np.zeros(K, dtype=np.int64)
    for i, c in enumerate(KEEP8):
        lut[c] = i
    keep = np.isin(y, KEEP8)
    y8 = lut[y[keep]]
    fa8 = float(np.mean([f1_macro(y8, lut[p[keep]], len(KEEP8)) for p in A]))
    fb8 = float(np.mean([f1_macro(y8, lut[p[keep]], len(KEEP8)) for p in B]))
    print(f"F1(8cls) : {a.arm_a}={fa8:.4f}  {a.arm_b}={fb8:.4f}  delta={fa8-fb8:+.4f}")

    names = sorted(set(ortho))
    groups = {o: np.where(ortho == o)[0] for o in names}
    rng = np.random.RandomState(a.seed)
    d = np.empty(a.boot)
    for i in range(a.boot):
        pick = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([groups[o][rng.randint(0, len(groups[o]), len(groups[o]))]
                              for o in pick])
        d[i] = arm_f1(y[idx], A[:, idx]) - arm_f1(y[idx], B[:, idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    verdict = "DEMONTRE" if (lo > 0 or hi < 0) else "NON DEMONTRE"
    print(f"IC95 [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  -> {verdict}")

    if a.out:
        json.dump({"arm_a": a.arm_a, "arm_b": a.arm_b, "f1_11cls": [fa, fb],
                   "f1_8cls": [fa8, fb8], "delta": fa - fb,
                   "ci95": [float(lo), float(hi)], "p": float(p),
                   "n_boot": a.boot, "seed": a.seed,
                   "unit": "ortho puis tuile"},
                  open(a.out, "w"), indent=1)
        print("écrit", a.out)


if __name__ == "__main__":
    main()
