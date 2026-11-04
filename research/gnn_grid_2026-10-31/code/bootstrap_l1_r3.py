#!/usr/bin/env python3
"""Bootstrap hiérarchique apparié L1 - R3 (attention apprise vs moyenne non apprise).

Même protocole que out/canon/bootstrap.json : orthos avec remise, puis tuiles avec
remise dans chaque ortho ; 10 000 réplicats, seed 7 ; F1 macro 11 classes.
L1 a 3 seeds -> on moyenne les F1 des seeds (jamais de vote majoritaire).
"""
from __future__ import annotations

import json
import numpy as np

SPLIT = "out/canonical_split.json"
PREDS = "out/canon/preds"
N_BOOT = 10_000
SEED = 7
K = 11


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    sup = cm.sum(1).astype(float)
    pred = cm.sum(0).astype(float)
    denom = sup + pred
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)
    ok = ~np.isnan(f1)
    return float(f1[ok].mean())


def main() -> None:
    tiles = json.load(open(SPLIT))["tiles"]
    test = [t for t in tiles if t["split"] == "test"]
    y = np.array([t["label"] for t in test], dtype=np.int64)
    ortho = np.array([t["ortho"] for t in test])
    l1 = np.load(f"{PREDS}/L1.npy")          # (3, n)
    r3 = np.load(f"{PREDS}/R3.npy")          # (n,)
    assert l1.shape[1] == r3.shape[0] == len(y)

    f1_l1 = [macro_f1(y, l1[s]) for s in range(l1.shape[0])]
    f1_r3 = macro_f1(y, r3)
    print(f"L1 seeds : {[round(v,4) for v in f1_l1]}  moyenne {np.mean(f1_l1):.4f}")
    print(f"R3       : {f1_r3:.4f}")
    delta = float(np.mean(f1_l1) - f1_r3)
    print(f"delta observe (L1 - R3) = {delta:+.4f}")

    names = sorted(set(ortho))
    groups = {o: np.where(ortho == o)[0] for o in names}
    rng = np.random.RandomState(SEED)
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked = rng.choice(names, size=len(names), replace=True)
        idx = np.concatenate([groups[o][rng.randint(0, len(groups[o]), len(groups[o]))]
                              for o in picked])
        yy = y[idx]
        a = np.mean([macro_f1(yy, l1[s][idx]) for s in range(l1.shape[0])])
        c = macro_f1(yy, r3[idx])
        deltas[b] = a - c

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    print(f"IC95 = [{lo:+.4f}, {hi:+.4f}]   p = {p:.4f}   "
          f"-> {'DIFFERENCE DEMONTREE' if (lo > 0 or hi < 0) else 'NON DEMONTREE'}")
    json.dump({"delta_mean": delta, "ci95": [float(lo), float(hi)],
               "p_two_sided_boot": float(p), "n_boot": N_BOOT, "seed": SEED,
               "unit": "ortho puis tuile", "metric": "f1_macro_11cls",
               "l1_seed_f1": f1_l1, "r3_f1": f1_r3},
              open("out/canon/bootstrap_L1_R3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
