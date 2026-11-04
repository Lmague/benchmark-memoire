#!/usr/bin/env python3
"""Le lissage décisionnel dépend-il de l'échelle du voisinage ?

Hypothèse laissée ouverte par l'étude précédente : à stride 112 et tuiles de 224, les
voisins se recouvrent à 50 % (orthogonal) et 25 % (diagonal), donc le voisinage est
redondant et l'agrégation sature. Si c'est la bonne explication, un voisinage à ±224 px
(recouvrement nul, 4x la surface) devrait apporter davantage.

Test : mêmes probabilités par tuile (donc aucune feature recalculée), seul le graphe de
lissage change. Le couple (beta, itérations) est réglé sur validation pour chaque échelle.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_smoothing import f1_macro, meanfield, onehot  # noqa: E402

STRIDE = 112
N_CLS = 11


def graph_at(pos, steps):
    """Graphe 8-voisins aux décalages donnés (en unités de stride)."""
    idx = {p: i for i, p in enumerate(pos)}
    rows, cols = [], []
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for i, (o, r, c) in enumerate(pos):
        for k in steps:
            for dr, dc in offs:
                j = idx.get((o, r + STRIDE * k * dr, c + STRIDE * k * dc))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(pos), len(pos)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--features", default="out/aux/features_tile.npy")
    ap.add_argument("--out", default="out/decoding/scales.json")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    tiles = json.load(open(a.split))["tiles"]
    X = np.load(a.features).astype(np.float32)
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
    pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)
    best = (-1, None, None)
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        clf = make_canonical_lr(C=C)
        clf.fit(Xs[tr], y[tr])
        v = f1_macro(y[va], clf.predict(Xs[va]))
        if v > best[0]:
            best = (v, C, clf)
    _, C, clf = best
    P = clf.predict_proba(Xs).astype(np.float32)
    logp = np.log(np.maximum(P, 1e-12)).astype(np.float32)
    init = P.argmax(1)
    base = f1_macro(y[te], init[te])
    print(f"[scale] base R0 test={base:.4f} (val={f1_macro(y[va], init[va]):.4f}, C={C})",
          flush=True)

    res = {"base_test": base, "C": C, "grid": []}
    for label, steps in (("+-112", [1]), ("+-224", [2]), ("+-336", [3]),
                         ("112+224", [1, 2]), ("112+224+336", [1, 2, 3])):
        A = graph_at(pos, steps)
        cov = A.nnz / len(pos)
        bestc = (-1, None)
        for beta in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            for it in (1, 2, 3, 5):
                L = meanfield(logp, A, beta, it, init)
                v = f1_macro(y[va], L[va])
                if v > bestc[0]:
                    bestc = (v, (beta, it, L))
        v, (bb, bi, L) = bestc
        t = f1_macro(y[te], L[te])
        res["grid"].append({"scale": label, "voisins_moyens": float(cov), "beta": bb,
                            "iters": bi, "val": v, "test": t})
        np.save(os.path.join(os.path.dirname(a.out), "preds",
                             f"R0_MF_{label.replace('+', '')}.npy"), L[te].astype(np.uint8))
        print(f"[scale] {label:12s} voisins={cov:5.2f} beta={bb:<5g} it={bi} "
              f"val={v:.4f} test={t:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
