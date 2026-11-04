#!/usr/bin/env python3
"""Fusion features de fondation gelées + descripteurs non appris (couleur, texture).

Question : les canaux RGB bruts, les indices de végétation et la texture apportent-ils
quelque chose que le ViT iNat ne code pas déjà ?

Bras (tous avec la même sonde canonique, C choisi sur validation) :
  COL        descripteurs couleur/texture de la tuile seule
  COL3       leur moyenne 3x3 (le voisinage aide-t-il la couleur ?)
  R3         moyenne 3x3 des features ViT (référence, 0,5205)
  R3+COL     concaténation ViT-3x3 + couleur tuile
  R3+COL3    concaténation ViT-3x3 + couleur 3x3
  R0+COL     tuile ViT + couleur (sans contexte)

Si COL3 >> COL mais R3+COL3 ≈ R3, le voisinage apporte à la couleur ce que le ViT
possède déjà : c'est la signature d'une information redondante avec le backbone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

STRIDE = 112
N_CLS = 11
KEEP8 = [0, 2, 4, 5, 6, 8, 9, 10]


def f1_macro(y, p, k=N_CLS):
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    d = cm.sum(1) + cm.sum(0)
    f = np.where(d > 0, 2 * tp / np.maximum(d, 1), np.nan)
    return float(f[~np.isnan(f)].mean())


def build_A(pos):
    idx = {p: i for i, p in enumerate(pos)}
    rows, cols = [], []
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for i, (o, r, c) in enumerate(pos):
        for dr, dc in offs:
            j = idx.get((o, r + STRIDE * dr, c + STRIDE * dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(pos), len(pos)))


def mean3(X, A):
    A1 = A.copy()
    A1.setdiag(1.0)
    return np.asarray(A1 @ X) / np.asarray(A1.sum(1)).ravel()[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--vit", default="out/aux/features_tile.npy")
    ap.add_argument("--color", default="out/color/color_features.npy")
    ap.add_argument("--out", default="out/color/fusion.json")
    ap.add_argument("--repo", default="/home/erazal/Documents/Mémoire")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    tiles = json.load(open(a.split))["tiles"]
    Xv = np.load(a.vit).astype(np.float32)
    Xc = np.load(a.color).astype(np.float32)
    assert Xc.shape[0] == Xv.shape[0] == len(tiles), (Xc.shape, Xv.shape, len(tiles))
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
    A = build_A([(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles])
    print(f"[fuse] {len(tiles)} tuiles  {len(tr)}/{len(va)}/{len(te)}  couleur={Xc.shape[1]}d",
          flush=True)

    V3, C3 = mean3(Xv, A), mean3(Xc, A)
    arms = {"COL": (Xc,), "COL3": (C3,), "R0": (Xv,), "R3": (V3,),
            "R0+COL": (Xv, Xc), "R3+COL": (V3, Xc), "R3+COL3": (V3, C3),
            "COL3+COL": (C3, Xc)}
    res, preds = {}, {}
    for name, blocks in arms.items():
        # standardisation par bloc, ajustée sur le train uniquement
        parts = []
        for B in blocks:
            sc = StandardScaler().fit(B[tr])
            parts.append(sc.transform(B).astype(np.float32))
        X = np.concatenate(parts, axis=1)
        best = (-1, None, None)
        for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            clf = make_canonical_lr(C=C)
            clf.fit(X[tr], y[tr])
            v = f1_macro(y[va], clf.predict(X[va]))
            if v > best[0]:
                best = (v, C, clf)
        v, C, clf = best
        p = clf.predict(X[te]).astype(np.uint8)
        res[name] = {"f1_val": v, "f1_test": f1_macro(y[te], p), "C": C,
                     "dim": int(X.shape[1])}
        preds[name] = p
        print(f"[fuse] {name:9s} dim={X.shape[1]:5d} C={C:<6g} val={v:.4f} "
              f"test={res[name]['f1_test']:.4f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pdir = os.path.join(os.path.dirname(a.out), "preds")
    os.makedirs(pdir, exist_ok=True)
    for k, v in preds.items():
        np.save(os.path.join(pdir, f"{k.replace('+', '_')}.npy"), v)
    json.dump({"arms": res, "n_test": int(len(te))}, open(a.out, "w"), indent=1)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
