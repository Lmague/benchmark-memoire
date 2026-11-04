#!/usr/bin/env python3
"""Ablation : d'où vient le gain de contexte, des features ou de la décision ?

Quatre conditions, même sonde, même graphe, même réglage sur validation :
  R0        tuile seule, aucune information de voisinage
  R0+MF     tuile seule + lissage spatial des probabilités
  R3        moyenne 3x3 des features (contexte au niveau des features)
  R3+MF     moyenne 3x3 + lissage (les deux niveaux)

Si R0+MF rejoint R3, le contexte feature n'apporte rien que le contexte décisionnel
n'apporte déjà : les deux exploitent la même redondance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_smoothing import (build_graph, f1_macro, icm, mean_features,  # noqa: E402
                             meanfield)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--features", default="out/aux/features_tile.npy")
    ap.add_argument("--out", default="out/decoding/ablation.json")
    ap.add_argument("--repo", default="/home/erazal/Documents/Mémoire")
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    tiles = json.load(open(a.split))["tiles"]
    X = np.load(a.features).astype(np.float32)
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
    A = build_graph([(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles])
    Xm, _ = mean_features(X, A)

    results, preds = {}, {}
    for name, F in (("R0", X), ("R3", Xm)):
        sc = StandardScaler().fit(F[tr])
        Fs = sc.transform(F).astype(np.float32)
        best = (-1, None, None)
        for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            clf = make_canonical_lr(C=C)
            clf.fit(Fs[tr], y[tr])
            v = f1_macro(y[va], clf.predict(Fs[va]))
            if v > best[0]:
                best = (v, C, clf)
        _, C, clf = best
        P = clf.predict_proba(Fs).astype(np.float32)
        logp = np.log(np.maximum(P, 1e-12)).astype(np.float32)
        init = P.argmax(1)
        base = f1_macro(y[te], init[te])
        preds[f"{name}_base"] = init[te].astype(np.uint8)
        print(f"[abl] {name:3s} base      val={f1_macro(y[va], init[va]):.4f} test={base:.4f}",
              flush=True)

        bestmf = (-1, None)
        for beta in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
            for it in (1, 2, 3, 5):
                L = meanfield(logp, A, beta, it, init)
                v = f1_macro(y[va], L[va])
                if v > bestmf[0]:
                    bestmf = (v, (beta, it, L))
        v, (bb, bi, L) = bestmf
        t = f1_macro(y[te], L[te])
        preds[f"{name}_MF"] = L[te].astype(np.uint8)
        print(f"[abl] {name:3s}+MF     val={v:.4f} (beta={bb}, it={bi})  test={t:.4f}", flush=True)

        besticm = (-1, None)
        for beta in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0):
            for it in (1, 2, 3, 5):
                L = icm(logp, A, beta, it, init)
                v = f1_macro(y[va], L[va])
                if v > besticm[0]:
                    besticm = (v, (beta, it, L))
        v, (ib, ii, L) = besticm
        preds[f"{name}_ICM"] = L[te].astype(np.uint8)
        results[name] = {"C": C, "base_test": base, "base_val": f1_macro(y[va], init[va]),
                         "mf": {"beta": bb, "iters": bi, "val": v, "test": t},
                         "icm": {"beta": ib, "iters": ii, "val": f1_macro(y[va], L[va]),
                                 "test": f1_macro(y[te], L[te])}}
        print(f"[abl] {name:3s}+ICM    val={f1_macro(y[va], L[va]):.4f} test="
              f"{f1_macro(y[te], L[te]):.4f}", flush=True)

    pdir = os.path.join(os.path.dirname(a.out), "preds")
    os.makedirs(pdir, exist_ok=True)
    for k, v in preds.items():
        np.save(os.path.join(pdir, f"{k}.npy"), v)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
