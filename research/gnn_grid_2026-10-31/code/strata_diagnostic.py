#!/usr/bin/env python3
"""Diagnostic : où le lissage agit-il ?

Question posée par un jury : les étiquettes venant de polygones, deux tuiles voisines
ont la même classe en partie PAR CONSTRUCTION. Si le gain du lissage est concentré
dans les intérieurs homogènes, une part du résultat exploite la géométrie de
l'annotation plutôt que la radiométrie.

Proxy disponible sans les polygones : le nombre de voisins (8) portant la même
étiquette de vérité terrain. 8/8 = intérieur de tache, 0-2 = bordure ou mosaïque.

Sortie : effectifs, exactitude et F1 par strate, et surtout le tableau des
renversements (faux→juste, juste→faux), qui dit ce que le lissage fait réellement.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_smoothing import build_graph, f1_macro  # noqa: E402

STRATA = [(8, 8, "intérieur (8/8)"), (6, 7, "quasi-homogène (6-7)"),
          (3, 5, "mixte (3-5)"), (0, 2, "bordure (0-2)")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--preds-dir", default="out/decoding/preds")
    ap.add_argument("--out", default="out/decoding/strata.json")
    a = ap.parse_args()

    tiles = json.load(open(a.split))["tiles"]
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]
    te = np.where(s == "test")[0]

    A = build_graph(pos)
    # nombre de voisins de même étiquette, calculé sur la vérité terrain
    Y = np.zeros((len(y), 11), np.float32)
    Y[np.arange(len(y)), y] = 1.0
    S = A @ Y                      # (n, 11) : compte des voisins par classe
    same = S[np.arange(len(y)), y]  # voisins portant la même étiquette que le centre
    nb = np.asarray(A.sum(1)).ravel()

    preds = {}
    for arm in ("R0_base", "R3_base", "R3_MF"):
        p = os.path.join(a.preds_dir, f"{arm}.npy")
        if os.path.exists(p):
            arr = np.load(p)
            if len(arr) == len(tiles):
                arr = arr[te]
            preds[arm] = arr.astype(np.int64)
    print("bras disponibles :", list(preds), flush=True)

    yt = y[te]
    res = {"strata": [], "n_test": int(len(te))}
    for lo, hi, name in STRATA:
        m = (same[te] >= lo) & (same[te] <= hi)
        if m.sum() == 0:
            continue
        row = {"strate": name, "n": int(m.sum()),
               "part": float(m.mean())}
        for arm, pr in preds.items():
            row[arm] = {"acc": float((pr[m] == yt[m]).mean()),
                        "f1": f1_macro(yt[m], pr[m])}
        if "R3_base" in preds and "R3_MF" in preds:
            b, f = preds["R3_base"][m], preds["R3_MF"][m]
            ok_b, ok_f = b == yt[m], f == yt[m]
            row["flips"] = {"faux_vers_juste": int((~ok_b & ok_f).sum()),
                            "juste_vers_faux": int((ok_b & ~ok_f).sum()),
                            "net": int((~ok_b & ok_f).sum() - (ok_b & ~ok_f).sum())}
        res["strata"].append(row)
        line = f"{name:<22} n={m.sum():6d} ({100*m.mean():4.1f} %)"
        for arm in preds:
            line += f" | {arm} acc={row[arm]['acc']:.4f} f1={row[arm]['f1']:.4f}"
        if "flips" in row:
            line += f" | net {row['flips']['net']:+d}"
        print(line, flush=True)

    # vue complémentaire : par classe, part des tuiles en intérieur
    per_class = {}
    for c in range(11):
        mc = (yt == c) & (same[te] >= 0)
        if mc.sum() < 20:
            continue
        inner = (yt == c) & (same[te] == 8)
        per_class[int(c)] = {"n": int(mc.sum()), "part_interieur": float(inner.sum() / mc.sum())}
        if "R3_base" in preds and "R3_MF" in preds:
            d = ((preds["R3_MF"][mc] == yt[mc]).mean() - (preds["R3_base"][mc] == yt[mc]).mean())
            per_class[int(c)]["delta_acc"] = float(d)
    res["per_class"] = per_class
    print("par classe :", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()}
                                      for k, v in per_class.items()}), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
