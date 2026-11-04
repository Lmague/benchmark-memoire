#!/usr/bin/env python3
"""Métriques secondaires 8 classes (exigées par GOAL_SPEC §6) depuis les prédictions sauvées.

Schéma 8 classes du projet : ALDE(0) BIRC(2) LICH(4) MOSS(5) PETF(6) SEDG(8) TUSS(9) WILL(10)
— on retire ARCA/DRYI/RUBC, jamais fiables (F1 ≈ 0 pour tous les modèles, manuscrit §résultats).

Aucune inférence : lit out/preds/<bras>.npy (déjà produit) + out/pilot_split.json.
Usage : python code/add_8cls_metrics.py --out-dir out
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

KEEP = [0, 2, 4, 5, 6, 8, 9, 10]
NAMES8 = ["ALDE", "BIRC", "LICH", "MOSS", "PETF", "SEDG", "TUSS", "WILL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out-dir", default=os.path.join(root, "out"))
    ap.add_argument("--split", default=None,
                    help="split JSON (défaut: <out-dir>/pilot_split.json)")
    a = ap.parse_args()
    sp = a.split or os.path.join(a.out_dir, "pilot_split.json")
    split = json.load(open(sp))
    tiles = split["tiles"]
    # les prédictions sauvées ne couvrent QUE les tuiles de test, dans l'ordre du split
    y = np.array([t["label"] for t in tiles if t["split"] == "test"])
    mask = np.isin(y, KEEP)
    remap = {c: i for i, c in enumerate(KEEP)}
    y8 = np.array([remap[int(v)] for v in y[mask]])
    print(f"test 8 classes : {len(y8)} tuiles, support {np.bincount(y8, minlength=8).tolist()}")

    out = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "preds", "*.npy"))):
        name = os.path.basename(p)[:-4]
        pr = np.load(p)
        pr = pr[:, mask] if pr.ndim == 2 else pr[mask]
        if pr.ndim == 2:                        # (S, n) : moyenne des F1 par seed
            f1s = [f1_score(y8, np.array([remap[int(v)] if int(v) in remap else 0 for v in s]),
                            average="macro", zero_division=0, labels=list(range(8)))
                   for s in pr]
            accs = [accuracy_score(y8, np.array([remap[int(v)] if int(v) in remap else 0
                                                 for v in s])) for s in pr]
            f1, sd, acc = float(np.mean(f1s)), float(np.std(f1s, ddof=1)), float(np.mean(accs))
        else:
            mapped = np.array([remap[int(v)] if int(v) in remap else 0 for v in pr])
            f1 = float(f1_score(y8, mapped, average="macro", zero_division=0,
                                labels=list(range(8))))
            sd, acc = 0.0, float(accuracy_score(y8, mapped))
        pc = f1_score(y8, np.array([remap[int(v)] if int(v) in remap else 0
                                    for v in (pr[0] if pr.ndim == 2 else pr)]),
                      average=None, zero_division=0, labels=list(range(8)))
        out[name] = {"f1_macro_8cls": f1, "f1_macro_8cls_std": sd, "accuracy_8cls": acc,
                     "per_class_f1_8cls": dict(zip(NAMES8, [float(x) for x in pc])),
                     "n_tiles": int(len(y8)), "classes": KEEP}
        print(f"{name:8s} F1_8={f1:.4f}±{sd:.4f}  acc={acc:.4f}")
    json.dump(out, open(os.path.join(a.out_dir, "metrics_8cls.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
