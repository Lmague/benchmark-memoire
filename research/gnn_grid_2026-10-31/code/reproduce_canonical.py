#!/usr/bin/env python3
"""Reproduction du chiffre canonique tuile-seule avec la sonde de l'étude gnn_grid.

But : vérifier que le pipeline de sonde (StandardScaler + LR lbfgs, grille C, F1 macro)
reproduit 0.4717 (SimDINOv2-B tuile seule, split canonique). Si oui, l'écart du pilote
(0.3175) vient du split, pas du code.

BLAS mono-thread forcé avant tout import numpy (AGENTS.md §4.8).
"""
import os
from pathlib import Path

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "1"

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", default=str(Path(__file__).resolve().parents[3] / "embeddings"))
ap.add_argument("--backbone", default="simdinov2_vitb16")
ap.add_argument("--out", default="")
a = ap.parse_args()
EMB = a.emb_dir


def load(split):
    X = np.load(f"{EMB}/{a.backbone}_{split}.npy").astype(np.float32)
    y = np.load(f"{EMB}/{a.backbone}_{split}_labels.npy").astype(np.int64)
    return X, y


Xtr, ytr = load("train")
Xva, yva = load("val")
Xte, yte = load("test")
print("brut:", Xtr.shape, Xva.shape, Xte.shape)
print("labels train:", dict(zip(*np.unique(ytr, return_counts=True))))
print("labels test :", dict(zip(*np.unique(yte, return_counts=True))))

present = np.unique(yte)
keep = np.isin(ytr, present)
if not keep.all():
    print(f"retiré {int((~keep).sum())} lignes d'entraînement hors des classes de test (RHOL)")
Xtr, ytr = Xtr[keep], ytr[keep]

sc = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

best = None
for C in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000, random_state=42, n_jobs=1)
    clf.fit(Xtr, ytr)
    v = f1_score(yva, clf.predict(Xva), average="macro", zero_division=0, labels=present)
    t = f1_score(yte, clf.predict(Xte), average="macro", zero_division=0, labels=present)
    print(f"C={C:<8g} valF1={v:.4f} testF1={t:.4f}", flush=True)
    if best is None or v > best[0]:
        best = (v, C, t, clf)

v, C, t, clf = best
pred = clf.predict(Xte)
pc = f1_score(yte, pred, average=None, zero_division=0, labels=present)
print(f"\nMEILLEUR C={C}  val={v:.4f}  test={t:.4f}  acc={accuracy_score(yte, pred):.4f}")
print("F1 par label:", {int(l): round(float(f), 3) for l, f in zip(present, pc)})
print("référence canonique : 0.4717 (acc 0.7021, best_C 0.001)")
if a.out:
    json.dump({"best_C": float(C), "val_f1": float(v), "test_f1": float(t),
               "accuracy": float(accuracy_score(yte, pred)),
               "per_label": {int(l): float(f) for l, f in zip(present, pc)},
               "labels": [int(x) for x in present],
               "n_train": int(len(ytr)), "n_test": int(len(yte))},
              open(a.out, "w"), indent=1)
