#!/usr/bin/env python3
"""Sonde tronquée par PCA : combien de dimensions la tâche utilise-t-elle ?

Question testée : « faut-il utiliser / optimiser toutes les dimensions de
l'embedding ? ». On reprojette les features GELÉES d'un modèle sur leurs `k`
premières composantes principales (PCA fittée sur le train standardisé), puis on
rejoue la sonde CANONIQUE à chaque `k` (StandardScaler train, LogisticRegression
lbfgs multinomial, `best_C` sélectionné sur validation, f1_macro_pres test).

Sortie : results/rapport_data/dim_probe_<model>.json

    python3 scripts/rapport/dim_probe.py            # défaut : dinov3_vitb16_lvd
"""
from __future__ import annotations

import os as _os
# MONO-THREAD OBLIGATOIRE (AGENTS.md §4.8) avant tout import numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "1"

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from sklearn.decomposition import PCA                       # noqa: E402
from sklearn.preprocessing import StandardScaler            # noqa: E402
from sklearn.metrics import f1_score                        # noqa: E402

from registry import OUT                                    # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "dinov3_vitb16_lvd"
KS = [10, 20, 50, 100, 200, 400]
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
N_CLASSES = 11
SEED = 42


def f1_pres(y_true, y_pred, k=N_CLASSES):
    """F1 macro sur les classes présentes (miroir de significance_tier.f1_pres)."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    denom = 2.0 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1[cm.sum(1) > 0].mean())


def load(split):
    X = np.load(os.path.join(_ROOT, "embeddings", f"{MODEL}_{split}.npy"))
    y = np.load(os.path.join(_ROOT, "embeddings",
                             f"{MODEL}_{split}_labels.npy")).astype(np.int64).ravel()
    keep = y != 7                      # RHOL, absente de val/test
    X, y = X[keep], y[keep].copy()
    y[y > 7] -= 1
    return X, y


def main():
    from src.utils import make_canonical_lr
    D = {s: load(s) for s in ("train", "val", "test")}
    sc = StandardScaler().fit(D["train"][0])
    for s in D:
        D[s] = (sc.transform(D[s][0]).astype(np.float32), D[s][1])
    xtr, ytr = D["train"]
    xva, yva = D["val"]
    xte, yte = D["test"]
    print(f"{MODEL}: train={xtr.shape} val={xva.shape} test={xte.shape}", flush=True)

    ks = KS + [xtr.shape[1]]
    out = []
    for k in ks:
        t0 = time.time()
        if k >= xtr.shape[1]:
            A, B, C_ = xtr, xva, xte
        else:
            p = PCA(n_components=k, svd_solver="randomized", random_state=SEED).fit(xtr)
            A, B, C_ = p.transform(xtr), p.transform(xva), p.transform(xte)
        best = (-1.0, None, None)
        for c in C_GRID:
            clf = make_canonical_lr(c, max_iter=2000, random_state=SEED).fit(A, ytr)
            f1v = f1_score(yva, clf.predict(B), average="macro", zero_division=0,
                           labels=list(range(N_CLASSES)))
            if f1v > best[0]:
                best = (f1v, c, clf)
        f1t = f1_pres(yte, best[2].predict(C_))
        out.append({"k": k, "best_C": best[1], "val_f1": round(best[0], 4),
                    "test_f1": round(f1t, 4), "s": round(time.time() - t0)})
        print(f"  k={k:4d}  best_C={best[1]:<7g} val={best[0]:.4f} "
              f"TEST={f1t:.4f}  ({out[-1]['s']}s)", flush=True)
        json.dump(out, open(os.path.join(OUT, f"dim_probe_{MODEL}.json"), "w"), indent=1)
    print(f"[OK] {os.path.join(OUT, 'dim_probe_' + MODEL + '.json')}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
