#!/usr/bin/env python3
"""Lissage Potts à poids contrastés (CRF) : le couplage dépend de la similarité.

Motivation mesurée : le lissage uniforme gagne surtout dans les voisinages mixtes
(3-5 voisins de même classe) mais DÉGRADE les vraies bordures (0-2 voisins) :
exactitude 0,334 -> 0,277. Il traverse donc les frontières entre communautés.

Correctif classique en segmentation (Potts à poids contrastés) : pondérer l'arête
par la similarité des features,
    w_ij = exp(-||f_i - f_j||^2 / (2 sigma^2)),
de sorte que deux tuiles de communautés différentes se parlent peu.

sigma est paramétré en multiple de la distance médiane observée sur les arêtes.
Grille (alpha, beta, itérations) choisie SUR VALIDATION ; le test ne sert qu'à
rapporter. Le bras uniforme est recalculé dans le même passage pour garantir une
comparaison à numériques identiques.
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
from label_smoothing import build_graph, f1_macro, mean_features, onehot  # noqa: E402

STRIDE = 112
N_CLS = 11


def to_row_stochastic(W):
    d = np.asarray(W.sum(1)).ravel()
    D = sp.diags(1.0 / np.maximum(d, 1e-9))
    return (D @ W).tocsr()


def meanfield_w(logp, Wn, beta, iters, init):
    Q = onehot(init)
    for _ in range(iters):
        Q = Wn @ Q
        Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-9)
        Z = logp + beta * Q
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        Q = (E / E.sum(1, keepdims=True)).astype(np.float32)
    return Q.argmax(1).astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--features", default="out/aux/features_tile.npy")
    ap.add_argument("--out", default="out/decoding/weighted.json")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    tiles = json.load(open(a.split))["tiles"]
    X = np.load(a.features).astype(np.float32)
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    pos = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]

    A = build_graph(pos)
    Xm, _ = mean_features(X, A)
    sc = StandardScaler().fit(Xm[tr])
    Xs = sc.transform(Xm).astype(np.float32)

    best = (-1, None, None)
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        clf = make_canonical_lr(C=C)
        clf.fit(Xs[tr], y[tr])
        v = f1_macro(y[va], clf.predict(Xs[va]))
        if v > best[0]:
            best = (v, C, clf)
    v0, C, clf = best
    P = clf.predict_proba(Xs).astype(np.float32)
    logp = np.log(np.maximum(P, 1e-12)).astype(np.float32)
    base_te = f1_macro(y[te], P[te].argmax(1))
    print(f"[w] base C={C} val={v0:.4f} test={base_te:.4f}", flush=True)

    # --- poids contrastés sur les arêtes existantes
    rows, cols = A.nonzero()
    d2 = ((Xs[rows] - Xs[cols]) ** 2).sum(1).astype(np.float32)
    med = float(np.median(d2))
    print(f"[w] distance^2 médiane sur {len(rows)} arêtes = {med:.3f}", flush=True)

    def build_w(alpha):
        # sigma^2 = alpha x distance^2 mediane : alpha < 1 accentue le contraste
        w = np.exp(-d2 / (2.0 * alpha * med)).astype(np.float32)
        W = sp.csr_matrix((w, (rows, cols)), shape=A.shape)
        return to_row_stochastic(W), float(w.mean()), float(np.percentile(w, 10)), float(np.percentile(w, 90))

    init = P.argmax(1)
    out = {"C": C, "base_test": base_te, "base_val": v0, "median_d2": med,
           "uniform": [], "weighted": []}

    # --- référence uniforme (même code, mêmes numériques)
    An = to_row_stochastic(A.astype(np.float32))
    deg = float(np.asarray(A.sum(1)).mean())
    for beta in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
        for it in (2, 3, 5):
            L = meanfield_w(logp, An, beta * deg, it, init)
            out["uniform"].append({"beta": beta, "iters": it,
                                   "val": f1_macro(y[va], L[va]),
                                   "test": f1_macro(y[te], L[te])})
    best_u = max(out["uniform"], key=lambda r: r["val"])
    print(f"[w] uniforme  beta={best_u['beta']} it={best_u['iters']} "
          f"val={best_u['val']:.4f} test={best_u['test']:.4f}", flush=True)

    # --- grille pondérée
    for alpha in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        Wn, wmean, wp10, wp90 = build_w(alpha)
        for beta in (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
            for it in (2, 3, 5):
                L = meanfield_w(logp, Wn, beta, it, init)
                out["weighted"].append({"alpha": alpha, "beta": beta, "iters": it,
                                        "w_moyen": wmean, "w_p10": wp10, "w_p90": wp90,
                                        "val": f1_macro(y[va], L[va]),
                                        "test": f1_macro(y[te], L[te])})
        bw = max([r for r in out["weighted"] if r["alpha"] == alpha], key=lambda r: r["val"])
        print(f"[w] alpha={alpha:<5} meilleur val={bw['val']:.4f} "
              f"(beta={bw['beta']}, it={bw['iters']}, w: moy={wmean:.3f} p10={wp10:.3f}) test={bw['test']:.4f}",
              flush=True)

    bw = max(out["weighted"], key=lambda r: r["val"])
    out["best_uniform"] = best_u
    out["best_weighted"] = bw
    print(f"[w] MEILLEUR pondéré : alpha={bw['alpha']} beta={bw['beta']} it={bw['iters']} "
          f"val={bw['val']:.4f} test={bw['test']:.4f}", flush=True)
    print(f"[w] écart vs uniforme (test) : {bw['test'] - best_u['test']:+.4f}", flush=True)

    # prédictions du meilleur pondéré, pour bootstrap apparié et strates
    Wn, _, _, _ = build_w(bw["alpha"])
    L = meanfield_w(logp, Wn, bw["beta"], bw["iters"], init)
    pdir = os.path.join(os.path.dirname(a.out), "preds")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, "R3_WMF.npy"), L[te].astype(np.uint8))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
