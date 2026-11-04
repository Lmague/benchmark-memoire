#!/usr/bin/env python3
"""Post-traitement décisionnel : lissage spatial des probabilités (Potts / ICM / champ moyen).

Idée testée : l'agrégation de features sature parce que les voisins sont redondants,
mais la décision de chaque tuile reste bruitée. Régulariser la CARTE (pas les features)
exploite la même géométrie à un autre niveau, pour un coût nul.

Protocole : sonde canonique (R3 = moyenne 3x3 des tuiles annotées), C choisi sur
validation, puis lissage sur le graphe des tuiles. Le paramètre de couplage et le
nombre d'itérations sont choisis SUR VALIDATION, jamais sur le test.

Toutes les étiquettes utilisées pour le lissage sont PRÉDITES (y compris sur le train) :
aucune étiquette de vérité terrain n'entre dans la propagation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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


def build_graph(pos):
    idx = {p: i for i, p in enumerate(pos)}
    rows, cols = [], []
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for i, (o, r, c) in enumerate(pos):
        for dr, dc in offs:
            j = idx.get((o, r + STRIDE * dr, c + STRIDE * dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(pos)
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    return A


def mean_features(X, A):
    """Moyenne des voisins présents, tuile centrale incluse (comme R3)."""
    A1 = A.copy()
    A1.setdiag(1.0)
    cnt = np.asarray(A1.sum(1)).ravel()
    return np.asarray(A1 @ X) / cnt[:, None], cnt


def onehot(labels, k=N_CLS):
    O = np.zeros((len(labels), k), np.float32)
    O[np.arange(len(labels)), labels] = 1.0
    return O


def icm(logp, A, beta, iters, init):
    L = init.copy()
    for _ in range(iters):
        cnt = A @ onehot(L)
        L = np.argmax(logp + beta * cnt, axis=1).astype(np.int64)
    return L


def meanfield(logp, A, beta, iters, init):
    Q = onehot(init)
    for _ in range(iters):
        Q = A @ Q
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
    ap.add_argument("--out", default="out/decoding/decoding.json")
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

    print(f"[dec] {len(tiles)} tuiles, {len(tr)}/{len(va)}/{len(te)}", flush=True)
    A = build_graph(pos)
    print(f"[dec] arêtes={A.nnz} moyenne={A.nnz / len(pos):.2f}/tuile", flush=True)

    # --- redondance du voisinage (explication du résultat négatif)
    Xm, cnt = mean_features(X, A)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rows, cols = A.nonzero()
    cos = (Xn[rows] * Xn[cols]).sum(1)
    red = {"cos_centre_voisin_moyen": float(cos.mean()), "cos_sd": float(cos.std()),
           "cos_p10": float(np.percentile(cos, 10)),
           "voisins_moyens": float(A.nnz / len(pos))}
    # part de variance expliquée par le centre vs le voisinage (PCA sur un échantillon)
    rng = np.random.RandomState(0)
    samp = rng.choice(len(pos), size=min(20000, len(pos)), replace=False)
    D = np.concatenate([Xm[samp], X[samp]], axis=0)
    D = (D - D.mean(0)) / (D.std(0) + 1e-6)
    _, sv, _ = np.linalg.svd(D, full_matrices=False)
    ev = sv ** 2
    red["pca_var_top1"] = float(ev[0] / ev.sum())
    red["pca_var_top10"] = float(ev[:10].sum() / ev.sum())
    red["pca_var_top50"] = float(ev[:50].sum() / ev.sum())
    print("[dec] redondance:", json.dumps({k: round(v, 4) for k, v in red.items()}), flush=True)

    # --- sonde canonique sur la moyenne 3x3
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
    base_va = f1_macro(y[va], P[va].argmax(1))
    base_te = f1_macro(y[te], P[te].argmax(1))
    print(f"[dec] base : C={C} val={base_va:.4f} test={base_te:.4f}", flush=True)

    init = P.argmax(1)
    out = {"C": C, "base_val": base_va, "base_test": base_te, "redundance": red,
           "grid": []}

    # --- ICM (Potts, mise à jour dure)
    best_icm = (-1, None)
    for beta in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
        for it in (1, 2, 3, 5, 10):
            L = icm(logp, A, beta, it, init)
            v = f1_macro(y[va], L[va])
            out["grid"].append({"method": "icm", "beta": beta, "iters": it,
                                "val": v, "test": f1_macro(y[te], L[te])})
            if v > best_icm[0]:
                best_icm = (v, (beta, it, L))
    bv, (bb, bi, BL) = best_icm
    print(f"[dec] ICM   meilleur val={bv:.4f} (beta={bb}, iters={bi})  test={f1_macro(y[te], BL[te]):.4f}",
          flush=True)

    # --- champ moyen (mise à jour douce)
    best_mf = (-1, None)
    for beta in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0):
        for it in (1, 2, 3, 5):
            L = meanfield(logp, A, beta, it, init)
            v = f1_macro(y[va], L[va])
            out["grid"].append({"method": "meanfield", "beta": beta, "iters": it,
                                "val": v, "test": f1_macro(y[te], L[te])})
            if v > best_mf[0]:
                best_mf = (v, (beta, it, L))
    mv, (mb, mi, ML) = best_mf
    print(f"[dec] champ moyen val={mv:.4f} (beta={mb}, iters={mi})  test={f1_macro(y[te], ML[te]):.4f}",
          flush=True)

    out["best_icm"] = {"beta": bb, "iters": bi, "val": bv, "test": f1_macro(y[te], BL[te])}
    out["best_meanfield"] = {"beta": mb, "iters": mi, "val": mv, "test": f1_macro(y[te], ML[te])}
    out["n_test"] = int(len(te))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    pdir = os.path.join(os.path.dirname(a.out), "preds")
    os.makedirs(pdir, exist_ok=True)
    np.save(os.path.join(pdir, "R3_ICM.npy"), BL[te].astype(np.uint8))
    np.save(os.path.join(pdir, "R3_MF.npy"), ML[te].astype(np.uint8))
    np.save(os.path.join(pdir, "R3_base.npy"), P[te].argmax(1).astype(np.uint8))
    print("écrit", a.out)


if __name__ == "__main__":
    main()
