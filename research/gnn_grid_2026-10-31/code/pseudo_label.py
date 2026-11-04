#!/usr/bin/env python3
"""Pseudo-étiquetage des 40 030 tuiles non annotées (self-training).

Contexte : ajouter ces tuiles comme contexte brut DÉGRADE (−0,028, p<0,001), parce qu'elles
ne sont pas distribuées comme les tuiles annotées. La bonne façon de les exploiter n'est pas
de les moyenner, c'est de les étiqueter puis de les ajouter à l'entraînement supervisé.

Protocole :
 1. sonde canonique sur les tuiles annotées (moyenne 3x3 du graphe complet) ;
 2. prédiction des 40 030 tuiles non annotées, confiance = probabilité maximale ;
 3. sélection par seuil (et variante équilibrée par classe), ajout au train ;
 4. réentraînement, évaluation sur le test — C re-sélectionné sur validation à chaque fois.

Aucune étiquette de test n'intervient. Le seuil est choisi sur validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_smoothing import f1_macro  # noqa: E402

STRIDE = 112
N_CLS = 11


def graph_full(pos_all, n_lab):
    idx = {p: i for i, p in enumerate(pos_all)}
    rows, cols = [], []
    offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    for i, (o, r, c) in enumerate(pos_all):
        for dr, dc in offs:
            j = idx.get((o, r + STRIDE * dr, c + STRIDE * dc))
            if j is not None:
                rows.append(i)
                cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(pos_all), len(pos_all)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="out/canonical_split.json")
    ap.add_argument("--features", default="out/aux/features_tile.npy")
    ap.add_argument("--aux-features", default="out/aux/aux_features.npy")
    ap.add_argument("--aux-index", default="out/aux/aux_index.json")
    ap.add_argument("--out", default="out/pseudo/pseudo.json")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()

    sys.path.insert(0, a.repo)
    from sklearn.preprocessing import StandardScaler
    from src.utils import make_canonical_lr

    tiles = json.load(open(a.split))["tiles"]
    Xl = np.load(a.features).astype(np.float32)
    Xa = np.load(a.aux_features).astype(np.float32)
    aux = json.load(open(a.aux_index))
    y = np.array([t["label"] for t in tiles], np.int64)
    s = np.array([t["split"] for t in tiles])
    tr, va, te = np.where(s == "train")[0], np.where(s == "val")[0], np.where(s == "test")[0]
    pos_all = [(t["ortho"], t["tile_row_off"], t["tile_col_off"]) for t in tiles]
    pos_all += [(d["ortho"], d["row"], d["col"]) for d in aux]
    X = np.concatenate([Xl, Xa], axis=0)
    n_lab = len(tiles)
    print(f"[pseudo] annotées={n_lab} auxiliaires={len(aux)}", flush=True)

    A = graph_full(pos_all, n_lab)
    A1 = A.copy()
    A1.setdiag(1.0)
    M = np.asarray(A1 @ X) / np.asarray(A1.sum(1)).ravel()[:, None]
    print(f"[pseudo] arêtes={A.nnz}", flush=True)

    def fit_probe(rows, labels, Cgrid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
        sc = StandardScaler().fit(M[rows])
        Ms = sc.transform(M).astype(np.float32)
        best = (-1, None, None)
        for C in Cgrid:
            clf = make_canonical_lr(C=C)
            clf.fit(Ms[rows], labels)
            v = f1_macro(y[va], clf.predict(Ms[va]))
            if v > best[0]:
                best = (v, C, clf)
        v, C, clf = best
        return v, C, clf, Ms

    # 1. sonde de départ sur les seules tuiles annotées
    v0, C0, clf0, _ = fit_probe(tr, y[tr])
    sc0 = StandardScaler().fit(M[tr])
    Pa = clf0.predict_proba(sc0.transform(M[n_lab:]).astype(np.float32))
    conf = Pa.max(1)
    plab = Pa.argmax(1)
    base_te = f1_macro(y[te], clf0.predict(sc0.transform(M[te]).astype(np.float32)))
    print(f"[pseudo] base : C={C0} val={v0:.4f} test={base_te:.4f}", flush=True)
    print(f"[pseudo] aux : confiance médiane={np.median(conf):.3f}, "
          f">0.7 : {(conf > 0.7).mean() * 100:.1f} %, >0.9 : {(conf > 0.9).mean() * 100:.1f} %",
          flush=True)

    res = {"base_val": v0, "base_test": base_te, "C0": C0,
           "aux_conf_median": float(np.median(conf)), "runs": []}

    # 2. seuils simples
    for tau in (0.5, 0.7, 0.9, 0.95):
        keep = conf >= tau
        if keep.sum() < 50:
            continue
        rows = np.concatenate([tr, n_lab + np.where(keep)[0]])
        labs = np.concatenate([y[tr], plab[keep]])
        v, C, clf, Ms = fit_probe(rows, labs)
        t = f1_macro(y[te], clf.predict(Ms[te]))
        res["runs"].append({"policy": f"tau={tau}", "n_ajoutees": int(keep.sum()),
                            "val": v, "test": t, "C": C})
        print(f"[pseudo] tau={tau:<5} +{keep.sum():5d} tuiles  val={v:.4f} test={t:.4f}",
              flush=True)

    # 3. équilibré par classe : même nombre par classe prédite, plus confiantes d'abord
    k = int(0.25 * (conf >= 0.5).sum() / N_CLS)
    if k > 20:
        sel = []
        for c in range(N_CLS):
            idx_c = np.where(plab == c)[0]
            if len(idx_c) == 0:
                continue
            idx_c = idx_c[np.argsort(-conf[idx_c])][:k]
            sel.append(idx_c)
        sel = np.concatenate(sel)
        rows = np.concatenate([tr, n_lab + sel])
        labs = np.concatenate([y[tr], plab[sel]])
        v, C, clf, Ms = fit_probe(rows, labs)
        t = f1_macro(y[te], clf.predict(Ms[te]))
        res["runs"].append({"policy": "equilibre_par_classe", "n_ajoutees": int(len(sel)),
                            "par_classe": k, "val": v, "test": t, "C": C})
        print(f"[pseudo] équilibré  +{len(sel):5d} ({k}/classe)  val={v:.4f} test={t:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    best = max(res["runs"], key=lambda r: r["val"]) if res["runs"] else None
    if best:
        print(f"[pseudo] meilleur sur validation : {best['policy']} test={best['test']:.4f} "
              f"(base {base_te:.4f})", flush=True)
    print("écrit", a.out)


if __name__ == "__main__":
    main()
