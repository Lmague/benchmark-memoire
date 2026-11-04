#!/usr/bin/env python3
"""Corrélations géométrie ↔ F1, recalculées sur la table `geometry_models.csv`.

Les corrélations publiées dans `geometrie.tex` proviennent de sources hétérogènes
(n = 12, 14, 16 ou 19 selon le fichier). Ici tout est recalculé sur **une seule**
table — `results/rapport_data/geometry_models.csv`, protocole identique pour tous les
modèles — et pour trois populations emboîtées :

  all      tous les modèles retenus (registry.CANONICAL_F1)
  frozen   les 9 modèles gelés seulement
  topK     le palier compétitif — les K premiers par F1, K = registry.TIER_K, fixé
           par la rupture mesurée du classement et non par une coupe de rang arbitraire

Pour chaque métrique : Spearman ρ, IC95 bootstrap (n = 2 000, seed 42), p bilatéral,
p corrigé Benjamini-Hochberg, et ρ en leave-one-out (min/max) pour repérer les
corrélations portées par un seul modèle.

Sortie : results/rapport_data/correlations_geometry.csv

    python3 scripts/rapport/correlations.py
"""
from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
from scipy.stats import spearmanr

from registry import CANONICAL_F1, FROZEN_MODELS, OUT, TIER_K, TIER_POP, ensure_out

METRICS = ["rankme_sigma_raw", "rankme_sigma_centered", "eff_rank_sigma2",
           "rankme_sigma_raw_norm", "eff_rank_sigma2_norm", "stable_rank",
           "participation_ratio", "alpha_spectral", "aniso_cosine",
           "aniso_spectral", "fisher_ratio", "knn_purity", "intrinsic_dim_twonn",
           "nc1_canonical", "silhouette", "dbi", "chi", "nc2_mean_cos",
           "logme_train", "logme_test"]
LABELS = {
    "rankme_sigma_raw": "RankMe (σ, bruts)",
    "rankme_sigma_centered": "RankMe (σ, centrés)",
    "eff_rank_sigma2": "Rang effectif (σ², centrés)",
    "rankme_sigma_raw_norm": "RankMe (σ) normalisé",
    "eff_rank_sigma2_norm": "Rang effectif (σ²) normalisé",
    "stable_rank": "Stable rank (= NESum)",
    "participation_ratio": "Participation ratio",
    "alpha_spectral": "Exposant spectral α",
    "aniso_cosine": "Anisotropie (cosinus moyen)",
    "aniso_spectral": "Anisotropie spectrale (λ₁·D/Σλ)",
    "fisher_ratio": "Ratio de Fisher",
    "knn_purity": "Pureté k-NN (k=10)",
    "intrinsic_dim_twonn": "Dimension intrinsèque (TwoNN)",
    "nc1_canonical": "NC1 — dispersion intra / inter",
    "silhouette": "Silhouette",
    "dbi": "Davies-Bouldin",
    "chi": "Calinski-Harabasz",
    "nc2_mean_cos": "NC2 — cosinus moyen des centroïdes",
    "logme_train": "LogME (train)",
    "logme_test": "LogME (test)",
}
N_BOOT, SEED = 2000, 42


def boot_ci(x, y, n=N_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(x), len(x))
        if len(np.unique(idx)) < 3:
            continue
        r = spearmanr(x[idx], y[idx])[0]
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def loo(x, y):
    vals = [spearmanr(np.delete(x, i), np.delete(y, i))[0] for i in range(len(x))]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(min(vals)), float(max(vals))) if vals else (float("nan"),) * 2


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        adj[i] = prev
    return adj


def main():
    ensure_out()
    with open(os.path.join(OUT, "geometry_models.csv")) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["model"] in CANONICAL_F1]
    for r in rows:
        r["f1"] = CANONICAL_F1[r["model"]][0]
    rows.sort(key=lambda r: -r["f1"])

    pops = {
        "all": rows,
        "frozen": [r for r in rows if r["model"] in FROZEN_MODELS],
        TIER_POP: rows[:TIER_K],
    }
    out = []
    for pop, sub in pops.items():
        recs = []
        for m in METRICS:
            vals = [(float(r[m]), r["f1"]) for r in sub if r.get(m) not in ("", None)]
            if len(vals) < 5:
                continue
            x = np.array([v[0] for v in vals])
            y = np.array([v[1] for v in vals])
            rho, pv = spearmanr(x, y)
            lo, hi = boot_ci(x, y)
            lmin, lmax = loo(x, y)
            recs.append({"population": pop, "n": len(vals), "metric": m,
                         "label": LABELS[m], "rho": float(rho), "p": float(pv),
                         "ci95_low": lo, "ci95_high": hi,
                         "loo_rho_min": lmin, "loo_rho_max": lmax,
                         "signif_ci": bool(lo * hi > 0)})
        for rec, q in zip(recs, bh([r["p"] for r in recs])):
            rec["p_bh"] = float(q)
            rec["signif_bh"] = bool(q < 0.05)
        out += recs
        print(f"\n== population « {pop} » (n = {recs[0]['n']}) ==")
        for r in sorted(recs, key=lambda r: -abs(r["rho"]))[:8]:
            flag = "✓" if r["signif_ci"] else " "
            print(f"  {flag} {r['label']:36s} ρ={r['rho']:+.3f}  "
                  f"IC95 [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]  "
                  f"p_BH={r['p_bh']:.3f}  LOO [{r['loo_rho_min']:+.2f}, "
                  f"{r['loo_rho_max']:+.2f}]")

    cols = ["population", "n", "metric", "label", "rho", "ci95_low", "ci95_high",
            "p", "p_bh", "signif_ci", "signif_bh", "loo_rho_min", "loo_rho_max"]
    with open(os.path.join(OUT, "correlations_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\n[OK] correlations_geometry.csv ({len(out)} lignes)")


if __name__ == "__main__":
    main()
