#!/usr/bin/env python3
"""Métriques géométriques étendues + séparabilité de classes, et corrélation avec le F1.

Pour robustifier la contribution « géométrie ↔ F1 » au-delà de RankMe + anisotropie
(n=9, IC95 larges), on calcule sur les embeddings TEST de chaque modèle :

  Spectre (sur sous-échantillon 20k, SVD) :
    - rankme               : rang effectif entropique (Garrido 2023)  [déjà dans latent.py]
    - rankme_normalized    : rankme / dim
    - stable_rank          : Σσ² / σ_max²
    - participation_ratio  : (Σσ²)² / Σσ⁴   (dimensionnalité effective)
    - alpha_spectral       : exposant de décroissance loi-puissance du spectre (log-log)
  Voisinage :
    - anisotropy           : cosinus moyen (Ethayarajh 2019)          [déjà dans latent.py]
    - intrinsic_dim_twonn  : dimension intrinsèque TwoNN (Facco 2017)
  Séparabilité de classes (lié mécaniquement au F1) :
    - fisher_ratio         : tr(S_b)/tr(S_w) sur embeddings standardisés
    - knn_purity           : fraction de k=10 plus-proches-voisins de même classe

Puis corrélations Spearman/Kendall (+ IC95 bootstrap) de chaque métrique avec
f1_macro_pres, sur les 9 modèles.

Lit les embeddings TEST directement (``{key}_test.npy`` / ``_test_labels.npy``) — pas
besoin du train. F1 depuis --probe-json. numpy + scipy + sklearn.

Usage :
  python scripts/geometry_extended.py \\
      --config configs/frozen_eval.yaml \\
      --probe-json results/without_rhol/probe_knn_cgrid.json \\
      --output results/geometry_extended.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.latent import anisotropy, rankme, subsample

ALL_MODELS = ["resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
              "simdinov2_vitb16", "dinov3_vitl16_sat", "dinov3_vitl16_lvd",
              "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16"]


# ------------------------------------------------------------------ metrics
def _singular_values(E: np.ndarray) -> np.ndarray:
    Ec = E - E.mean(0, keepdims=True)
    return np.linalg.svd(Ec, compute_uv=False)


def stable_rank(sigmas: np.ndarray) -> float:
    s2 = sigmas.astype(np.float64) ** 2
    return float(s2.sum() / (s2.max() + 1e-12))


def participation_ratio(sigmas: np.ndarray) -> float:
    s2 = sigmas.astype(np.float64) ** 2
    return float((s2.sum() ** 2) / ((s2 ** 2).sum() + 1e-12))


def alpha_spectral(sigmas: np.ndarray, kmin: int = 5, kmax: int = 1000) -> dict:
    """Exposant α de la loi-puissance eig(rank) ∝ rank^(-α) (régression log-log)."""
    eig = (sigmas.astype(np.float64) ** 2)
    eig = eig[eig > 1e-12]
    k = min(kmax, eig.shape[0])
    lo = min(kmin, max(1, k // 10))
    ranks = np.arange(1, k + 1)[lo:]
    vals = eig[:k][lo:]
    if vals.shape[0] < 3:
        return {"alpha_spectral": None, "alpha_r2": None}
    x, y = np.log(ranks), np.log(vals)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    return {"alpha_spectral": float(-slope), "alpha_r2": float(1.0 - ss_res / ss_tot)}


def intrinsic_dim_twonn(E: np.ndarray, n: int = 5000, seed: int = 42) -> float:
    """Dimension intrinsèque TwoNN (Facco 2017) : pente de log(1-F(μ)) vs log(μ)."""
    from sklearn.neighbors import NearestNeighbors
    Es = subsample(E, n, seed).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=3).fit(Es)
    d, _ = nn.kneighbors(Es)
    r1, r2 = d[:, 1], d[:, 2]
    mask = r1 > 1e-12
    mu = (r2[mask] / r1[mask])
    mu = mu[mu > 1.0]
    mu_sorted = np.sort(mu)
    N = mu_sorted.shape[0]
    F = np.arange(1, N + 1) / (N + 1.0)
    x = np.log(mu_sorted)
    y = -np.log(1.0 - F)
    # régression par l'origine : d = Σ x y / Σ x²
    d_est = float((x * y).sum() / ((x * x).sum() + 1e-12))
    return d_est


def fisher_ratio(E: np.ndarray, L: np.ndarray) -> float:
    """tr(S_b)/tr(S_w) sur embeddings standardisés (séparabilité linéaire globale)."""
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(E.astype(np.float64))
    mu = X.mean(0)
    sb = 0.0
    sw = 0.0
    for c in np.unique(L):
        Xc = X[L == c]
        muc = Xc.mean(0)
        sb += Xc.shape[0] * float(((muc - mu) ** 2).sum())
        sw += float(((Xc - muc) ** 2).sum())
    return float(sb / (sw + 1e-12))


def knn_purity(E: np.ndarray, L: np.ndarray, k: int = 10, n: int = 5000,
               seed: int = 42) -> float:
    """Fraction moyenne de k plus-proches-voisins partageant la classe (embeddings L2)."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.RandomState(seed)
    idx = rng.choice(E.shape[0], min(n, E.shape[0]), replace=False)
    Es = E[idx].astype(np.float32)
    Ls = L[idx]
    Es = Es / (np.linalg.norm(Es, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Es)
    _, ind = nn.kneighbors(Es)
    neigh = Ls[ind[:, 1:]]                       # exclure self
    return float((neigh == Ls[:, None]).mean())


def compute_all(E: np.ndarray, L: np.ndarray, subsample_n: int = 20000,
                seed: int = 42) -> dict:
    Es = subsample(E, subsample_n, seed)
    sig = _singular_values(Es)
    dim = int(E.shape[1])
    rm = rankme(Es)
    out = {
        "dim": dim,
        "rankme": rm,
        "rankme_normalized": rm / dim,
        "stable_rank": stable_rank(sig),
        "participation_ratio": participation_ratio(sig),
        "anisotropy": anisotropy(Es, seed=seed),
        "intrinsic_dim_twonn": intrinsic_dim_twonn(E, seed=seed),
        "fisher_ratio": fisher_ratio(Es, subsample(L.reshape(-1, 1), subsample_n, seed).ravel()
                                     if E.shape[0] > subsample_n else L),
        "knn_purity": knn_purity(E, L, seed=seed),
    }
    out.update(alpha_spectral(sig))
    return out


# ------------------------------------------------------------------ correlations
def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def _kendall(x, y):
    from scipy.stats import kendalltau
    return float(kendalltau(x, y)[0])


def corr_with_ci(x, y, n_boot=1000, seed=42):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return {"n": n, "spearman_r": None, "kendall_tau": None}
    rng = np.random.RandomState(seed)
    sp = np.empty(n_boot)
    kt = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        sp[b] = _spearman(x[idx], y[idx])
        kt[b] = _kendall(x[idx], y[idx])
    return {
        "n": n,
        "spearman_r": _spearman(x, y),
        "spearman_ci95": [float(np.nanpercentile(sp, 2.5)), float(np.nanpercentile(sp, 97.5))],
        "kendall_tau": _kendall(x, y),
        "kendall_ci95": [float(np.nanpercentile(kt, 2.5)), float(np.nanpercentile(kt, 97.5))],
    }


def _load_f1(probe_json: str) -> dict:
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["test"]["f1_macro_pres"]) for m, d in probe.items()
            if "test" in d and "f1_macro_pres" in d["test"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--probe-json", default="results/without_rhol/probe_knn_cgrid.json")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--output", default="results/geometry_extended.json")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    emb_dir = cfg.paths.emb_dir
    models = args.models or ALL_MODELS
    f1 = _load_f1(args.probe_json)

    metrics: dict[str, dict] = {}
    for m in models:
        ep = os.path.join(emb_dir, f"{m}_test.npy")
        lp = os.path.join(emb_dir, f"{m}_test_labels.npy")
        if not (os.path.exists(ep) and os.path.exists(lp)):
            print(f"[skip] {m}: embeddings test absents")
            continue
        E = np.load(ep).astype(np.float32)
        L = np.load(lp).astype(np.int64)
        metrics[m] = compute_all(E, L, seed=args.seed)
        mm = metrics[m]
        print(f"[geom] {m:22} RankMe={mm['rankme']:7.1f} stable={mm['stable_rank']:6.1f} "
              f"PR={mm['participation_ratio']:6.1f} alpha={mm['alpha_spectral']:.2f} "
              f"twoNN={mm['intrinsic_dim_twonn']:5.1f} Fisher={mm['fisher_ratio']:.3f} "
              f"kNNpur={mm['knn_purity']:.3f}")

    # Corrélations métrique ↔ F1 (modèles ayant métriques + F1)
    metric_keys = ["rankme", "rankme_normalized", "stable_rank", "participation_ratio",
                   "alpha_spectral", "anisotropy", "intrinsic_dim_twonn",
                   "fisher_ratio", "knn_purity"]
    avail = [m for m in metrics if m in f1]
    correlations = {}
    for mk in metric_keys:
        xs = [metrics[m][mk] for m in avail if metrics[m].get(mk) is not None]
        ys = [f1[m] for m in avail if metrics[m].get(mk) is not None]
        correlations[mk] = corr_with_ci(xs, ys, n_boot=args.n_bootstrap, seed=args.seed)
        correlations[mk]["models"] = [m for m in avail if metrics[m].get(mk) is not None]

    out = {
        "probe_source": args.probe_json,
        "n_models": len(metrics),
        "n_bootstrap": args.n_bootstrap,
        "metrics": metrics,
        "f1_values_used": {m: f1[m] for m in avail},
        "correlations_vs_f1": correlations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}  ({len(metrics)} modèles)")
    print("\nCorrélations métrique ↔ f1_macro_pres (n=%d) :" % len(avail))
    for mk in metric_keys:
        c = correlations[mk]
        if c.get("spearman_r") is None:
            continue
        ci = c.get("spearman_ci95", ["—", "—"])
        print(f"  {mk:22} ρ={c['spearman_r']:+.3f}  IC95=[{ci[0]:+.2f},{ci[1]:+.2f}]  "
              f"τ={c['kendall_tau']:+.3f}")


if __name__ == "__main__":
    main()
