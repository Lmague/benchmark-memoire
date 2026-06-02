"""Analyse de l'espace latent : RankMe, anisotropie, courbes couche-par-couche.

Formules EXACTES des notebooks / ``latent_v3_from_embeddings.py`` :
- RankMe (Garrido 2023) : exp de l'entropie de Shannon des valeurs singulières normalisées.
- Anisotropie (Ethayarajh 2019) : cosinus moyen sur ``n_pairs`` paires aléatoires (seed 42).
La séparabilité linéaire est mesurée séparément via le linear probe (:mod:`src.probe`).

Pur numpy — aucun import lourd.
"""
from __future__ import annotations

import numpy as np


def rankme(Z, eps: float = 1e-7) -> float:
    """Rang effectif de ``Z`` [N, D] : exp(-Σ p_k log p_k), p_k = σ_k / Σσ."""
    Z = np.asarray(Z, dtype=np.float32)
    sigmas = np.linalg.svd(Z, compute_uv=False)
    p = sigmas / (sigmas.sum() + eps)
    h = -(p * np.log(p + eps)).sum()
    return float(np.exp(h))


def anisotropy(Z, n_pairs: int = 10000, seed: int = 42) -> float:
    """Cosinus moyen sur ``n_pairs`` paires aléatoires distinctes (embeddings L2-normalisés)."""
    Z = np.asarray(Z, dtype=np.float32)
    rng = np.random.RandomState(seed)
    n = Z.shape[0]
    zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    i = rng.randint(0, n, n_pairs)
    j = rng.randint(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    return float((zn[i] * zn[j]).sum(axis=1).mean())


def l2norm(Z):
    """Normalisation L2 par ligne (eps 1e-12)."""
    Z = np.asarray(Z, dtype=np.float32)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def subsample(E, n: int, seed: int = 42):
    """Sous-échantillonne ``n`` lignes (sans remise) si ``E`` en compte plus (sinon E entier)."""
    E = np.asarray(E)
    if not n or E.shape[0] <= n:
        return E
    rng = np.random.RandomState(seed)
    return E[rng.choice(E.shape[0], n, replace=False)]


def drop_class(E, L, drop_label: int):
    """Retire la classe ``drop_label`` et comble le trou d'indices (labels > drop décalés de -1)."""
    E = np.asarray(E)
    L = np.asarray(L)
    keep = L != drop_label
    e2, l2 = E[keep], L[keep].copy()
    l2[l2 > drop_label] -= 1
    return e2, l2


def metrics_on(E_rankme, E_aniso, n_pairs: int = 10000, subsample_n: int = 20000,
               seed: int = 42) -> dict:
    """RankMe sur ``E_rankme`` et anisotropie sur ``E_aniso`` (chacun sous-échantillonné à 20k).

    NOTE : les résultats de référence (``out/split_v3/results``) calculent RankMe sur le
    train (20k) mais l'anisotropie sur le **val** — d'où deux tableaux séparés. Le script
    ``latent_v3_from_embeddings.py`` les calculait tous deux sur le train ; ce comportement
    est reproductible en passant le même tableau (cf. :func:`layerwise_curves`). Les splits
    sont pilotés par ``config.latent.rankme_split`` / ``anisotropy_split``.
    """
    return {
        "rankme": rankme(subsample(E_rankme, subsample_n, seed)),
        "anisotropy": anisotropy(subsample(E_aniso, subsample_n, seed), n_pairs=n_pairs, seed=seed),
        "dim": int(np.asarray(E_rankme).shape[1]),
    }


def layerwise_curves(layer_feats: dict, n_pairs: int = 10000, subsample_n: int = 20000,
                     seed: int = 42) -> dict:
    """RankMe + anisotropie par couche (même tableau pour les deux) : ``{layer: {...}}``."""
    out = {}
    for li, (E, _L) in sorted(layer_feats.items()):
        out[int(li)] = metrics_on(E, E, n_pairs=n_pairs, subsample_n=subsample_n, seed=seed)
    return out
