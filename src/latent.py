"""Analyse de l'espace latent : RankMe, anisotropie, α-ReQ, NESum, courbes couche-par-couche.

Formules EXACTES des notebooks / ``latent_v3_from_embeddings.py`` :
- RankMe (Garrido 2023) : exp de l'entropie de Shannon des valeurs singulières normalisées.
- Anisotropie (Ethayarajh 2019) : cosinus moyen sur ``n_pairs`` paires aléatoires (seed 42).
- α-ReQ (Agrawal et al., NeurIPS 2022) : pente (en valeur absolue) d'une régression linéaire
  sur ``log(λⱼ) ~ -α log(j)`` où ``λⱼ`` sont les valeurs propres de la matrice de covariance
  empirique centrée.  Goldilocks zone : α ≈ 1.
- NESum (He & Ozay, ICML 2022, Déf. 4.1) : ``Σᵢ λᵢ / λ₁`` sur le spectre trié de la covariance
  empirique centrée.  Range : [0, D] ; NESum = 0 → collapse, D → blanchi.
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


def rankme_normalized(Z, eps: float = 1e-7) -> float:
    """RankMe normalisé par la dimension : rankme(Z) / D."""
    dim = int(np.asarray(Z).shape[1])
    return rankme(Z, eps=eps) / dim


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


# --------------------------------------------------------------------------- #
# TÂCHE A — α-ReQ (Agrawal et al., NeurIPS 2022) et NESum (He & Ozay, 2022).  #
# Ajoutées 2026-06-14 ; ne modifient AUCUNE fonction existante.              #
# --------------------------------------------------------------------------- #


def _eigvals_centered(Z: np.ndarray) -> np.ndarray:
    """Valeurs propres triées (desc.) de la covariance empirique centrée (1/N) Z_cᵀ Z_c.

    Convention : centrée par colonne (mean = 0).  N, D = Z.shape ; retourne np.ndarray
    de longueur D (spectre complet, pas tronqué).  Helper privé partagé par α-ReQ et
    NESum pour garantir qu'ils opèrent sur la MÊME décomposition spectrale que
    celle utilisée pour RankMe (SVD sur la matrice centrée donne le même spectre
    non-zigzag que ``cov(Z_c)``).
    """
    Z = np.asarray(Z, dtype=np.float32)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # eigh est plus stable que SVD pour une matrice symétrique semi-définie positive
    # ;  on prend la partie symétrique pour blinder contre le bruit d'arrondi.
    C = (Zc.T @ Zc) / Zc.shape[0]
    C = 0.5 * (C + C.T)
    w = np.linalg.eigvalsh(C)        # ascendant
    w = w[::-1]                       # descendant
    w = np.clip(w, 0.0, None)         # rabotage des négatifs dus à l'arrondi
    return w


def alpha_req(Z: np.ndarray, k_max: int | None = None,
              eps: float = 1e-12) -> float:
    """α-ReQ (Agrawal et al., NeurIPS 2022, Def. / Eq. après Theorem 2.1).

    Idée : on suppose ``λⱼ ∝ j^{-α}`` pour ``j ∈ [λ_min, λ_max]``.  On ajuste une
    droite ``log(λⱼ) = -α · log(j) + c`` par moindres carrés (numpy ``polyfit`` deg 1)
    et on retourne ``α = -pente``.

    Paramètres
    ----------
    Z : ndarray [N, D]
        Embeddings (pooler output).
    k_max : int | None
        Borne haute du fit (nombre de plus grandes valeurs propres).  Par défaut
        ``min(N, D)`` (spectre complet), comme dans la Sec. 3.1 de Agrawal 2022.
    eps : float
        Seuil pour ignorer les λⱼ ≤ 0 (numerical zeros).

    Renvoie
    -------
    alpha : float
        Pente (en valeur absolue) de la régression log-log.  Goldilocks zone : α ≈ 1.
        Un α grand (>1) correspond à un decay rapide (encodage sparse), un α petit
        (<1) à un encodage dense.
    """
    w = _eigvals_centered(Z)
    if k_max is None:
        k_max = len(w)
    k_max = min(k_max, len(w))
    w = w[:k_max]
    mask = w > eps
    if mask.sum() < 2:
        # Pas de signal → on retourne NaN pour signaler sans crash
        return float("nan")
    js = np.arange(1, len(w) + 1, dtype=np.float64)[mask]
    ls = np.log(w[mask].astype(np.float64))
    # log(λ) = -α log(j) + c  ⇒  α = -pente
    pente, _intercept = np.polyfit(np.log(js), ls, deg=1)
    return float(-pente)


def nesum(Z: np.ndarray) -> float:
    """NESum (He & Ozay, ICML 2022, Déf. 4.1) : ``Σᵢ λᵢ / λ₁`` sur le spectre trié.

    Notes d'implémentation :
      - on calcule les valeurs propres de la covariance empirique centrée (et non
        de ``Z Zᵀ``), conformément à la Déf. 4.1 qui définit Σ = (1/N) z(X)ᵀ z(X)
        après centrage.
      - λ₁ = plus grande valeur propre (spectre déjà trié en ordre décroissant
        par :func:`_eigvals_centered`).
      - Range théorique : [0, D] (0 = collapse complet ; D = blanchiment parfait).
    """
    w = _eigvals_centered(Z)
    if w[0] <= 0:
        return 0.0
    return float(w.sum() / w[0])


def metrics_with_spectrum(E_rankme: np.ndarray, E_aniso: np.ndarray,
                          n_pairs: int = 10000, subsample_n: int = 20000,
                          seed: int = 42) -> dict:
    """Métriques latentes complètes : RankMe + anisotropie + α-ReQ + NESum.

    RankMe et anisotropie réutilisent :func:`metrics_on` (sur le MÊME sous-échantillon
    de 20 000 tuiles, même seed).  α-ReQ et NESum sont calculés sur le MÊME
    sous-échantillon pour rester comparables (un seul SVD partagé par les quatre
    via :func:`_eigvals_centered`).
    """
    base = metrics_on(E_rankme, E_aniso, n_pairs=n_pairs,
                      subsample_n=subsample_n, seed=seed)
    # Le sous-échantillon de RankMe (20 000) sert aussi au spectre
    Z = subsample(E_rankme, subsample_n, seed)
    base["alpha_req"] = alpha_req(Z)
    base["nesum"] = nesum(Z)
    return base


# Rétro-compat : ``metrics_on`` reste exporté (utilisé par ``analyze.py``).
