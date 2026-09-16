"""Descripteurs de texture « classiques » par tuile — familles GLCM/GLRLM/GLSZM/GLDM/NGTDM,
ondelettes et spectre de Fourier.

Pourquoi ce module existe
-------------------------
Le benchmark Arctic-TVC ne teste que des modèles de fondation sur du RGB. Or le GSD de
2,2 mm/px place une grande partie du signal discriminatif SOUS le patch d'un ViT (16 px
= 3,5 cm) : la texture fine — moucheture du lichen, tapis de mousse, linaires de
linaigrette — est lissée avant d'entrer dans le réseau. Ces familles sont l'outil standard
de la télédétection de végétation pour capturer cette échelle (Kulich et al. 2026,
doi:10.3389/fpls.2026.1841696 ; Deng et al. 2022, doi:10.1038/s41598-022-17620-2), et elles
sont ici utilisées comme features d'appoint CONCATÉNÉES à l'embedding — jamais comme entrée
du patch embedding d'un ViT gelé (cf. en-tête de scripts/texture_features.py).

Pourquoi pas PyRadiomics
------------------------
PyRadiomics est l'implémentation de référence (conforme IBSI) mais ne s'installe pas sur
cette machine (``metadata-generation-failed``, échec de compilation) et ajoute une
dépendance C fragile. Ce module réimplémente les matrices en numpy/scipy pur :
déterministe, sans compilation, vérifiable par tests unitaires (tests/test_texture_features.py).

Définitions des matrices
------------------------
==========================  ======================================================  ==========
Famille                     Unité comptée                                           Rotation
==========================  ======================================================  ==========
GLCM  (Haralick 1973)       paires de pixels à un décalage (d, θ)                   non
GLRLM (Galloway 1975)       segments (« runs ») de même niveau, direction θ         non
GLSZM (Thibault 2009)       zones connexes de même niveau (8-connexité)             oui
GLDM  (Sun & Wee 1983)      voisins dépendants (|Δniveau| ≤ δ) en 8-connexité       oui
NGTDM (Amadasun & King 89)  différence au niveau moyen du voisinage                 oui
==========================  ======================================================  ==========

C'est cette complémentarité (transitions vs linéarité vs granularité vs dépendance) qui
justifie de les tester séparément puis combinées — voir scripts/texture_ablation.py.

Convention de quantification
----------------------------
La quantification se fait sur une plage FIXE (0–255 par défaut), pas sur le min/max de
chaque tuile : sinon deux tuiles de luminosité différente ne sont plus comparables (IBSI
recommande une largeur de bin fixe). ``quantize(gray, levels)`` produit des entiers dans
``[0, levels-1]``. **Le nombre de features est FIXE** pour un jeu de paramètres donné : le
design matrix garde les mêmes colonnes sur toutes les tuiles, ce dont dépend la sonde.

Convention de nommage des features
----------------------------------
``<famille>[_<paramètre>]_<nom>`` — ex. ``glcm_d1_contrast``, ``glszm_lae``, ``dwt_L2_HH``.
Le préfixe avant le premier ``_`` est la FAMILLE, ce qui permet à scripts/texture_ablation.py
d'isoler une famille sans table de correspondance à maintenir (cf. :func:`family_of`).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "FAMILIES",
    "GLCM_SUFFIXES",
    "GLRLM_SUFFIXES",
    "GLSZM_SUFFIXES",
    "GLDM_SUFFIXES",
    "NGTDM_SUFFIXES",
    "DWT_SUFFIXES",
    "FOURIER_SUFFIXES",
    "to_gray",
    "quantize",
    "tile_features",
    "family_of",
]

#: Ordre canonique des familles (utilisé par la table d'ablation).
FAMILIES: tuple[str, ...] = ("glcm", "glrlm", "glszm", "gldm", "ngtdm", "dwt", "fourier")

#: Suffixes de features par famille — SOURCE UNIQUE DE VÉRITÉ.
#: Toute feature produite doit figurer ici, et réciproquement : un décalage entre les deux
#: produirait un design matrix à largeur variable selon la tuile (cf. tests unitaires).
GLCM_SUFFIXES: tuple[str, ...] = ("mean", "std", "correlation", "contrast", "dissimilarity",
                                  "homogeneity", "asm", "energy", "maximum", "entropy")
GLRLM_SUFFIXES: tuple[str, ...] = ("sre", "lre", "gln", "glnn", "rln", "rlnn", "rp", "glv",
                                   "rv", "re", "lglre", "hglre", "srlgle", "srhgle",
                                   "lrlgre", "lrhgle")
GLSZM_SUFFIXES: tuple[str, ...] = ("sae", "lae", "gln", "glnn", "szn", "sznn", "zp", "glv",
                                   "zv", "ze", "lglze", "hglze", "salgale", "sahgle",
                                   "lalgale", "lahgle")
GLDM_SUFFIXES: tuple[str, ...] = ("sde", "lde", "gln", "glnn", "dn", "dnn", "de", "lgle",
                                  "hgle", "glv", "dv", "sdlgle", "sdhgle", "ldlgle", "ldhgle")
NGTDM_SUFFIXES: tuple[str, ...] = ("coarseness", "contrast", "busyness", "complexity", "strength")

# Offsets (dy, dx) des 4 directions : 0°, 45°, 90°, 135°.
_OFFSETS: tuple[tuple[int, int], ...] = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
_EPS = 1e-12
#: NGTDM : eps dédié, aligné sur PyRadiomics (1e-6) et non 1e-12. Une tuile uniforme a
#: s_i = 0 pour tout i, donc coarseness = 1/eps : avec 1e-12 la feature vaudrait 1e12 et
#: écraserait la standardisation sur les 49 281 tuiles de train.
_EPS_NGTDM = 1e-6


def family_of(name: str) -> str:
    """Famille d'une feature (préfixe avant le premier ``_``)."""
    return name.split("_", 1)[0]


# ─────────────────────────────────────────────────────────────── image → niveaux


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB (H, W, 3) uint8 → luminance uint8 (ITU-R BT.601, la convention des codecs)."""
    a = np.asarray(rgb)
    if a.ndim == 2:
        return a.astype(np.uint8, copy=False)
    if a.shape[-1] == 1:
        return a[..., 0].astype(np.uint8, copy=False)
    if a.shape[-1] < 3:
        raise ValueError(f"image à {a.shape[-1]} canaux, 3 attendus (RGB)")
    f = a[..., :3].astype(np.float32)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return np.clip(np.rint(y), 0, 255).astype(np.uint8)


def quantize(gray: np.ndarray, levels: int = 16, lo: int = 0, hi: int = 255) -> np.ndarray:
    """Quantifie ``gray`` sur ``levels`` niveaux dans la plage FIXE ``[lo, hi]``.

    Plage fixe (et non min/max de la tuile) : deux tuiles de luminosité différente doivent
    tomber dans les mêmes bins, sinon les features ne sont plus comparables entre tuiles.
    """
    if levels < 2:
        raise ValueError(f"levels doit être ≥ 2 (reçu {levels})")
    g = np.asarray(gray).astype(np.float32)
    span = float(hi - lo) if hi > lo else 1.0
    q = np.floor((g - lo) / span * levels).astype(np.int64)
    return np.clip(q, 0, levels - 1).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────── GLCM


def _glcm_matrix(q: np.ndarray, levels: int, d: int, dy: int, dx: int) -> np.ndarray:
    """Matrice de co-occurrence symétrique normalisée, pour un décalage ``(dy, dx) * d``."""
    h, w = q.shape
    ys, xs = max(0, -dy * d), max(0, -dx * d)
    ye, xe = h - max(0, dy * d), w - max(0, dx * d)
    if ye <= ys or xe <= xs:
        return np.zeros((levels, levels), dtype=np.float64)
    a = q[ys:ye, xs:xe].astype(np.int64)
    b = q[ys + dy * d:ye + dy * d, xs + dx * d:xe + dx * d].astype(np.int64)
    idx = a.ravel() * levels + b.ravel()
    P = np.bincount(idx, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    P += P.T                       # convention symétrique (Haralick / IBSI / PyRadiomics)
    s = P.sum()
    return P / s if s > 0 else P


def _glcm_features(P: np.ndarray, levels: int, tag: str) -> dict[str, float]:
    """Les métriques de Hall-Beyer (celles retenues par Kulich et al. 2026)."""
    i = np.arange(levels, dtype=np.float64)[:, None]
    j = np.arange(levels, dtype=np.float64)[None, :]
    marg = P.sum(axis=1)
    mean = float((i[:, 0] * marg).sum())
    var = float((((i[:, 0] - mean) ** 2) * marg).sum())
    std = float(np.sqrt(max(var, 0.0)))
    sd_j = float(np.sqrt(max((((j[0] - mean) ** 2) * P.sum(axis=0)).sum(), 0.0)))
    corr = (float((((i - mean) * (j - mean)) * P).sum() / (std * sd_j))
            if std > _EPS and sd_j > _EPS else 0.0)
    nz = P > 0
    vals = {
        "mean": mean,
        "std": std,
        "correlation": corr,
        "contrast": float((((i - j) ** 2) * P).sum()),
        "dissimilarity": float((np.abs(i - j) * P).sum()),
        "homogeneity": float((P / (1.0 + (i - j) ** 2)).sum()),
        "asm": float((P ** 2).sum()),
        "energy": float(np.sqrt((P ** 2).sum())),
        "maximum": float(P.max()),
        "entropy": float(-(P[nz] * np.log2(P[nz])).sum()),
    }
    return {f"{tag}_{k}": vals[k] for k in GLCM_SUFFIXES}


def glcm_features(q: np.ndarray, levels: int, distances: tuple[int, ...]) -> dict[str, float]:
    """GLCM moyennée sur les 4 directions (0/45/90/135°), une série par distance."""
    out: dict[str, float] = {}
    for d in distances:
        acc: dict[str, float] = {}
        for dy, dx in _OFFSETS:
            for k, v in _glcm_features(_glcm_matrix(q, levels, d, dy, dx),
                                       levels, f"glcm_d{d}").items():
                acc[k] = acc.get(k, 0.0) + v
        n = float(len(_OFFSETS))
        out.update({k: v / n for k, v in acc.items()})
    return out


# ────────────────────────────────────────────────────────────────────── GLRLM


def _lines(q: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Lignes de balayage selon la direction ``(dy, dx)``, paddées à la longueur max.

    Le padding vaut ``MAX_LEVEL`` (255) — une valeur qui ne peut jamais être un niveau
    valide après quantification (≤ 255 mais le padding est filtré par ``lvl < levels``
    dans :func:`_rl_hist`). Il ne crée donc pas de segments fantômes comptés.
    """
    h, w = q.shape
    if dy == 0 and dx == 1:                       # 0° : lignes
        return q
    if dy == -1 and dx == 0:                      # 90° : colonnes
        return np.ascontiguousarray(q.T)
    if dy == -1 and dx == 1:                      # 45° : anti-diagonales
        src = np.fliplr(q)
    elif dy == -1 and dx == -1:                   # 135° : diagonales
        src = q
    else:
        raise ValueError(f"direction non gérée : ({dy}, {dx})")
    diags = [np.diagonal(src, offset=k) for k in range(-h + 1, w)]
    lmax = max(int(d.size) for d in diags)
    out = np.full((len(diags), lmax), 255, dtype=np.uint8)
    for r, d in enumerate(diags):
        out[r, :d.size] = d
    return out


def _rl_hist(lines: np.ndarray, levels: int) -> np.ndarray:
    """P(i, j) = nombre de segments de niveau i de longueur j (j indexé 0..L-1)."""
    n, L = lines.shape
    a = lines.astype(np.int64)
    change = np.ones((n, L), dtype=bool)
    change[:, 1:] = a[:, 1:] != a[:, :-1]
    run_id = np.cumsum(change, axis=1) - 1
    ids = (run_id + np.arange(n, dtype=np.int64)[:, None] * L).ravel()
    vals = a.ravel()
    order = np.argsort(ids, kind="stable")
    ids_s, vals_s = ids[order], vals[order]
    starts = np.flatnonzero(np.r_[True, ids_s[1:] != ids_s[:-1]])
    counts = np.diff(np.r_[starts, ids_s.size]).astype(np.int64)
    lvl = vals_s[starts]
    valid = lvl < levels
    P = np.zeros((levels, L), dtype=np.float64)
    if valid.any():
        np.add.at(P, (lvl[valid], counts[valid] - 1), 1.0)
    return P


def _glrlm_features(P: np.ndarray, levels: int, n_pixels: int, tag: str) -> dict[str, float]:
    """Les features IBSI de la GLRLM (i et j indexés à partir de 1)."""
    zero = {f"{tag}_{s}": 0.0 for s in GLRLM_SUFFIXES}
    n_r = float(P.sum())
    if n_r <= 0:
        return zero
    L = P.shape[1]
    i = np.arange(1, levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, L + 1, dtype=np.float64)[None, :]
    pi = P.sum(axis=1)
    pj = P.sum(axis=0)
    mu_i = float((i[:, 0] * pi).sum() / n_r)
    mu_j = float((j[0] * pj).sum() / n_r)
    vals = {
        "sre": float((P / j ** 2).sum() / n_r),
        "lre": float((P * j ** 2).sum() / n_r),
        "gln": float((pi ** 2).sum() / n_r),
        "glnn": float((pi ** 2).sum() / n_r ** 2),
        "rln": float((pj ** 2).sum() / n_r),
        "rlnn": float((pj ** 2).sum() / n_r ** 2),
        "rp": float(n_r / n_pixels) if n_pixels > 0 else 0.0,
        "glv": float((pi * (i[:, 0] - mu_i) ** 2).sum() / n_r),
        "rv": float((pj * (j[0] - mu_j) ** 2).sum() / n_r),
        "re": float(-(P[P > 0] * np.log2(P[P > 0])).sum() / n_r),
        "lglre": float((P / i ** 2).sum() / n_r),
        "hglre": float((P * i ** 2).sum() / n_r),
        "srlgle": float((P / (i ** 2 * j ** 2)).sum() / n_r),
        "srhgle": float((P * i ** 2 / j ** 2).sum() / n_r),
        "lrlgre": float((P * j ** 2 / i ** 2).sum() / n_r),
        "lrhgle": float((P * i ** 2 * j ** 2).sum() / n_r),
    }
    return {f"{tag}_{k}": vals[k] for k in GLRLM_SUFFIXES}


def glrlm_features(q: np.ndarray, levels: int) -> dict[str, float]:
    """GLRLM moyennée sur les 4 directions (0/45/90/135°)."""
    acc: dict[str, float] = {}
    for dy, dx in _OFFSETS:
        f = _glrlm_features(_rl_hist(_lines(q, dy, dx), levels), levels, int(q.size), "glrlm")
        for k, v in f.items():
            acc[k] = acc.get(k, 0.0) + v
    n = float(len(_OFFSETS))
    return {k: v / n for k, v in acc.items()}


# ────────────────────────────────────────────────────────────────────── GLSZM


def _glszm_matrix(q: np.ndarray, levels: int) -> np.ndarray:
    """P(i, j) = nombre de zones connexes de niveau i et de taille j (8-connexité)."""
    from scipy import ndimage as ndi
    st = np.ones((3, 3), dtype=bool)
    sizes_per_level: list[np.ndarray] = []
    max_sz = 1
    for g in range(levels):
        mask = q == g
        if not mask.any():
            sizes_per_level.append(np.zeros(0, dtype=np.int64))
            continue
        lab, n = ndi.label(mask, structure=st)
        if n == 0:
            sizes_per_level.append(np.zeros(0, dtype=np.int64))
            continue
        sizes = np.bincount(lab.ravel())[1:].astype(np.int64)
        sizes_per_level.append(sizes)
        max_sz = max(max_sz, int(sizes.max()))
    P = np.zeros((levels, max_sz), dtype=np.float64)
    for g, sizes in enumerate(sizes_per_level):
        if sizes.size:
            np.add.at(P[g], sizes - 1, 1.0)
    return P


def _glszm_features(P: np.ndarray, levels: int, n_pixels: int, tag: str) -> dict[str, float]:
    """Les features IBSI de la GLSZM (i et j indexés à partir de 1)."""
    n_z = float(P.sum())
    if n_z <= 0:
        return {f"{tag}_{s}": 0.0 for s in GLSZM_SUFFIXES}
    L = P.shape[1]
    i = np.arange(1, levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, L + 1, dtype=np.float64)[None, :]
    pi = P.sum(axis=1)
    pj = P.sum(axis=0)
    mu_i = float((i[:, 0] * pi).sum() / n_z)
    mu_j = float((j[0] * pj).sum() / n_z)
    vals = {
        "sae": float((P / j ** 2).sum() / n_z),
        "lae": float((P * j ** 2).sum() / n_z),
        "gln": float((pi ** 2).sum() / n_z),
        "glnn": float((pi ** 2).sum() / n_z ** 2),
        "szn": float((pj ** 2).sum() / n_z),
        "sznn": float((pj ** 2).sum() / n_z ** 2),
        "zp": float(n_z / n_pixels) if n_pixels > 0 else 0.0,
        "glv": float((pi * (i[:, 0] - mu_i) ** 2).sum() / n_z),
        "zv": float((pj * (j[0] - mu_j) ** 2).sum() / n_z),
        "ze": float(-(P[P > 0] * np.log2(P[P > 0])).sum() / n_z),
        "lglze": float((P / i ** 2).sum() / n_z),
        "hglze": float((P * i ** 2).sum() / n_z),
        "salgale": float((P / (i ** 2 * j ** 2)).sum() / n_z),
        "sahgle": float((P * i ** 2 / j ** 2).sum() / n_z),
        "lalgale": float((P * j ** 2 / i ** 2).sum() / n_z),
        "lahgle": float((P * i ** 2 * j ** 2).sum() / n_z),
    }
    return {f"{tag}_{k}": vals[k] for k in GLSZM_SUFFIXES}


# ─────────────────────────────────────────────────────────────────────── GLDM


def _gldm_matrix(q: np.ndarray, levels: int, alpha: int, delta: int) -> np.ndarray:
    """P(i, j) = nombre de pixels de niveau i ayant j voisins dépendants (|Δ| ≤ delta)."""
    h, w = q.shape
    a = q.astype(np.int16)
    dep = np.zeros((h, w), dtype=np.int16)
    for dy in range(-alpha, alpha + 1):
        for dx in range(-alpha, alpha + 1):
            if dy == 0 and dx == 0:
                continue
            ys, xs = max(0, -dy), max(0, -dx)
            ye, xe = h - max(0, dy), w - max(0, dx)
            if ye <= ys or xe <= xs:
                continue
            dep[ys:ye, xs:xe] += (np.abs(a[ys:ye, xs:xe] - a[ys + dy:ye + dy, xs + dx:xe + dx]) <= delta)
    jmax = int(dep.max()) if dep.size else 0
    P = np.zeros((levels, jmax + 1), dtype=np.float64)
    np.add.at(P, (a.ravel().astype(np.int64), dep.ravel().astype(np.int64)), 1.0)
    return P


def _gldm_features(P: np.ndarray, levels: int, tag: str) -> dict[str, float]:
    """Les features IBSI de la GLDM (i et j indexés à partir de 1)."""
    n_p = float(P.sum())
    if n_p <= 0:
        return {f"{tag}_{s}": 0.0 for s in GLDM_SUFFIXES}
    J = P.shape[1]
    i = np.arange(1, levels + 1, dtype=np.float64)[:, None]
    j = np.arange(1, J + 1, dtype=np.float64)[None, :]
    pi = P.sum(axis=1)
    pj = P.sum(axis=0)
    mu_i = float((i[:, 0] * pi).sum() / n_p)
    mu_j = float((j[0] * pj).sum() / n_p)
    vals = {
        "sde": float((P / j ** 2).sum() / n_p),
        "lde": float((P * j ** 2).sum() / n_p),
        "gln": float((pi ** 2).sum() / n_p),
        "glnn": float((pi ** 2).sum() / n_p ** 2),
        "dn": float((pj ** 2).sum() / n_p),
        "dnn": float((pj ** 2).sum() / n_p ** 2),
        "de": float(-(P[P > 0] * np.log2(P[P > 0])).sum() / n_p),
        "lgle": float((P / i ** 2).sum() / n_p),
        "hgle": float((P * i ** 2).sum() / n_p),
        "glv": float((pi * (i[:, 0] - mu_i) ** 2).sum() / n_p),
        "dv": float((pj * (j[0] - mu_j) ** 2).sum() / n_p),
        "sdlgle": float((P / (i ** 2 * j ** 2)).sum() / n_p),
        "sdhgle": float((P * i ** 2 / j ** 2).sum() / n_p),
        "ldlgle": float((P * j ** 2 / i ** 2).sum() / n_p),
        "ldhgle": float((P * i ** 2 * j ** 2).sum() / n_p),
    }
    return {f"{tag}_{k}": vals[k] for k in GLDM_SUFFIXES}


# ────────────────────────────────────────────────────────────────────── NGTDM


def ngtdm_features(q: np.ndarray, levels: int) -> dict[str, float]:
    """Les 5 features d'Amadasun & King (1989) : coarseness, contrast, busyness,
    complexity, strength. NGTDM n'a pas de paramètre δ dans l'implémentation de référence
    (le voisinage est le 3×3 immédiat) ; les formules suivent PyRadiomics/IBSI.
    """
    from scipy.ndimage import convolve
    k = np.ones((3, 3), dtype=np.float64)
    k[1, 1] = 0.0
    k /= k.sum()
    a = q.astype(np.float64)
    diff = np.abs(a - convolve(a, k, mode="reflect"))
    counts = np.bincount(q.ravel(), minlength=levels).astype(np.float64)
    n_p = float(q.size)
    p = counts / n_p
    present = counts > 0
    idx = np.flatnonzero(present).astype(np.float64)
    pv = p[present]
    ssum = np.bincount(q.ravel(), weights=diff.ravel(), minlength=levels)
    sv = np.divide(ssum[present], counts[present], out=np.zeros_like(pv), where=counts[present] > 0)
    n_gp = float(present.sum())
    s_tot = float(sv.sum())
    Pi, Pj = pv[:, None], pv[None, :]
    Ii, Ij = idx[:, None], idx[None, :]
    d2 = (Ii - Ij) ** 2
    ps = float((pv * sv).sum())
    coarseness = 1.0 / (ps + _EPS_NGTDM)
    contrast = (float((Pi * Pj * d2).sum()) / (n_gp * (n_gp - 1.0) + _EPS)) * (s_tot / (n_gp + _EPS))
    busyness = ps / (float(np.abs(Ii * Pi - Ij * Pj).sum()) + _EPS_NGTDM)
    complexity = float((np.abs(Ii - Ij) * (Pi * sv[:, None] + Pj * sv[None, :])
                        / (Pi + Pj + _EPS)).sum()) / n_p
    strength = float(((Pi + Pj) * d2).sum()) / (s_tot + _EPS_NGTDM)
    vals = {"coarseness": coarseness, "contrast": contrast, "busyness": busyness,
            "complexity": complexity, "strength": strength}
    return {f"ngtdm_{k2}": vals[k2] for k2 in NGTDM_SUFFIXES}


# ───────────────────────────────────────────────────────────────── ondelettes


def dwt_features(gray: np.ndarray, wavelet: str = "db4", level: int = 3) -> dict[str, float]:
    """Énergie RELATIVE de chaque sous-bande d'une DWT 2D, par niveau.

    Relative (÷ énergie totale) pour être comparable entre tuiles de luminosité
    différente. L'orientation est portée par ``LH``/``HL``/``HH`` : ``HH`` = contenu haute
    fréquence fin (moucheture), ``LH``/``HL`` = structures orientées (linaires, faisceaux).
    """
    import pywt
    a = gray.astype(np.float32)
    coeffs = pywt.wavedec2(a, wavelet=wavelet, level=level, mode="periodization")
    total = float((a ** 2).sum()) + _EPS
    out: dict[str, float] = {}
    out[f"dwt_L{level}_LL"] = float((coeffs[0] ** 2).sum()) / total
    # Ordre d'émission = ordre de dwt_suffixes (L1 = niveau le plus FIN, LL = le plus
    # grossier). pywt indexe l'inverse (coeffs[1] = détail le plus grossier), d'où le
    # renversement. Le contrat d'ordre est de toute façon imposé par tile_features().
    for idx in range(1, level + 1):
        cH, cV, cD = coeffs[level - idx + 1]
        out[f"dwt_L{idx}_LH"] = float((cH ** 2).sum()) / total
        out[f"dwt_L{idx}_HL"] = float((cV ** 2).sum()) / total
        out[f"dwt_L{idx}_HH"] = float((cD ** 2).sum()) / total
    return out


def dwt_suffixes(level: int = 3) -> tuple[str, ...]:
    """Noms de sous-bandes produits par :func:`dwt_features` (dimensions fixes)."""
    names = [f"dwt_L{level}_LL"]
    for idx in range(1, level + 1):
        names += [f"dwt_L{idx}_LH", f"dwt_L{idx}_HL", f"dwt_L{idx}_HH"]
    return tuple(names)


# ──────────────────────────────────────────────────────────────── Fourier (PSD)


def fourier_features(gray: np.ndarray, n_bins: int = 8, n_bands: int = 4,
                     fit_half: float = 0.5) -> dict[str, float]:
    """Pente spectrale β, profil radial par bandes d'octave et profil angulaire.

    - ``fourier_beta``       : exposant spectral, convention ``P(r) ∝ r^(−β)`` → **β > 0
                               pour une image lisse**, β ≈ 0 pour du bruit blanc
                               (β ≈ 2 pour une image naturelle). La régression log-log est
                               restreinte aux basses fréquences (``fit_half`` de ``rmax``)
                               sinon le plancher de bruit haute fréquence écrase la pente.
                               DÉGÉNÉRÉ pour un motif périodique (damier) : le spectre est
                               concentré sur une raie, la pente n'a plus de sens.
    - ``fourier_bandK``      : énergie relative par bande (profil radial, log-espacé).
    - ``fourier_angK``       : énergie relative par secteur d'orientation — capture les
                               structures orientées que GLCM supprime en moyennant les
                               directions.
    - ``fourier_anisotropy`` : (max − min)/(max + min) des secteurs angulaires.
    """
    from numpy.fft import fft2, fftshift
    a = gray.astype(np.float32)
    h, w = a.shape
    win = (a - a.mean()) * np.hanning(h)[:, None] * np.hanning(w)[None, :]
    P = np.abs(fftshift(fft2(win))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    rmax = int(min(cy, cx))
    cnt = np.bincount(r.ravel(), minlength=rmax + 1)[1:rmax + 1].astype(np.float64)
    prof = np.bincount(r.ravel(), weights=P.ravel(), minlength=rmax + 1)[1:rmax + 1]
    prof = prof / np.maximum(cnt, 1.0)
    radii = np.arange(1, rmax + 1, dtype=np.float64)
    out: dict[str, float] = {"fourier_beta": 0.0}
    # Fit sur les BASSES fréquences seulement : au-delà, le spectre est un plancher de
    # bruit et la pente log-log ne mesure plus la rugosité. Le signe est renversé pour
    # respecter la convention P(r) ∝ r^(−β) (β > 0 = image lisse).
    ok = (prof > 0) & (radii <= max(int(rmax * fit_half), 4))
    if ok.sum() >= 4:
        slope = np.polyfit(np.log(radii[ok]), np.log(prof[ok]), 1)[0]
        out["fourier_beta"] = -float(slope)
    tot = float(prof.sum()) + _EPS
    # Bandes garanties à n_bands éléments, quel que soit rmax (log2 → taille fixe).
    band = np.minimum((np.log2(radii) / (np.log2(rmax + 1) + _EPS) * n_bands).astype(np.int64),
                      n_bands - 1)
    bsum = np.bincount(band, weights=prof, minlength=n_bands)[:n_bands]
    for kb in range(n_bands):
        out[f"fourier_band{kb + 1}"] = float(bsum[kb]) / tot
    # Profil angulaire (n_bins garanti).
    sel = (r > 0) & (r <= rmax)
    ang = np.arctan2((yy - cy).astype(np.float64), (xx - cx).astype(np.float64))
    abin = np.minimum(((ang[sel] + np.pi) / (2 * np.pi) * n_bins).astype(np.int64), n_bins - 1)
    pw = np.bincount(abin, weights=P[sel], minlength=n_bins)[:n_bins]
    pw = pw / (float(pw.sum()) + _EPS)
    for ka in range(n_bins):
        out[f"fourier_ang{ka + 1}"] = float(pw[ka])
    wmax, wmin = float(pw.max()), float(pw.min())
    out["fourier_anisotropy"] = (wmax - wmin) / (wmax + wmin + _EPS)
    return out


def fourier_suffixes(n_bins: int = 8, n_bands: int = 4) -> tuple[str, ...]:
    """Noms produits par :func:`fourier_features` (dimensions fixes)."""
    return (("fourier_beta",)
            + tuple(f"fourier_band{k + 1}" for k in range(n_bands))
            + tuple(f"fourier_ang{k + 1}" for k in range(n_bins))
            + ("fourier_anisotropy",))


def expected_feature_names(
    levels: int = 16,
    distances: tuple[int, ...] = (1, 2, 4),
    with_dwt: bool = True,
    with_fourier: bool = True,
    wavelet: str = "db4",
    dwt_level: int = 3,
    fourier_bins: int = 8,
    fourier_bands: int = 4,
    gldm_alpha: int = 1,
    gldm_delta: int = 0,
) -> tuple[str, ...]:
    """Liste ORDONNÉE des features produites par :func:`tile_features`.

    Sert de contrat : le cache et la sonde vérifient que les colonnes sont exactement
    celles-ci, dans cet ordre (un décalage silencieux serait un bug de design matrix).

    La signature accepte les mêmes paramètres que :func:`tile_features` pour pouvoir lui
    être passée telle quelle. ``gldm_alpha``/``gldm_delta`` ne changent PAS les noms (ils
    ne modifient que les valeurs) : c'est ``params`` dans le JSON de provenance, pas cette
    fonction, qui doit détecter un changement de ces réglages.
    """
    names: list[str] = []
    for d in distances:
        names += [f"glcm_d{d}_{s}" for s in GLCM_SUFFIXES]
    names += [f"glrlm_{s}" for s in GLRLM_SUFFIXES]
    names += [f"glszm_{s}" for s in GLSZM_SUFFIXES]
    names += [f"gldm_{s}" for s in GLDM_SUFFIXES]
    names += [f"ngtdm_{s}" for s in NGTDM_SUFFIXES]
    if with_dwt:
        names += list(dwt_suffixes(dwt_level))
    if with_fourier:
        names += list(fourier_suffixes(fourier_bins, fourier_bands))
    return tuple(names)


# ─────────────────────────────────────────────────────────────── orchestrateur


def tile_features(
    gray: np.ndarray,
    levels: int = 16,
    distances: tuple[int, ...] = (1, 2, 4),
    with_dwt: bool = True,
    with_fourier: bool = True,
    gldm_alpha: int = 1,
    gldm_delta: int = 0,
    wavelet: str = "db4",
    dwt_level: int = 3,
    fourier_bins: int = 8,
    fourier_bands: int = 4,
) -> dict[str, float]:
    """Toutes les familles activées pour une tuile, en un dict ``{nom: valeur}``.

    ``gray`` : image 2D uint8 (utiliser :func:`to_gray` pour du RGB). La quantification se
    fait en interne sur ``levels`` niveaux à plage fixe.
    """
    g = np.asarray(gray)
    if g.ndim != 2:
        raise ValueError(f"tile_features attend une image 2D, reçu {g.shape}")
    q = quantize(g, levels=levels)
    out: dict[str, float] = {}
    out.update(glcm_features(q, levels, distances))
    out.update(glrlm_features(q, levels))
    out.update(_glszm_features(_glszm_matrix(q, levels), levels, int(q.size), "glszm"))
    out.update(_gldm_features(_gldm_matrix(q, levels, gldm_alpha, gldm_delta), levels, "gldm"))
    out.update(ngtdm_features(q, levels))
    if with_dwt:
        out.update(dwt_features(g, wavelet=wavelet, level=dwt_level))
    if with_fourier:
        out.update(fourier_features(g, n_bins=fourier_bins, n_bands=fourier_bands))
    # Contrat d'ordre : la largeur du design matrix ne doit jamais dépendre de l'ordre
    # d'insertion interne des familles (un décalage silencieux = bug de colonnes).
    names = expected_feature_names(levels=levels, distances=distances, with_dwt=with_dwt,
                                   with_fourier=with_fourier, wavelet=wavelet,
                                   dwt_level=dwt_level, fourier_bins=fourier_bins,
                                   fourier_bands=fourier_bands)
    missing = [n for n in names if n not in out]
    if missing:
        raise RuntimeError(f"features manquantes : {missing[:5]} (+{len(missing) - 5})")
    extra = set(out) - set(names)
    if extra:
        raise RuntimeError(f"features non déclarées : {sorted(extra)[:5]}")
    return {n: float(out[n]) for n in names}
