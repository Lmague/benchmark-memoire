"""Heuristique couleur **apprise** sur les annotations existantes.

Principe : à 2,4 mm/px et début août, les espèces cibles se distinguent d'abord
par la couleur de leur inflorescence vue du dessus (jaune des solidages, blanc
des ombelles de carotte, mauve des eupatoires...). Plutôt que d'écrire des seuils
HSV à la main — impossibles à régler et faux dès que la lumière change — on
apprend la distribution des couleurs directement sur les points déjà annotés de
``Annotations.gpkg`` et on la compare à la distribution du fond.

Le modèle est un simple **rapport de vraisemblance** :

    score(couleur) = log p(couleur | espèce) - log p(couleur | fond)

estimé par histogrammes 3D en CIE Lab (robuste à l'intensité : L est séparé de la
chromaticité a*/b*), lissés par un noyau gaussien pour combler les cases vides.
Appliqué à un bloc d'image, cela devient une simple indexation vectorisée : le
balayage d'une orthomosaïque entière coûte quelques dizaines de secondes.

Rien n'est codé en dur : les seuils, les couleurs et les tailles viennent des
données. Le seul choix arbitraire est la résolution des histogrammes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Bornes des axes Lab (sRGB tient largement dedans) et résolution des histogrammes.
_L_RANGE = (0.0, 100.0)
_A_RANGE = (-60.0, 70.0)
_B_RANGE = (-60.0, 90.0)
_NL, _NA, _NB = 12, 20, 20

# Un pixel « blanc pur » est du remplissage de bord d'orthomosaïque, jamais une plante.
PAD_THRESHOLD = 248


# ---------------------------------------------------------------------------
# sRGB -> CIE Lab (D65), en numpy, sans dépendance supplémentaire.
# ---------------------------------------------------------------------------

_M_SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype="float32",
)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype="float32")


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(h, w, 3) uint8 sRGB -> (h, w, 3) float32 Lab."""
    a = np.asarray(rgb, dtype="float32") / 255.0
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_SRGB_TO_XYZ.T
    xyz /= _WHITE_D65
    eps = np.float32(216.0 / 24389.0)
    kappa = np.float32(24389.0 / 27.0)
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    out = np.empty_like(f)
    out[..., 0] = 116.0 * fy - 16.0
    out[..., 1] = 500.0 * (fx - fy)
    out[..., 2] = 200.0 * (fy - fz)
    return out


def _bin_indices(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lab -> indices de cases entiers, bornés."""
    def q(v, lo, hi, n):
        idx = ((v - lo) / (hi - lo) * n).astype("int32")
        return np.clip(idx, 0, n - 1)

    return (
        q(lab[..., 0], *_L_RANGE, _NL),
        q(lab[..., 1], *_A_RANGE, _NA),
        q(lab[..., 2], *_B_RANGE, _NB),
    )


def _histogram(lab_samples: np.ndarray) -> np.ndarray:
    """Histogramme 3D normalisé et lissé à partir d'un tableau (n, 3) Lab."""
    from scipy.ndimage import gaussian_filter

    il, ia, ib = _bin_indices(lab_samples)
    flat = (il * _NA + ia) * _NB + ib
    counts = np.bincount(flat.ravel(), minlength=_NL * _NA * _NB)
    hist = counts.reshape(_NL, _NA, _NB).astype("float32")
    hist = gaussian_filter(hist, sigma=(0.8, 1.2, 1.2), mode="nearest")
    hist += 1e-3                      # lissage de Laplace : aucune case à zéro
    hist /= hist.sum()
    return hist


# ---------------------------------------------------------------------------
# Échantillonnage sur le raster
# ---------------------------------------------------------------------------


def _patch_pixels(tiler, x: float, y: float, radius_px: int, factor: int) -> np.ndarray:
    """Pixels d'un disque centré sur un point annoté, lus de façon décimée."""
    size_full = (2 * radius_px + 1) * factor
    out = (2 * radius_px + 1)
    arr = tiler.read_centered(x, y, size_full, out_size=out)
    if arr.size == 0:
        return np.empty((0, 3), dtype="uint8")
    h, w = arr.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2 + 0.5
    px = arr[..., :3][disc]
    return px[~np.all(px >= PAD_THRESHOLD, axis=1)]


def sample_species_colors(
    tiler,
    xs: np.ndarray,
    ys: np.ndarray,
    factor: int,
    radius_px: int = 5,
    max_points: int = 800,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pixels échantillonnés autour de points annotés tombant dans ce raster."""
    rng = rng or np.random.default_rng(0)
    keep = np.array([tiler.contains_utm(x, y) for x, y in zip(xs, ys)], dtype=bool) \
        if xs.size else np.zeros(0, dtype=bool)
    xs, ys = xs[keep], ys[keep]
    if xs.size == 0:
        return np.empty((0, 3), dtype="uint8")
    if xs.size > max_points:
        sel = rng.choice(xs.size, size=max_points, replace=False)
        xs, ys = xs[sel], ys[sel]
    chunks = [_patch_pixels(tiler, x, y, radius_px, factor) for x, y in zip(xs, ys)]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.empty((0, 3), dtype="uint8")
    return np.concatenate(chunks, axis=0)


def sample_background_colors(
    tiler,
    factor: int,
    n_windows: int = 80,
    window_px: int = 96,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pixels de fond : fenêtres aléatoires lues à la même décimation.

    Les fenêtres majoritairement blanches (remplissage de bord) sont rejetées, de
    même que les pixels blancs isolés.
    """
    rng = rng or np.random.default_rng(1)
    out: List[np.ndarray] = []
    tries = 0
    size_full = window_px * factor
    while len(out) < n_windows and tries < n_windows * 6:
        tries += 1
        col = int(rng.integers(0, max(1, tiler.width - size_full)))
        row = int(rng.integers(0, max(1, tiler.height - size_full)))
        arr = tiler.read_window(col, row, size_full, size_full,
                                out_shape=(window_px, window_px))
        if arr.size == 0:
            continue
        px = arr[..., :3].reshape(-1, 3)
        valid = px[~np.all(px >= PAD_THRESHOLD, axis=1)]
        if valid.shape[0] < 0.35 * px.shape[0]:
            continue                          # fenêtre de bordure : ignorée
        out.append(valid)
    if not out:
        return np.empty((0, 3), dtype="uint8")
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------


@dataclass
class SpeciesColorStats:
    """Diagnostic honnête de ce que la couleur seule peut faire pour une espèce."""

    n_points: int
    n_pixels: int
    auc: float                      # séparabilité pixel espèce vs fond (0,5 = aucune)
    percentiles: Dict[str, float]   # centiles du score SUR LE FOND -> seuils naturels

    def threshold(self, pct: float = 99.5) -> float:
        """Seuil tel qu'une fraction (100-pct) % des pixels de fond le dépasse."""
        key = _pct_key(pct)
        if key in self.percentiles:
            return self.percentiles[key]
        # interpolation grossière entre les centiles disponibles
        known = sorted((float(k), v) for k, v in self.percentiles.items())
        for p, v in known:
            if p >= pct:
                return v
        return known[-1][1] if known else 0.0

    def as_dict(self) -> Dict:
        return {"n_points": self.n_points, "n_pixels": self.n_pixels,
                "auc": round(self.auc, 4),
                "percentiles": {k: round(v, 4) for k, v in self.percentiles.items()}}


def _pct_key(p: float) -> str:
    return f"{float(p):g}"


_BG_PERCENTILES = (90.0, 95.0, 99.0, 99.5, 99.9, 99.99)


class ColorModel:
    """Rapport de vraisemblance couleur par espèce, appris sur une orthomosaïque."""

    def __init__(
        self,
        raster_name: str,
        factor: int,
        llr: Dict[str, np.ndarray],
        stats: Dict[str, SpeciesColorStats],
        gsd_m: float = 0.01,
    ):
        self.raster_name = raster_name
        self.factor = int(factor)
        # Résolution au sol à laquelle le modèle a été appris : le balayage doit
        # utiliser la même, sinon les statistiques de couleur ne correspondent plus.
        self.gsd_m = float(gsd_m)
        self.llr = llr
        self.stats = stats
        self._luts: Dict[str, np.ndarray] = {}

    # ---- construction ---------------------------------------------------------

    @classmethod
    def learn(
        cls,
        tilers,
        points_by_species: Dict[str, Tuple[np.ndarray, np.ndarray]],
        factor_for,
        gsd_m: float = 0.01,
        radius_px: int = 5,
        max_points: int = 800,
        seed: int = 0,
        log=print,
    ) -> "ColorModel":
        """Apprend le modèle en mutualisant TOUTES les orthomosaïques fournies.

        Indispensable : chaque espèce n'est annotée que sur une partie des vols
        (Lotus et Leucanthemum uniquement sur « Maison », Daucus surtout sur
        « TrailErable »...). Un modèle appris sur une seule orthomosaïque serait
        aveugle à la moitié des espèces. Toutes les orthomosaïques datent du même
        vol (9 août 2023), la radiométrie est donc comparable.

        ``factor_for`` est une fonction ``tiler -> facteur de décimation``, pour
        que tous les échantillons soient pris à la même résolution au sol malgré
        des GSD légèrement différentes d'un vol à l'autre.
        """
        tilers = list(tilers)
        rng = np.random.default_rng(seed)
        factors = {t.path.name: int(factor_for(t)) for t in tilers}

        log("  fond : échantillonnage de fenêtres aléatoires…")
        bg_parts = [
            sample_background_colors(t, factors[t.path.name],
                                     n_windows=max(20, 80 // len(tilers)), rng=rng)
            for t in tilers
        ]
        bg_parts = [b for b in bg_parts if b.size]
        bg = np.concatenate(bg_parts, axis=0) if bg_parts else np.empty((0, 3), "uint8")
        if bg.shape[0] < 5000:
            raise RuntimeError("Pas assez de pixels de fond exploitables.")
        bg_lab = rgb_to_lab(bg)
        h_bg = _histogram(bg_lab)
        log(f"  fond : {bg.shape[0]} pixels sur {len(tilers)} orthomosaïque(s)")

        llr: Dict[str, np.ndarray] = {}
        stats: Dict[str, SpeciesColorStats] = {}
        for code, (xs, ys) in points_by_species.items():
            parts, n_in = [], 0
            per_raster = max(50, max_points // max(1, len(tilers)))
            for t in tilers:
                inside = int(sum(t.contains_utm(x, y) for x, y in zip(xs, ys))) \
                    if xs.size else 0
                if not inside:
                    continue
                n_in += inside
                parts.append(sample_species_colors(
                    t, xs, ys, factors[t.path.name], radius_px, per_raster, rng))
            parts = [p for p in parts if p.size]
            px = np.concatenate(parts, axis=0) if parts else np.empty((0, 3), "uint8")
            if px.shape[0] < 300:
                log(f"  {code:8s} : {n_in} point(s), {px.shape[0]} pixels "
                    f"-> trop peu, espèce ignorée")
                continue
            sp_lab = rgb_to_lab(px)
            h_sp = _histogram(sp_lab)
            ratio = np.log(h_sp) - np.log(h_bg)
            llr[code] = ratio.astype("float32")

            s_sp = cls._lookup(ratio, sp_lab)
            s_bg = cls._lookup(ratio, bg_lab)
            auc = _fast_auc(s_sp, s_bg)
            pcts = {_pct_key(p): float(np.percentile(s_bg, p)) for p in _BG_PERCENTILES}
            stats[code] = SpeciesColorStats(n_in, int(px.shape[0]), auc, pcts)
            log(f"  {code:8s} : {n_in} point(s), {px.shape[0]} pixels, "
                f"AUC couleur={auc:.3f}, seuil(99,5%)={pcts[_pct_key(99.5)]:.2f}")

        if not llr:
            raise RuntimeError("Aucune espèce n'a assez de points annotés.")
        return cls(", ".join(t.path.name for t in tilers),
                   int(np.median(list(factors.values()))), llr, stats, gsd_m)

    @staticmethod
    def _lookup(table: np.ndarray, lab: np.ndarray) -> np.ndarray:
        il, ia, ib = _bin_indices(lab)
        return table[il, ia, ib]

    # ---- application ----------------------------------------------------------

    def species(self) -> List[str]:
        return list(self.llr.keys())

    def _lut(self, code: str) -> np.ndarray:
        """Table RGB 6 bits -> score, construite une fois par espèce.

        Le passage par Lab coûte cher (puissance 2,4 puis racine cubique sur
        chaque canal) : sur une orthomosaïque décimée de 200 Mpx cela dominerait
        le balayage. Comme l'entrée est de l'uint8, on précalcule le score des
        262 144 couleurs 6 bits ; noter un bloc devient une simple indexation, et
        le balayage redevient limité par les entrées-sorties disque.
        L'erreur de quantification (2 niveaux sur 256) est très inférieure à la
        largeur des cases de l'histogramme.
        """
        lut = self._luts.get(code)
        if lut is None:
            grid = (np.arange(64, dtype="float32") * 4 + 2)
            r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
            rgb = np.stack([r, g, b], axis=-1).reshape(-1, 3)
            lab = rgb_to_lab(rgb.astype("uint8"))
            lut = self._lookup(self.llr[code], lab).astype("float32")
            self._luts[code] = lut
        return lut

    @staticmethod
    def _rgb6(rgb: np.ndarray) -> np.ndarray:
        a = rgb[..., :3].astype("int32") >> 2
        return (a[..., 0] << 12) | (a[..., 1] << 6) | a[..., 2]

    def score_block(self, rgb: np.ndarray, code: str) -> np.ndarray:
        """Carte de score (h, w) float32 pour une espèce sur un bloc RGB."""
        return self._lut(code)[self._rgb6(rgb)]

    def score_all(self, rgb: np.ndarray) -> Dict[str, np.ndarray]:
        """Toutes les espèces d'un coup (index 6 bits calculé une seule fois)."""
        idx = self._rgb6(rgb)
        return {code: self._lut(code)[idx] for code in self.llr}

    def threshold(self, code: str, pct: float = 99.5) -> float:
        st = self.stats.get(code)
        return st.threshold(pct) if st else 0.0

    # ---- persistance ----------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        codes = list(self.llr.keys())
        np.savez_compressed(
            path,
            codes=np.array(codes),
            factor=np.array([self.factor]),
            raster_name=np.array([self.raster_name]),
            stack=np.stack([self.llr[c] for c in codes]).astype("float32"),
            gsd_m=np.array([self.gsd_m], dtype="float64"),
            stats=np.array([json.dumps({c: s.as_dict() for c, s in self.stats.items()})]),
        )

    @classmethod
    def load(cls, path: Path) -> "ColorModel":
        d = np.load(Path(path), allow_pickle=False)
        codes = [str(c) for c in d["codes"]]
        stack = d["stack"]
        raw = json.loads(str(d["stats"][0]))
        stats = {
            c: SpeciesColorStats(v["n_points"], v["n_pixels"], v["auc"],
                                 {k: float(x) for k, x in v["percentiles"].items()})
            for c, v in raw.items()
        }
        gsd = float(d["gsd_m"][0]) if "gsd_m" in d.files else 0.01
        return cls(str(d["raster_name"][0]), int(d["factor"][0]),
                   {c: stack[i] for i, c in enumerate(codes)}, stats, gsd)

    def summary(self) -> Dict:
        return {
            "raster": self.raster_name,
            "factor": self.factor,
            "gsd_m": self.gsd_m,
            "species": {c: s.as_dict() for c, s in self.stats.items()},
        }


def _fast_auc(pos: np.ndarray, neg: np.ndarray, max_n: int = 60000) -> float:
    """AUC de Mann-Whitney sur des sous-échantillons (assez précis, peu coûteux)."""
    rng = np.random.default_rng(7)
    if pos.size > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if neg.size > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    if pos.size == 0 or neg.size == 0:
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, allv.size + 1)
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))
