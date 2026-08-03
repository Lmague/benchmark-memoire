"""Détection dense zero-shot par appariement de jetons de patch (DINOv3).

Motivation mesurée : l'heuristique couleur trouve très bien ce qui est en fleur
le 9 août (eupatoire, carotte sauvage), mais elle est aveugle à l'asclépiade,
défleurie à cette date — ses larges feuilles vertes ont la couleur du fond. Il
faut alors un descripteur de **texture et de forme**, pas de couleur.

Un seul passage avant sur une fenêtre de 784x784 donne 49x49 jetons de patch en
0,8 s sur ce CPU, soit une carte de descripteurs à ~10 cm de résolution au sol
pour une fenêtre de 5 m. On compare chaque jeton à une banque de jetons de
référence prélevés **aux points déjà annotés** :

    score(jeton) = max cos(jeton, jetons de l'espèce) - max cos(jeton, jetons de fond)

Aucun apprentissage : uniquement des produits scalaires dans un espace figé.

Astuce qui rend la construction bon marché : les points annotés sont groupés, si
bien qu'une seule fenêtre de 5 m en contient souvent des dizaines. Un passage
avant fournit donc des dizaines de jetons de référence d'un coup — et tous les
jetons éloignés de toute annotation servent de jetons de fond.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rasterio import transform as rio_transform

from .colormodel import PAD_THRESHOLD

# Distance au-delà de laquelle un jeton est considéré comme du fond sûr.
_BACKGROUND_MARGIN_M = 0.8


@dataclass
class DenseParams:
    span_m: float = 5.0        # côté de la fenêtre analysée, en mètres
    side_px: int = 784         # côté de l'entrée du réseau (multiple de 16)
    windows_per_species: int = 8
    max_tokens_per_species: int = 1200
    max_background: int = 6000

    @property
    def grid(self) -> int:
        return int(self.side_px) // 16

    def token_m(self) -> float:
        return self.span_m / self.grid

    def as_dict(self) -> Dict:
        return {"span_m": self.span_m, "side_px": self.side_px,
                "windows_per_species": self.windows_per_species,
                "max_tokens_per_species": self.max_tokens_per_species,
                "max_background": self.max_background}


def _window_tokens(tiler, embedder, x_center: float, y_center: float,
                   params: DenseParams) -> Optional[Tuple[np.ndarray, float, float]]:
    """Jetons d'une fenêtre carrée centrée sur une coordonnée UTM.

    Renvoie ``(tokens (g, g, d), left, top)`` ou None si la fenêtre déborde.
    Traite une seule fenêtre : pour un balayage, préférer ``_read_windows`` +
    ``DenseMatcher.score_batch`` (batch=1 laisse la VRAM GPU inutilisée).
    """
    reads = _read_windows(tiler, [(x_center, y_center)], params)
    arr, left, top = reads[0]
    if arr is None:
        return None
    tokens = embedder.dense(arr, params.side_px)
    return tokens, left, top


def _read_windows(tiler, centers: Sequence[Tuple[float, float]], params: DenseParams
                  ) -> List[Tuple[Optional[np.ndarray], Optional[float], Optional[float]]]:
    """Lit (sans encoder) une liste de fenêtres carrées centrées sur des coordonnées UTM.

    Renvoie une entrée par centre : ``(vignette, left, top)`` ou ``(None, None, None)``
    si la fenêtre déborde du raster ou est vide/bordure de remplissage.
    """
    size_px = max(16, int(round(params.span_m / tiler.res_x)))
    out: List[Tuple[Optional[np.ndarray], Optional[float], Optional[float]]] = []
    for x_center, y_center in centers:
        row, col = rio_transform.rowcol(tiler.transform, x_center, y_center)
        c0, r0 = int(col) - size_px // 2, int(row) - size_px // 2
        if c0 < 0 or r0 < 0 or c0 + size_px > tiler.width or r0 + size_px > tiler.height:
            out.append((None, None, None))
            continue
        arr = tiler.read_window(c0, r0, size_px, size_px,
                                out_shape=(params.side_px, params.side_px))
        if arr.size == 0 or float(np.all(arr[..., :3] >= PAD_THRESHOLD, axis=2).mean()) > 0.6:
            out.append((None, None, None))
            continue
        left, top = rio_transform.xy(tiler.transform, r0, c0, offset="ul")
        out.append((arr, float(left), float(top)))
    return out


def _cell_distance(cell_x: np.ndarray, cell_y: np.ndarray,
                   px: np.ndarray, py: np.ndarray, span_m: float) -> np.ndarray:
    """Distance de chaque cellule au point le plus proche (grand si aucun point)."""
    if px.size == 0:
        return np.full(cell_x.shape, 1e9, dtype="float32")
    left, top = float(cell_x.min()), float(cell_y.max())
    sel = ((px >= left - 1.0) & (px <= left + span_m + 1.0)
           & (py <= top + 1.0) & (py >= top - span_m - 1.0))
    if not sel.any():
        return np.full(cell_x.shape, 1e9, dtype="float32")
    d = np.sqrt((cell_x[..., None] - px[sel]) ** 2
                + (cell_y[..., None] - py[sel]) ** 2)
    return d.min(axis=-1).astype("float32")


def _dense_cells(xs: np.ndarray, ys: np.ndarray, span_m: float, k: int
                 ) -> List[Tuple[float, float]]:
    """Centres des ``k`` cellules les plus densément annotées."""
    if xs.size == 0:
        return []
    from collections import Counter

    keys = [(int(math.floor(x / span_m)), int(math.floor(y / span_m)))
            for x, y in zip(xs, ys)]
    return [((kx + 0.5) * span_m, (ky + 0.5) * span_m)
            for (kx, ky), _ in Counter(keys).most_common(k)]


class DenseMatcher:
    """Banque de jetons de référence + notation d'une fenêtre."""

    def __init__(self, model_id: str, params: DenseParams,
                 species: Dict[str, np.ndarray], background: np.ndarray):
        self.model_id = model_id
        self.params = params
        self.species = species
        self.background = background

    # ---- construction ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        tilers,
        embedder,
        points_by_species: Dict[str, Tuple[np.ndarray, np.ndarray]],
        params: Optional[DenseParams] = None,
        seed: int = 0,
        log=print,
    ) -> "DenseMatcher":
        params = params or DenseParams()
        rng = np.random.default_rng(seed)
        g = params.grid
        tok_m = params.token_m()

        by_code: Dict[str, List[np.ndarray]] = {}
        bg: List[np.ndarray] = []

        # Points de TOUTES les espèces, restreints à chaque orthomosaïque : ils
        # servent à décider quelles cellules sont du fond sûr (une cellule loin
        # de toute annotation, quelle que soit l'espèce).
        all_xy = {}
        for tiler in tilers:
            axs, ays = [], []
            for xs, ys in points_by_species.values():
                m = tiler.inside_mask(xs, ys)
                if m.any():
                    axs.append(xs[m])
                    ays.append(ys[m])
            all_xy[tiler.path.name] = (
                np.concatenate(axs) if axs else np.empty(0),
                np.concatenate(ays) if ays else np.empty(0),
            )

        gy, gx = np.mgrid[0:g, 0:g]
        for code, (xs, ys) in points_by_species.items():
            if xs.size == 0:
                continue
            got = 0
            for tiler in tilers:
                inside = tiler.inside_mask(xs, ys)
                if not inside.any() or got >= params.max_tokens_per_species:
                    continue
                cx, cy = xs[inside], ys[inside]
                axs, ays = all_xy[tiler.path.name]
                for (wx, wy) in _dense_cells(cx, cy, params.span_m,
                                             params.windows_per_species):
                    res = _window_tokens(tiler, embedder, wx, wy, params)
                    if res is None:
                        continue
                    tokens, left, top = res
                    cell_x = left + (gx + 0.5) * tok_m
                    cell_y = top - (gy + 0.5) * tok_m

                    d_own = _cell_distance(cell_x, cell_y, cx, cy, params.span_m)
                    hit = d_own <= tok_m * 0.75
                    if hit.any():
                        by_code.setdefault(code, []).append(tokens[hit])
                        got += int(hit.sum())

                    d_any = _cell_distance(cell_x, cell_y, axs, ays, params.span_m)
                    far = d_any > _BACKGROUND_MARGIN_M
                    if far.any():
                        cand = tokens[far]
                        if cand.shape[0] > 250:
                            cand = cand[rng.choice(cand.shape[0], 250, replace=False)]
                        bg.append(cand)
                    if got >= params.max_tokens_per_species:
                        break
            log(f"  {code:8s} : {got} jeton(s) de référence")

        species: Dict[str, np.ndarray] = {}
        for code, parts in by_code.items():
            mat = np.concatenate(parts, axis=0)
            if mat.shape[0] > params.max_tokens_per_species:
                sel = rng.choice(mat.shape[0], params.max_tokens_per_species,
                                 replace=False)
                mat = mat[sel]
            species[code] = mat.astype("float32")
        background = np.concatenate(bg, axis=0).astype("float32") if bg \
            else np.zeros((0, 384), dtype="float32")
        if background.shape[0] > params.max_background:
            sel = rng.choice(background.shape[0], params.max_background, replace=False)
            background = background[sel]
        log(f"  fond     : {background.shape[0]} jeton(s)")
        if not species:
            raise RuntimeError("Aucun jeton de référence : vérifier les annotations.")
        return cls(embedder.model_id, params, species, background)

    # ---- notation --------------------------------------------------------------

    def codes(self) -> List[str]:
        return list(self.species.keys())

    def score_tokens(self, tokens: np.ndarray,
                     codes: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """(g, g, d) -> carte de score (g, g) par espèce."""
        g = tokens.shape[0]
        flat = tokens.reshape(-1, tokens.shape[-1])
        bg_max = (self.background @ flat.T).max(axis=0) if self.background.size \
            else np.zeros(flat.shape[0], dtype="float32")
        out: Dict[str, np.ndarray] = {}
        for code in (codes or self.species.keys()):
            mat = self.species.get(code)
            if mat is None or mat.size == 0:
                continue
            sim = (mat @ flat.T).max(axis=0)
            out[code] = (sim - bg_max).reshape(g, g).astype("float32")
        return out

    def score_window(self, tiler, embedder, x_center: float, y_center: float,
                     codes: Optional[Sequence[str]] = None
                     ) -> Optional[Tuple[Dict[str, np.ndarray], float, float]]:
        res = _window_tokens(tiler, embedder, x_center, y_center, self.params)
        if res is None:
            return None
        tokens, left, top = res
        return self.score_tokens(tokens, codes), left, top

    def score_batch(self, tiler, embedder, centers: Sequence[Tuple[float, float]],
                    codes: Optional[Sequence[str]] = None, embed_batch_size: int = 16,
                    ) -> List[Optional[Tuple[Dict[str, np.ndarray], float, float]]]:
        """``score_window`` pour plusieurs centres, avec UN SEUL passage GPU par lot.

        C'est la version qui utilise la VRAM disponible : lit toutes les fenêtres
        du lot, les encode ensemble via ``Embedder.dense_batch`` (au lieu d'un
        forward pass par fenêtre), puis note chacune séparément (produits
        scalaires, négligeables face au forward pass). Renvoie une entrée par
        centre, dans le même ordre, ``None`` pour les fenêtres hors raster/vides.
        """
        reads = _read_windows(tiler, centers, self.params)
        valid = [i for i, (arr, _, _) in enumerate(reads) if arr is not None]
        out: List[Optional[Tuple[Dict[str, np.ndarray], float, float]]] = [None] * len(centers)
        if not valid:
            return out
        arrays = [reads[i][0] for i in valid]
        tokens_batch = embedder.dense_batch(arrays, self.params.side_px,
                                            batch_size=embed_batch_size)
        for k, i in enumerate(valid):
            _, left, top = reads[i]
            out[i] = (self.score_tokens(tokens_batch[k], codes), left, top)
        return out

    def peaks(self, score: np.ndarray, left: float, top: float,
              threshold: float, max_peaks: int = 40
              ) -> List[Tuple[float, float, float]]:
        """Maxima locaux d'une carte de score -> ``[(x, y, score), ...]`` en UTM."""
        from scipy import ndimage

        mx = ndimage.maximum_filter(score, size=3, mode="nearest")
        mask = (score >= mx) & (score > threshold)
        if not mask.any():
            return []
        rows, cols = np.nonzero(mask)
        vals = score[rows, cols]
        order = np.argsort(vals)[::-1][:max_peaks]
        tok = self.params.token_m()
        return [(float(left + (cols[i] + 0.5) * tok),
                 float(top - (rows[i] + 0.5) * tok),
                 float(vals[i])) for i in order]

    # ---- persistance ----------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=np.array([json.dumps({"model_id": self.model_id,
                                       "params": self.params.as_dict(),
                                       "codes": list(self.species.keys())})]),
            background=self.background,
            **{f"sp__{c}": v for c, v in self.species.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "DenseMatcher":
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"][0]))
        params = DenseParams(**meta["params"])
        species = {c: d[f"sp__{c}"] for c in meta["codes"]}
        return cls(meta["model_id"], params, species, d["background"])

    def summary(self) -> Dict:
        return {"model_id": self.model_id, "params": self.params.as_dict(),
                "n_background": int(self.background.shape[0]),
                "species": {c: int(v.shape[0]) for c, v in self.species.items()}}
