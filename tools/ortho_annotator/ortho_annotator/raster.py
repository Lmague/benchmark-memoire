"""Lecture fenêtrée d'orthomosaïques et grille de tuiles en mètres.

Contrainte centrale : aucun raster n'est jamais chargé en entier. Toute lecture
pixel passe par ``rasterio.windows.Window`` + ``src.read(window=...)`` bornée à la
tuile courante. Un cache LRU borné garde les dernières tuiles encodées.

La conversion pixel -> UTM n'est jamais codée à la main : elle délègue à
``rasterio.transform.xy`` (et ``rowcol`` pour l'inverse), qui gère l'inversion de
l'axe Y correctement.

Notes de performance (mesurées sur les orthomosaïques réelles) :

- Les GeoTIFF de ``Dataset_Leo`` sont **en bandes** (``blockysize=32``, pas de
  tuilage interne) : lire une fenêtre de 2000 px de haut oblige GDAL à
  décompresser ~63 bandes pleine largeur (0,2 à 1,2 s selon le cache disque).
  D'où : handles rasterio **par thread** (pas de verrou global qui sérialiserait
  les requêtes) et prefetch **directionnel** limité, jamais 8 voisins.
- L'encodage PNG d'une tuile de 1932² coûte ~290 ms ; le même contenu en JPEG
  qualité 90 coûte ~15 ms pour 4x moins d'octets. Le JPEG est donc le défaut.
- Les fichiers possèdent des **overviews internes** : toute lecture décimée
  (miniature, prospection) passe par ``out_shape`` et ne touche jamais le plein
  résolution.
"""

from __future__ import annotations

import io
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio import transform as rio_transform
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

# Plafond du cache GDAL : la machine cible a peu de RAM libre, on refuse que
# GDAL prenne les 5 % de RAM qu'il s'octroie par défaut.
os.environ.setdefault("GDAL_CACHEMAX", "256")

TILE_FORMATS = ("jpeg", "png")


@dataclass(frozen=True)
class TileInfo:
    index: int
    row: int
    col: int
    col_off: int
    row_off: int
    width: int
    height: int
    # Bounds UTM de la fenêtre : (left, bottom, right, top).
    left: float
    bottom: float
    right: float
    top: float


class RasterTiler:
    """Grille de tuiles calculée depuis la résolution réelle du raster."""

    def __init__(
        self,
        path: Path,
        tile_size_m: float,
        overlap_m: float,
        cache_size: int,
        tile_format: str = "jpeg",
        tile_quality: int = 90,
    ):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Raster introuvable : {self.path}")
        if tile_format not in TILE_FORMATS:
            raise ValueError(f"tile_format doit être dans {TILE_FORMATS}")

        self.tile_format = tile_format
        self.tile_quality = int(tile_quality)

        # Handles rasterio : un par thread (un DatasetReader n'est pas thread-safe,
        # mais en ouvrir plusieurs coûte quelques Ko et supprime le verrou global).
        self._local = threading.local()
        self._handles: List[rasterio.DatasetReader] = []
        self._handles_lock = threading.Lock()
        self._closed = False

        src = self._src()
        self.crs = src.crs
        self.width = int(src.width)
        self.height = int(src.height)
        self.count = int(src.count)
        self.dtypes = tuple(src.dtypes)
        self.nodatavals = tuple(src.nodatavals)
        self.res_x = abs(src.res[0])
        self.res_y = abs(src.res[1])
        self.transform = src.transform
        self.bounds = tuple(src.bounds)
        self.overviews = list(src.overviews(1)) if self.count else []

        # Conversion mètres -> pixels via la résolution réelle.
        self.tile_px_x = max(1, int(round(tile_size_m / self.res_x)))
        self.tile_px_y = max(1, int(round(tile_size_m / self.res_y)))
        overlap_px_x = int(round(overlap_m / self.res_x))
        overlap_px_y = int(round(overlap_m / self.res_y))
        self.stride_x = max(1, self.tile_px_x - overlap_px_x)
        self.stride_y = max(1, self.tile_px_y - overlap_px_y)
        self.tile_size_m = float(tile_size_m)
        self.overlap_m = float(overlap_m)

        self.n_cols = max(1, math.ceil(self.width / self.stride_x))
        self.n_rows = max(1, math.ceil(self.height / self.stride_y))
        self.n_tiles = self.n_cols * self.n_rows

        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_size = max(1, int(cache_size))

        # Prefetch : un seul thread de fond, « la dernière demande gagne ».
        self._prefetch_want: List[int] = []
        self._prefetch_evt = threading.Event()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_loop, name="tile-prefetch", daemon=True
        )
        self._prefetch_thread.start()

        # Carte de couverture + miniature via lecture DÉCIMÉE (overviews) :
        # ne charge jamais le plein résolution.
        self._thumb_png = b""
        self._thumb_w = 0
        self._thumb_h = 0
        self._coverage: List[bool] = []
        self.first_nonempty = 0
        self._build_coverage()

    # ---- handles par thread ----------------------------------------------------

    def _src(self) -> rasterio.DatasetReader:
        """Handle rasterio propre au thread courant (ouvert à la demande)."""
        src = getattr(self._local, "src", None)
        if src is not None and not src.closed:
            return src
        src = rasterio.open(self.path)
        self._local.src = src
        with self._handles_lock:
            if self._closed:
                src.close()
                raise RuntimeError("RasterTiler fermé")
            self._handles.append(src)
        return src

    # ---- géométrie de la grille ------------------------------------------------

    def _window_for(self, index: int) -> Tuple[int, int, int, int, int, int]:
        if not (0 <= index < self.n_tiles):
            raise IndexError(f"index de tuile hors bornes : {index} / {self.n_tiles}")
        row = index // self.n_cols
        col = index % self.n_cols
        col_off = col * self.stride_x
        row_off = row * self.stride_y
        w = min(self.tile_px_x, self.width - col_off)
        h = min(self.tile_px_y, self.height - row_off)
        return row, col, col_off, row_off, int(w), int(h)

    def tile_info(self, index: int) -> TileInfo:
        row, col, col_off, row_off, w, h = self._window_for(index)
        win = Window(col_off, row_off, w, h)
        left, bottom, right, top = window_bounds(win, self.transform)
        return TileInfo(
            index=index,
            row=row,
            col=col,
            col_off=col_off,
            row_off=row_off,
            width=w,
            height=h,
            left=float(left),
            bottom=float(bottom),
            right=float(right),
            top=float(top),
        )

    # ---- lecture fenêtrée ------------------------------------------------------

    def read_window(
        self,
        col_off: int,
        row_off: int,
        width: int,
        height: int,
        out_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Lecture fenêtrée brute -> tableau (h, w, bandes), uint8.

        ``out_shape`` (h, w) demande une lecture **décimée** : GDAL sert alors
        depuis les overviews internes, sans jamais toucher le plein résolution.
        Les fenêtres débordant du raster sont rognées (jamais d'erreur).
        """
        col_off = max(0, int(col_off))
        row_off = max(0, int(row_off))
        width = int(min(width, self.width - col_off))
        height = int(min(height, self.height - row_off))
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, self.count), dtype="uint8")
        win = Window(col_off, row_off, width, height)
        kwargs = {}
        if out_shape is not None:
            oh = max(1, int(out_shape[0]))
            ow = max(1, int(out_shape[1]))
            kwargs["out_shape"] = (self.count, oh, ow)
        data = self._src().read(window=win, **kwargs)
        return np.transpose(data, (1, 2, 0))

    def read_centered(self, x: float, y: float, size_px: int,
                      out_size: Optional[int] = None) -> np.ndarray:
        """Fenêtre carrée de ``size_px`` centrée sur une coordonnée UTM.

        Renvoie un tableau vide si la fenêtre sort du raster (le caller décide).
        """
        row, col = rio_transform.rowcol(self.transform, x, y)
        half = size_px // 2
        col_off, row_off = int(col) - half, int(row) - half
        if col_off < 0 or row_off < 0:
            return np.zeros((0, 0, self.count), dtype="uint8")
        if col_off + size_px > self.width or row_off + size_px > self.height:
            return np.zeros((0, 0, self.count), dtype="uint8")
        out = (out_size, out_size) if out_size else None
        return self.read_window(col_off, row_off, size_px, size_px, out_shape=out)

    def read_tile_image(self, index: int) -> bytes:
        """Tuile encodée (JPEG par défaut). Lecture strictement fenêtrée + cache LRU."""
        with self._cache_lock:
            if index in self._cache:
                self._cache.move_to_end(index)
                return self._cache[index]

        _, _, col_off, row_off, w, h = self._window_for(index)
        arr = self.read_window(col_off, row_off, w, h)
        blob = self._encode(arr)
        with self._cache_lock:
            self._cache[index] = blob
            self._cache.move_to_end(index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return blob

    @property
    def media_type(self) -> str:
        return "image/jpeg" if self.tile_format == "jpeg" else "image/png"

    def _encode(self, arr: np.ndarray) -> bytes:
        """Encode un tableau (h, w, bandes) au format de tuile configuré."""
        img = self._to_image(arr)
        buf = io.BytesIO()
        if self.tile_format == "jpeg":
            img.save(buf, format="JPEG", quality=self.tile_quality,
                     subsampling=0, optimize=False)
        else:
            img.save(buf, format="PNG", compress_level=1)
        return buf.getvalue()

    @staticmethod
    def _to_image(arr: np.ndarray) -> Image.Image:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        bands = arr.shape[2]
        if bands == 1:
            return Image.fromarray(arr[:, :, 0]).convert("RGB")
        if bands == 2:
            return Image.fromarray(np.repeat(arr[:, :, :1], 3, axis=2), mode="RGB")
        return Image.fromarray(np.ascontiguousarray(arr[:, :, :3]), mode="RGB")

    def encode_array(self, arr: np.ndarray) -> bytes:
        """Encode un tableau arbitraire (utilisé pour les vignettes de candidats)."""
        return self._encode(arr)

    # ---- couverture / miniature (lecture décimée via overviews) ---------------

    def _build_coverage(self, max_dim: int = 1024) -> None:
        scale = min(1.0, max_dim / max(self.width, self.height))
        tw = max(1, int(self.width * scale))
        th = max(1, int(self.height * scale))
        arr = self.read_window(0, 0, self.width, self.height, out_shape=(th, tw))
        buf = io.BytesIO()
        self._to_image(arr).save(buf, format="PNG", compress_level=6)
        self._thumb_png = buf.getvalue()
        self._thumb_w, self._thumb_h = tw, th

        a = arr.astype("int16")
        white = np.all(a >= 248, axis=2)
        self._white_thumb = white
        fx, fy = tw / self.width, th / self.height
        cov: List[bool] = []
        for idx in range(self.n_tiles):
            row, col = idx // self.n_cols, idx % self.n_cols
            x0 = int(col * self.stride_x * fx)
            x1 = max(x0 + 1, int(min(self.width, col * self.stride_x + self.tile_px_x) * fx))
            y0 = int(row * self.stride_y * fy)
            y1 = max(y0 + 1, int(min(self.height, row * self.stride_y + self.tile_px_y) * fy))
            block = white[y0:y1, x0:x1]
            empty = block.size == 0 or float(block.mean()) > 0.98
            cov.append(not empty)
        self._coverage = cov
        self.first_nonempty = next((i for i, v in enumerate(cov) if v), 0)

    def thumbnail_png(self) -> bytes:
        return self._thumb_png

    def thumb_size(self) -> Tuple[int, int]:
        return self._thumb_w, self._thumb_h

    def is_empty(self, index: int) -> bool:
        if 0 <= index < len(self._coverage):
            return not self._coverage[index]
        return True

    def coverage_list(self) -> List[int]:
        return [1 if v else 0 for v in self._coverage]

    def region_empty(self, col_off: int, row_off: int, width: int, height: int,
                     ratio: float = 0.98) -> bool:
        """Région majoritairement blanche ? Testé sur la miniature, sans lecture.

        Sert à sauter les bordures de remplissage pendant la prospection : sur
        certaines orthomosaïques elles représentent la moitié de l'emprise.
        """
        white = getattr(self, "_white_thumb", None)
        if white is None or white.size == 0:
            return False
        fx = white.shape[1] / self.width
        fy = white.shape[0] / self.height
        x0 = max(0, int(col_off * fx))
        y0 = max(0, int(row_off * fy))
        x1 = max(x0 + 1, int((col_off + width) * fx))
        y1 = max(y0 + 1, int((row_off + height) * fy))
        block = white[y0:y1, x0:x1]
        return block.size == 0 or float(block.mean()) > ratio

    def next_nonempty(self, index: int, step: int) -> int:
        i = index + step
        while 0 <= i < self.n_tiles:
            if self._coverage[i]:
                return i
            i += step
        return index

    # ---- prefetch (thread unique, dernière demande gagnante) -------------------

    def prefetch(self, indices: List[int]) -> None:
        """Demande le préchargement de quelques tuiles ; ne bloque jamais l'appelant."""
        wanted = [i for i in indices if 0 <= i < self.n_tiles]
        self._prefetch_want = wanted
        self._prefetch_evt.set()

    def _prefetch_loop(self) -> None:
        while True:
            self._prefetch_evt.wait()
            self._prefetch_evt.clear()
            if self._closed:
                return
            wanted = list(self._prefetch_want)
            for idx in wanted:
                # Une nouvelle demande annule celle en cours : l'utilisateur a bougé.
                if self._prefetch_evt.is_set() or self._closed:
                    break
                with self._cache_lock:
                    already = idx in self._cache
                if already:
                    continue
                try:
                    self.read_tile_image(idx)
                except Exception:
                    pass

    def neighbor_indices(self, index: int, direction: int = 1) -> List[int]:
        """Voisins à précharger : la suite du parcours d'abord, puis la rangée.

        On ne précharge PAS les 8 voisins : sur un TIFF en bandes chaque tuile
        coûte ~0,2-1 s de décompression, et 8 préchargements bloquants rendaient
        la navigation rapide inutilisable.
        """
        row, col, *_ = self._window_for(index)
        out: List[int] = []

        def push(r: int, c: int) -> None:
            if 0 <= r < self.n_rows and 0 <= c < self.n_cols:
                out.append(r * self.n_cols + c)

        step = 1 if direction >= 0 else -1
        push(row, col + step)          # continuation du parcours horizontal
        push(row + step, col)          # rangée suivante dans le même sens
        push(row, col - step)          # retour arrière immédiat
        return out

    # ---- conversions de coordonnées (déléguées à rasterio) ---------------------

    def pixel_to_utm(self, index: int, px: float, py: float) -> Tuple[float, float]:
        """Pixel (dans la tuile) -> coordonnées UTM, via rasterio.transform.xy.

        ``px`` = colonne depuis le bord gauche de la tuile ;
        ``py`` = ligne depuis le bord haut de la tuile.
        """
        _, _, col_off, row_off, w, h = self._window_for(index)
        full_col = col_off + float(px)
        full_row = row_off + float(py)
        x, y = rio_transform.xy(self.transform, full_row, full_col, offset="center")
        return float(x), float(y)

    def index_for_utm(self, x: float, y: float) -> int:
        """Index de la tuile contenant une coordonnée UTM."""
        full_row, full_col = rio_transform.rowcol(self.transform, x, y)
        col = min(self.n_cols - 1, max(0, int(full_col) // self.stride_x))
        row = min(self.n_rows - 1, max(0, int(full_row) // self.stride_y))
        return row * self.n_cols + col

    def utm_to_pixel(self, index: int, x: float, y: float) -> Tuple[float, float]:
        """Coordonnées UTM -> pixel (dans la tuile), via rasterio.transform.rowcol."""
        _, _, col_off, row_off, w, h = self._window_for(index)
        full_row, full_col = rio_transform.rowcol(self.transform, x, y)
        return float(full_col - col_off), float(full_row - row_off)

    def contains_utm(self, x: float, y: float) -> bool:
        left, bottom, right, top = self.bounds
        return left <= x <= right and bottom <= y <= top

    def inside_mask(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Version vectorisée de ``contains_utm`` pour des tableaux de points."""
        left, bottom, right, top = self.bounds
        xs = np.asarray(xs, dtype="float64")
        ys = np.asarray(ys, dtype="float64")
        if xs.size == 0:
            return np.zeros(0, dtype=bool)
        return (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)

    def best_factor_for(self, target_res_m: float) -> int:
        """Facteur de décimation (puissance de 2 disponible) le plus proche d'une GSD cible."""
        want = max(1.0, target_res_m / self.res_x)
        candidates = [1] + [f for f in self.overviews if f > 0]
        return min(candidates, key=lambda f: abs(math.log(f / want)) if f > 0 else 1e9)

    # ---- cycle de vie ----------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self._prefetch_evt.set()
        with self._handles_lock:
            handles, self._handles = self._handles, []
        for src in handles:
            try:
                if not src.closed:
                    src.close()
            except Exception:
                pass

    def metadata(self) -> Dict:
        return {
            "path": str(self.path),
            "name": self.path.name,
            "crs": str(self.crs),
            "epsg": self.crs.to_epsg() if self.crs else None,
            "width": self.width,
            "height": self.height,
            "count": self.count,
            "dtypes": list(self.dtypes),
            "nodatavals": [None if v is None else float(v) for v in self.nodatavals],
            "res_x": self.res_x,
            "res_y": self.res_y,
            "tile_size_m": self.tile_size_m,
            "overlap_m": self.overlap_m,
            "tile_px_x": self.tile_px_x,
            "tile_px_y": self.tile_px_y,
            "stride_x": self.stride_x,
            "stride_y": self.stride_y,
            "n_cols": self.n_cols,
            "n_rows": self.n_rows,
            "n_tiles": self.n_tiles,
            "bounds": {
                "left": self.bounds[0],
                "bottom": self.bounds[1],
                "right": self.bounds[2],
                "top": self.bounds[3],
            },
            "cache_size": self._cache_size,
            "first_nonempty": self.first_nonempty,
            "thumb_w": self._thumb_w,
            "thumb_h": self._thumb_h,
            "tile_format": self.tile_format,
        }


def list_rasters(folder: Path, include_padded: bool = False) -> List[str]:
    """Liste les .tif d'un dossier (hors .aux.xml et, par défaut, hors _padded)."""
    folder = Path(folder)
    names: List[str] = []
    for p in sorted(folder.glob("*.tif")):
        stem = p.stem
        if not include_padded and stem.endswith("_padded"):
            continue
        names.append(p.name)
    return names
