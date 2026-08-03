"""Lecture SEULE du jeu d'annotations existant (calque + source de prototypes).

Ce module n'écrit jamais et ne recopie jamais les données sources.

Les 10 757 points de ``Annotations.gpkg`` sont lus **une seule fois au démarrage**
et gardés en mémoire sous forme de tableaux numpy (≈ 170 Ko au total). L'ancienne
version rouvrait le GeoPackage six fois — une par couche — à **chaque changement
de tuile** ; sur une machine à mémoire saturée cela ajoutait plusieurs centaines
de millisecondes par déplacement. Le filtrage bbox devient un masque numpy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geopackage import read_layer_names


class ExistingAnnotations:
    def __init__(self, path: Optional[Path], target_epsg: int):
        self.path = Path(path).resolve() if path else None
        self.target_epsg = int(target_epsg)
        self.layers: List[str] = []
        # couche -> (xs, ys) en float64, dans le CRS cible
        self._pts: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.path and self.path.is_file():
            self.layers = read_layer_names(self.path)
            self._load()

    # ---- chargement unique -----------------------------------------------------

    def _load(self) -> None:
        import geopandas as gpd

        for layer in list(self.layers):
            try:
                gdf = gpd.read_file(self.path, layer=layer)
            except Exception:
                self._pts[layer] = (np.empty(0), np.empty(0))
                continue
            if gdf.crs is not None and gdf.crs.to_epsg() != self.target_epsg:
                gdf = gdf.to_crs(epsg=self.target_epsg)
            geom = gdf.geometry
            geom = geom[~(geom.isna() | geom.is_empty)]
            if len(geom) == 0:
                self._pts[layer] = (np.empty(0), np.empty(0))
                continue
            # Points uniquement ; représentant pour tout autre type de géométrie.
            if (geom.geom_type == "Point").all():
                xs = geom.x.to_numpy(dtype="float64")
                ys = geom.y.to_numpy(dtype="float64")
            else:
                rep = geom.representative_point()
                xs = rep.x.to_numpy(dtype="float64")
                ys = rep.y.to_numpy(dtype="float64")
            self._pts[layer] = (xs, ys)

    @property
    def available(self) -> bool:
        return bool(self.path and self.path.is_file() and self.layers)

    def species(self) -> List[str]:
        return list(self.layers)

    def counts(self) -> Dict[str, int]:
        return {lay: int(len(self._pts.get(lay, (np.empty(0),))[0])) for lay in self.layers}

    # ---- requêtes --------------------------------------------------------------

    def arrays(self, layer: str) -> Tuple[np.ndarray, np.ndarray]:
        """Tableaux (xs, ys) bruts d'une couche — utilisés par la prospection."""
        return self._pts.get(layer, (np.empty(0), np.empty(0)))

    def points_in_bbox(
        self, left: float, bottom: float, right: float, top: float
    ) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        if not self.available:
            return out
        for layer in self.layers:
            xs, ys = self._pts.get(layer, (np.empty(0), np.empty(0)))
            if xs.size == 0:
                out[layer] = []
                continue
            m = (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)
            out[layer] = [{"x": float(x), "y": float(y)}
                          for x, y in zip(xs[m], ys[m])]
        return out

    def all_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Tous les points, toutes couches confondues (pour l'exclusion des doublons)."""
        xs = [a for a, _ in self._pts.values() if a.size]
        ys = [b for _, b in self._pts.values() if b.size]
        if not xs:
            return np.empty(0), np.empty(0)
        return np.concatenate(xs), np.concatenate(ys)
