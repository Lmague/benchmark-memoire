"""Persistance des annotations dans UN SEUL GeoPackage de sortie multi-couches.

- Le fichier de sortie reproduit le schéma du jeu existant : une couche par
  espèce (même nom), colonnes ['Label', 'geometry'], géométrie Point, EPSG:32618.
- Les couches sont créées via Fiona (métadonnées GeoPackage + index rtree corrects),
  UNIQUEMENT si elles manquent. Aucune couche existante n'est jamais réécrite.
- Toute mutation (ajout / suppression) passe ensuite par sqlite3 en écriture
  incrémentale immédiate ; les déclencheurs rtree/feature-count posés par OGR se
  maintiennent seuls.
- Le jeu d'annotations existant est lu en LECTURE SEULE (GeoPandas + filtre bbox)
  et n'est jamais recopié dans la sortie.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import fiona
from fiona.crs import CRS

from .gpkg_blob import decode_bbox, decode_point, encode_bbox_polygon, encode_point

_SCHEMA = {"geometry": "Point", "properties": {"Label": "str:255"}}
_REVIEW_SCHEMA = {
    "geometry": "Polygon",
    "properties": {"note": "str:255", "created": "str:32"},
}
# Nom de la couche des zones "à revoir" (préfixe improbable pour un code d'espèce).
REVIEW_LAYER = "zones_a_revoir"


def read_layer_names(gpkg_path: Path) -> List[str]:
    """Renvoie la liste ordonnée des couches d'un GeoPackage."""
    return list(fiona.listlayers(str(gpkg_path)))


@dataclass
class _Action:
    kind: str  # "add" | "delete" | "review_add" | "review_delete"
    layer: str
    fid: int
    payload: dict


class AnnotationStore:
    """Gère le GeoPackage de sortie unique et l'historique d'annulation."""

    def __init__(
        self,
        output_path: Path,
        species: List[str],
        srs_id: int,
        undo_depth: int = 50,
    ):
        self.path = Path(output_path)
        self.species = list(species)
        self.srs_id = int(srs_id)
        self._lock = threading.RLock()
        self._history: Deque[_Action] = deque(maxlen=max(20, int(undo_depth)))

        self._ensure_layers()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute("PRAGMA foreign_keys=ON")
        self._register_gpkg_functions(self._con)

    # ---- fonctions GeoPackage requises par les déclencheurs rtree d'OGR --------

    @staticmethod
    def _register_gpkg_functions(con: sqlite3.Connection) -> None:
        """Enregistre les fonctions ST_* utilisées par les triggers rtree d'OGR.

        Une connexion sqlite3 nue ne fournit pas ces fonctions (contrairement à
        SpatiaLite/OGR). Les déclencheurs installés par OGR à la création des
        couches les appellent à chaque INSERT/DELETE pour tenir l'index spatial ;
        on les implémente ici en décodant nos blobs de points.
        """

        def _extent(blob, i):
            if blob is None:
                return None
            try:
                return decode_bbox(blob)[i]
            except Exception:
                return None

        con.create_function("ST_MinX", 1, lambda b: _extent(b, 0))
        con.create_function("ST_MinY", 1, lambda b: _extent(b, 1))
        con.create_function("ST_MaxX", 1, lambda b: _extent(b, 2))
        con.create_function("ST_MaxY", 1, lambda b: _extent(b, 3))

        def _is_empty(blob):
            if blob is None:
                return 1
            try:
                decode_bbox(blob)
                return 0
            except Exception:
                return 1

        con.create_function("ST_IsEmpty", 1, _is_empty)

    # ---- création idempotente des couches -------------------------------------

    def _ensure_layers(self) -> None:
        # Créer le dossier parent au besoin (GDAL n'ouvre pas un chemin inexistant).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_layer_names(self.path) if self.path.exists() else []
        for sp in self.species:
            if sp in existing:
                continue  # couche déjà présente : NE JAMAIS la réécrire
            with fiona.open(
                str(self.path),
                "w",
                driver="GPKG",
                crs=CRS.from_epsg(self.srs_id),
                schema=_SCHEMA,
                layer=sp,
            ):
                pass
        # Couche des zones "à revoir" (polygones), créée une seule fois.
        if REVIEW_LAYER not in read_layer_names(self.path):
            with fiona.open(
                str(self.path),
                "w",
                driver="GPKG",
                crs=CRS.from_epsg(self.srs_id),
                schema=_REVIEW_SCHEMA,
                layer=REVIEW_LAYER,
            ):
                pass

    # ---- mutations incrémentales ----------------------------------------------

    def add_point(self, species: str, x: float, y: float, label: Optional[str] = None,
                  record_history: bool = True) -> int:
        if species not in self.species:
            raise ValueError(f"espèce inconnue : {species!r}")
        label = species if label is None else label
        blob = encode_point(x, y, self.srs_id)
        with self._lock:
            cur = self._con.execute(
                f'INSERT INTO "{species}" ("geom", "Label") VALUES (?, ?)',
                (sqlite3.Binary(blob), label),
            )
            self._con.commit()
            fid = int(cur.lastrowid)
            if record_history:
                self._history.append(_Action(
                    "add", species, fid, {"x": x, "y": y, "label": label}))
        return fid

    def delete_point(self, species: str, fid: int, record_history: bool = True) -> bool:
        if species not in self.species:
            raise ValueError(f"espèce inconnue : {species!r}")
        with self._lock:
            row = self._con.execute(
                f'SELECT "geom", "Label" FROM "{species}" WHERE fid = ?', (fid,)
            ).fetchone()
            if row is None:
                return False
            x, y = decode_point(row[0])
            label = row[1]
            self._con.execute(f'DELETE FROM "{species}" WHERE fid = ?', (fid,))
            self._con.commit()
            if record_history:
                self._history.append(_Action(
                    "delete", species, fid, {"x": x, "y": y, "label": label}))
        return True

    # ---- zones "à revoir" (polygones, non étiquetées) --------------------------

    def add_review_box(self, minx: float, miny: float, maxx: float, maxy: float,
                       note: str = "", record_history: bool = True) -> int:
        import datetime
        blob = encode_bbox_polygon(minx, miny, maxx, maxy, self.srs_id)
        created = datetime.datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self._con.execute(
                f'INSERT INTO "{REVIEW_LAYER}" ("geom", "note", "created") VALUES (?, ?, ?)',
                (sqlite3.Binary(blob), note, created),
            )
            self._con.commit()
            fid = int(cur.lastrowid)
            if record_history:
                self._history.append(_Action("review_add", REVIEW_LAYER, fid, {
                    "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "note": note}))
        return fid

    def delete_review_box(self, fid: int, record_history: bool = True) -> bool:
        with self._lock:
            row = self._con.execute(
                f'SELECT "geom", "note" FROM "{REVIEW_LAYER}" WHERE fid = ?', (fid,)
            ).fetchone()
            if row is None:
                return False
            minx, miny, maxx, maxy = decode_bbox(row[0])
            note = row[1]
            self._con.execute(f'DELETE FROM "{REVIEW_LAYER}" WHERE fid = ?', (fid,))
            self._con.commit()
            if record_history:
                self._history.append(_Action("review_delete", REVIEW_LAYER, fid, {
                    "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy, "note": note}))
        return True

    def list_review_boxes(self) -> List[Dict]:
        with self._lock:
            rows = self._con.execute(
                f'SELECT fid, "geom", "note", "created" FROM "{REVIEW_LAYER}"'
            ).fetchall()
        out = []
        for fid, blob, note, created in rows:
            minx, miny, maxx, maxy = decode_bbox(blob)
            out.append({"fid": int(fid), "minx": minx, "miny": miny,
                        "maxx": maxx, "maxy": maxy, "note": note or "", "created": created})
        return out

    def review_boxes_in_bbox(self, left: float, bottom: float,
                            right: float, top: float) -> List[Dict]:
        with self._lock:
            rows = self._con.execute(
                f'SELECT s.fid, s."geom", s."note" FROM "{REVIEW_LAYER}" s '
                f'JOIN "rtree_{REVIEW_LAYER}_geom" r ON s.fid = r.id '
                f"WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ?",
                (right, left, top, bottom),
            ).fetchall()
        out = []
        for fid, blob, note in rows:
            minx, miny, maxx, maxy = decode_bbox(blob)
            out.append({"fid": int(fid), "minx": minx, "miny": miny,
                        "maxx": maxx, "maxy": maxy, "note": note or ""})
        return out

    def undo(self) -> Optional[Dict]:
        """Annule la dernière action. Renvoie un descriptif de l'effet, ou None."""
        with self._lock:
            if not self._history:
                return None
            a = self._history.pop()
            p = a.payload
            if a.kind == "add":
                self.delete_point(a.layer, a.fid, record_history=False)
                return {"undone": "add", "species": a.layer, "x": p["x"], "y": p["y"]}
            if a.kind == "delete":
                new_fid = self.add_point(a.layer, p["x"], p["y"], p["label"],
                                         record_history=False)
                for older in self._history:
                    if older.layer == a.layer and older.fid == a.fid:
                        older.fid = new_fid
                return {"undone": "delete", "species": a.layer,
                        "fid": new_fid, "x": p["x"], "y": p["y"]}
            if a.kind == "review_add":
                self.delete_review_box(a.fid, record_history=False)
                return {"undone": "review_add"}
            if a.kind == "review_delete":
                new_fid = self.add_review_box(p["minx"], p["miny"], p["maxx"], p["maxy"],
                                              p.get("note", ""), record_history=False)
                for older in self._history:
                    if older.layer == a.layer and older.fid == a.fid:
                        older.fid = new_fid
                return {"undone": "review_delete", "fid": new_fid}
            return None

    def history_length(self) -> int:
        with self._lock:
            return len(self._history)

    # ---- requêtes --------------------------------------------------------------

    def points_in_bbox(
        self, left: float, bottom: float, right: float, top: float
    ) -> Dict[str, List[Dict]]:
        out: Dict[str, List[Dict]] = {}
        with self._lock:
            for sp in self.species:
                rows = self._con.execute(
                    f'SELECT s.fid, s."geom" FROM "{sp}" s '
                    f'JOIN "rtree_{sp}_geom" r ON s.fid = r.id '
                    f"WHERE r.minx <= ? AND r.maxx >= ? AND r.miny <= ? AND r.maxy >= ?",
                    (right, left, top, bottom),
                ).fetchall()
                pts = []
                for fid, blob in rows:
                    x, y = decode_point(blob)
                    pts.append({"fid": int(fid), "x": x, "y": y})
                out[sp] = pts
        return out

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                sp: int(self._con.execute(f'SELECT COUNT(*) FROM "{sp}"').fetchone()[0])
                for sp in self.species
            }

    def iter_all_points(self) -> Dict[str, List[Tuple[float, float, str]]]:
        with self._lock:
            result: Dict[str, List[Tuple[float, float, str]]] = {}
            for sp in self.species:
                rows = self._con.execute(
                    f'SELECT "geom", "Label" FROM "{sp}"'
                ).fetchall()
                result[sp] = [(*decode_point(b), lab) for b, lab in rows]
        return result

    def close(self) -> None:
        with self._lock:
            self._con.commit()
            self._con.close()
