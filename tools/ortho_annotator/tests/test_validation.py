"""Tests de validation autonomes (exécutables séparément).

Lancement :
    cd tools/ortho_annotator
    python -m unittest tests.test_validation -v

Un raster réel est requis pour les tests géospatiaux. Par défaut on utilise
l'orthomosaïque Clairière ; surchargeable via la variable d'environnement
ORTHO_TEST_RASTER. Les tests dépendant d'un fichier absent sont ignorés (skip),
jamais échoués silencieusement.

Aucun test n'écrit dans Dataset_Leo/ : les sorties vont dans des dossiers
temporaires supprimés en fin de test.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Rendre le package importable quel que soit le cwd.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from ortho_annotator.gpkg_blob import (
    decode_bbox,
    decode_point,
    encode_bbox_polygon,
    encode_point,
)
from ortho_annotator.geopackage import REVIEW_LAYER, AnnotationStore, read_layer_names
from ortho_annotator.overlay import ExistingAnnotations
from ortho_annotator.raster import RasterTiler, list_rasters
from ortho_annotator.references import ReferenceLibrary
from ortho_annotator.validation import (
    run_coordinate_validation,
    run_roundtrip_validation,
)

_REPO_ROOT = _PKG_ROOT.parents[1]
_DATASET = _REPO_ROOT / "Dataset_Leo" / "Orthomosaiques"
_INAT = _REPO_ROOT / "Dataset_Leo" / "INaturalist"
_EXISTING = _DATASET / "Annotations.gpkg"

_DEFAULT_RASTER = _DATASET / "Orthom_Clairiere_9Aout23_WGS84UTM18N.tif"
_RASTER = Path(os.environ.get("ORTHO_TEST_RASTER", str(_DEFAULT_RASTER)))


def _raster_available() -> bool:
    return _RASTER.is_file()


class TestGpkgBlob(unittest.TestCase):
    def test_roundtrip_point(self):
        x, y = 716806.7699, 5166420.0583
        blob = encode_point(x, y, 32618)
        self.assertEqual(blob[:2], b"GP")
        dx, dy = decode_point(blob)
        self.assertAlmostEqual(dx, x, places=6)
        self.assertAlmostEqual(dy, y, places=6)

    def test_roundtrip_polygon_bbox(self):
        blob = encode_bbox_polygon(715300.0, 5166800.0, 715310.0, 5166820.0, 32618)
        self.assertEqual(blob[:2], b"GP")
        minx, miny, maxx, maxy = decode_bbox(blob)
        self.assertAlmostEqual(minx, 715300.0, places=4)
        self.assertAlmostEqual(miny, 5166800.0, places=4)
        self.assertAlmostEqual(maxx, 715310.0, places=4)
        self.assertAlmostEqual(maxy, 5166820.0, places=4)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="ortho_store_"))
        self.out = self.tmp / "out.gpkg"
        self.species = ["Lotcorn", "Solcan"]
        self.store = AnnotationStore(self.out, self.species, 32618)

    def tearDown(self):
        try:
            self.store.close()
        except Exception:
            pass
        for p in self.tmp.rglob("*"):
            if p.is_file():
                p.unlink()
        self.tmp.rmdir()

    def test_layers_created(self):
        layers = set(read_layer_names(self.out))
        self.assertTrue(set(self.species).issubset(layers))
        self.assertIn(REVIEW_LAYER, layers)

    def test_add_and_query_and_counts(self):
        fid = self.store.add_point("Solcan", 715300.0, 5166800.0)
        self.assertIsInstance(fid, int)
        self.assertEqual(self.store.counts()["Solcan"], 1)
        pts = self.store.points_in_bbox(715299, 5166799, 715301, 5166801)
        self.assertEqual(len(pts["Solcan"]), 1)
        self.assertAlmostEqual(pts["Solcan"][0]["x"], 715300.0, places=4)

    def test_delete(self):
        fid = self.store.add_point("Lotcorn", 715920.0, 5167900.0)
        self.assertTrue(self.store.delete_point("Lotcorn", fid))
        self.assertEqual(self.store.counts()["Lotcorn"], 0)
        self.assertFalse(self.store.delete_point("Lotcorn", fid))

    def test_undo_add_then_delete(self):
        fid = self.store.add_point("Solcan", 715300.0, 5166800.0)
        self.assertEqual(self.store.counts()["Solcan"], 1)
        self.store.undo()  # annule l'ajout
        self.assertEqual(self.store.counts()["Solcan"], 0)

        fid = self.store.add_point("Solcan", 715300.0, 5166800.0)
        self.store.delete_point("Solcan", fid)
        self.assertEqual(self.store.counts()["Solcan"], 0)
        res = self.store.undo()  # annule la suppression -> réinsertion
        self.assertEqual(res["undone"], "delete")
        self.assertEqual(self.store.counts()["Solcan"], 1)

    def test_history_depth_at_least_20(self):
        for i in range(30):
            self.store.add_point("Solcan", 715300.0 + i, 5166800.0)
        # Au moins 20 niveaux d'annulation disponibles.
        self.assertGreaterEqual(self.store.history_length(), 20)

    def test_reload_persists(self):
        self.store.add_point("Solcan", 715300.0, 5166800.0)
        self.store.add_point("Lotcorn", 715920.0, 5167900.0)
        self.store.close()
        # Nouvelle session sur le MÊME fichier : les points réapparaissent.
        store2 = AnnotationStore(self.out, self.species, 32618)
        try:
            self.assertEqual(store2.counts(), {"Lotcorn": 1, "Solcan": 1})
        finally:
            store2.close()
        self.store = AnnotationStore(self.out, self.species, 32618)  # pour tearDown

    def test_review_layer_created(self):
        self.assertIn(REVIEW_LAYER, read_layer_names(self.out))

    def test_review_box_add_list_delete(self):
        fid = self.store.add_review_box(715300.0, 5166800.0, 715310.0, 5166820.0, "à vérifier")
        boxes = self.store.list_review_boxes()
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0]["note"], "à vérifier")
        # requête spatiale bbox
        hit = self.store.review_boxes_in_bbox(715299, 5166799, 715311, 5166821)
        self.assertEqual(len(hit), 1)
        self.assertTrue(self.store.delete_review_box(fid))
        self.assertEqual(len(self.store.list_review_boxes()), 0)

    def test_review_undo(self):
        self.store.add_review_box(715300.0, 5166800.0, 715310.0, 5166820.0)
        self.assertEqual(len(self.store.list_review_boxes()), 1)
        self.store.undo()
        self.assertEqual(len(self.store.list_review_boxes()), 0)


class TestExportFlat(unittest.TestCase):
    def test_export_flat_single_layer(self):
        import geopandas as gpd

        from ortho_annotator.cli import main

        tmp = Path(tempfile.mkdtemp(prefix="ortho_export_"))
        src = tmp / "multi.gpkg"
        dest = tmp / "flat.gpkg"
        species = ["Lotcorn", "Solcan"]
        store = AnnotationStore(src, species, 32618)
        store.add_point("Solcan", 715300.0, 5166800.0)
        store.add_point("Lotcorn", 715920.0, 5167900.0)
        store.add_point("Solcan", 715301.0, 5166801.0)
        store.close()

        rc = main(["export-flat", "--source", str(src), "--dest", str(dest),
                   "--layer", "annotations"])
        self.assertEqual(rc, 0)

        layers = read_layer_names(dest)
        self.assertEqual(layers, ["annotations"])
        gdf = gpd.read_file(dest, layer="annotations")
        self.assertEqual(len(gdf), 3)
        self.assertIn("Label", gdf.columns)
        self.assertEqual(set(gdf["Label"]), {"Lotcorn", "Solcan"})

        for p in tmp.rglob("*"):
            if p.is_file():
                p.unlink()
        tmp.rmdir()


class TestReferences(unittest.TestCase):
    def test_discovery_bundled(self):
        species = ["Lotcorn", "Leuvul", "Ascsyr", "Daucar", "Eumac", "Solcan"]
        # Dossiers par défaut = dossier empaqueté avec l'outil (+ INaturalist optionnel).
        lib = ReferenceLibrary(ReferenceLibrary.default_roots(None), species)
        d = lib.as_dict()
        # Toutes les espèces ont des vues aériennes (dataset) ET des photos au sol.
        for sp in species:
            self.assertTrue(d[sp]["available"], f"{sp} sans image")
            self.assertGreater(d[sp]["aerial_count"], 0, f"{sp} sans vue aérienne")
            self.assertGreater(d[sp]["ground_count"], 0, f"{sp} sans photo au sol")
        # resolve_image renvoie un fichier existant, dans un dossier racine.
        for kind in ("aerial", "ground"):
            p = lib.resolve_image("Leuvul", kind, 0)
            self.assertIsNotNone(p, f"resolve {kind}")
            self.assertTrue(p.is_file())
        # index hors bornes -> None
        self.assertIsNone(lib.resolve_image("Leuvul", "aerial", 9999))


@unittest.skipUnless(_raster_available(), "raster de test absent")
class TestRasterGrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiler = RasterTiler(_RASTER, tile_size_m=5.0, overlap_m=0.0, cache_size=4)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()

    def test_grid_dimensions(self):
        import math
        t = self.tiler
        self.assertEqual(t.n_cols, max(1, math.ceil(t.width / t.stride_x)))
        self.assertEqual(t.n_rows, max(1, math.ceil(t.height / t.stride_y)))
        self.assertEqual(t.n_tiles, t.n_cols * t.n_rows)

    def test_interior_tile_full_size(self):
        # Une tuile intérieure (index 0 dans un raster large) doit être pleine.
        info = self.tiler.tile_info(0)
        self.assertEqual(info.width, self.tiler.tile_px_x)
        self.assertEqual(info.height, self.tiler.tile_px_y)

    def test_edge_tile_clipped(self):
        last = self.tiler.n_tiles - 1
        info = self.tiler.tile_info(last)
        self.assertGreater(info.width, 0)
        self.assertGreater(info.height, 0)
        self.assertLessEqual(info.col_off + info.width, self.tiler.width)
        self.assertLessEqual(info.row_off + info.height, self.tiler.height)

    def test_windowed_read_returns_encoded_tile(self):
        # Format par défaut : JPEG (20x plus rapide à encoder que PNG sur des
        # tuiles de ~1900 px, pour un quart du poids).
        blob = self.tiler.read_tile_image(0)
        self.assertTrue(blob.startswith(b"\xff\xd8"))
        self.assertEqual(self.tiler.media_type, "image/jpeg")
        lossless = RasterTiler(_RASTER, tile_size_m=5.0, overlap_m=0.0,
                               cache_size=2, tile_format="png")
        try:
            self.assertTrue(lossless.read_tile_image(0).startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            lossless.close()

    def test_overlap_stride(self):
        t = RasterTiler(_RASTER, tile_size_m=5.0, overlap_m=1.0, cache_size=2)
        try:
            self.assertLess(t.stride_x, t.tile_px_x)
            self.assertLess(t.stride_y, t.tile_px_y)
            self.assertGreaterEqual(t.n_tiles, self.tiler.n_tiles)
        finally:
            t.close()

    def test_pixel_utm_roundtrip(self):
        info = self.tiler.tile_info(0)
        x, y = self.tiler.pixel_to_utm(0, 10.0, 20.0)
        px, py = self.tiler.utm_to_pixel(0, x, y)
        self.assertAlmostEqual(px, 10.0, places=3)
        self.assertAlmostEqual(py, 20.0, places=3)
        self.assertTrue(info.left <= x <= info.right)
        self.assertTrue(info.bottom <= y <= info.top)

    def test_coverage_detects_blank_corner(self):
        # La tuile 0 de Clairière est un coin blanc (padding) -> détectée vide.
        self.assertTrue(self.tiler.is_empty(0))
        # first_nonempty pointe vers une tuile avec contenu.
        self.assertFalse(self.tiler.is_empty(self.tiler.first_nonempty))
        self.assertEqual(len(self.tiler.thumbnail_png()[:8]), 8)
        self.assertEqual(self.tiler.thumbnail_png()[:8], b"\x89PNG\r\n\x1a\n")

    def test_next_nonempty_skips_blanks(self):
        nxt = self.tiler.next_nonempty(0, 1)
        self.assertFalse(self.tiler.is_empty(nxt))

    def test_index_for_utm(self):
        info = self.tiler.tile_info(self.tiler.first_nonempty)
        cx = (info.left + info.right) / 2.0
        cy = (info.bottom + info.top) / 2.0
        self.assertEqual(self.tiler.index_for_utm(cx, cy), self.tiler.first_nonempty)


@unittest.skipUnless(_DATASET.is_dir(), "dossier Orthomosaiques absent")
class TestRasterListing(unittest.TestCase):
    def test_list_excludes_padded(self):
        names = list_rasters(_DATASET, include_padded=False)
        self.assertTrue(all(not n.endswith("_padded.tif") for n in names))
        self.assertTrue(any(n.startswith("Orthom_") for n in names))
        with_padded = list_rasters(_DATASET, include_padded=True)
        self.assertGreaterEqual(len(with_padded), len(names))


@unittest.skipUnless(_raster_available(), "raster de test absent")
class TestCoordinateValidation(unittest.TestCase):
    def test_corners_and_center(self):
        passed, report = run_coordinate_validation(_RASTER)
        for line in report:
            print(line)
        self.assertTrue(passed, "validation des coordonnées échouée")


@unittest.skipUnless(_raster_available(), "raster de test absent")
class TestRoundtripValidation(unittest.TestCase):
    def test_place_and_reread(self):
        species = ["Solcan"]
        passed, report = run_roundtrip_validation(_RASTER, species)
        for line in report:
            print(line)
        self.assertTrue(passed, "validation aller-retour échouée")


@unittest.skipUnless(_EXISTING.is_file(), "Annotations.gpkg absent")
class TestOverlayReadOnly(unittest.TestCase):
    def test_read_within_bbox(self):
        overlay = ExistingAnnotations(_EXISTING, 32618)
        self.assertTrue(overlay.available)
        # bbox englobant une partie connue de Solcan.
        pts = overlay.points_in_bbox(715150.0, 5166300.0, 716910.0, 5167920.0)
        total = sum(len(v) for v in pts.values())
        self.assertGreater(total, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
