"""Tests de la chaîne de prospection, sur un raster synthétique.

Aucun accès aux données réelles : un GeoTIFF est fabriqué à la volée avec des
taches de couleur connue, ce qui permet de vérifier que le modèle appris les
retrouve et que rien ne dérive dans les conversions de coordonnées.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ortho_annotator.colormodel import ColorModel, rgb_to_lab
from ortho_annotator.prospect import (STATUS_NEW, Candidate, CandidateStore, _Grid,
                                      ScanParams, deduplicate, detect_blobs,
                                      scan_raster)
from ortho_annotator.raster import RasterTiler

RES = 0.01          # 1 cm/px : un raster synthétique reste petit
SIZE = 600          # 6 m de côté
ORIGIN = (500000.0, 5000000.0)


def _make_raster(path: Path, blob_positions, colour=(230, 40, 40), radius=6):
    """Fond vert bruité + taches d'une couleur bien distincte."""
    rng = np.random.default_rng(3)
    img = np.stack([
        rng.integers(40, 80, (SIZE, SIZE)),
        rng.integers(90, 150, (SIZE, SIZE)),
        rng.integers(30, 70, (SIZE, SIZE)),
    ]).astype("uint8")
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for (r, c) in blob_positions:
        m = (yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2
        for b in range(3):
            img[b][m] = colour[b]
    transform = from_origin(ORIGIN[0], ORIGIN[1], RES, RES)
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE,
                       count=3, dtype="uint8", crs="EPSG:32618",
                       transform=transform) as dst:
        dst.write(img)
    return [(ORIGIN[0] + (c + 0.5) * RES, ORIGIN[1] - (r + 0.5) * RES)
            for (r, c) in blob_positions]


class ColourSpaceTests(unittest.TestCase):
    def test_rgb_to_lab_reference_values(self):
        rgb = np.array([[[255, 255, 255], [0, 0, 0], [255, 0, 0]]], dtype="uint8")
        lab = rgb_to_lab(rgb)[0]
        self.assertAlmostEqual(float(lab[0, 0]), 100.0, delta=0.3)
        self.assertAlmostEqual(float(lab[0, 1]), 0.0, delta=0.3)
        self.assertAlmostEqual(float(lab[1, 0]), 0.0, delta=0.3)
        # rouge pur : L~53, a~80, b~67
        self.assertAlmostEqual(float(lab[2, 0]), 53.2, delta=1.0)
        self.assertGreater(float(lab[2, 1]), 60.0)


class GridTests(unittest.TestCase):
    def test_has_within(self):
        g = _Grid(1.0)
        g.add(10.0, 10.0)
        self.assertTrue(g.has_within(10.2, 10.2, 0.5))
        self.assertFalse(g.has_within(12.0, 10.0, 0.5))
        # rayon supérieur à la maille : le voisinage doit s'élargir
        self.assertTrue(g.has_within(13.0, 10.0, 3.5))

    def test_deduplicate_keeps_best(self):
        cands = [
            Candidate("A", 0.0, 0.0, 1.0, 0.01, score=1.0),
            Candidate("A", 0.1, 0.0, 5.0, 0.01, score=5.0),
            Candidate("A", 3.0, 0.0, 2.0, 0.01, score=2.0),
            Candidate("B", 0.05, 0.0, 9.0, 0.01, score=9.0),
        ]
        out = deduplicate(cands, min_sep_m=0.5)
        self.assertEqual(len(out), 3)                     # A x2 fusionnés, B gardé
        kept_a = [c for c in out if c.species == "A"]
        self.assertEqual(len(kept_a), 2)
        self.assertIn(5.0, [c.score for c in kept_a])     # le meilleur survit


class ColorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="prospect_test_"))
        cls.blobs = [(r, c) for r in range(60, 540, 60) for c in range(60, 540, 60)]
        cls.points = _make_raster(cls.tmp / "synth.tif", cls.blobs)
        cls.tiler = RasterTiler(cls.tmp / "synth.tif", 2.0, 0.0, 4)

    @classmethod
    def tearDownClass(cls):
        cls.tiler.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _model(self):
        xs = np.array([p[0] for p in self.points])
        ys = np.array([p[1] for p in self.points])
        return ColorModel.learn(
            [self.tiler], {"Test": (xs, ys)},
            factor_for=lambda t: 1, gsd_m=RES, radius_px=3,
            max_points=40, log=lambda *a: None)

    def test_learn_separates_blobs_from_background(self):
        model = self._model()
        self.assertIn("Test", model.species())
        # la couleur des taches est très séparable du fond
        self.assertGreater(model.stats["Test"].auc, 0.95)

    def test_detect_blobs_recovers_positions(self):
        model = self._model()
        block = self.tiler.read_window(0, 0, SIZE, SIZE)
        found = detect_blobs(block, model, RES, ["Test"], {"Test": 90.0},
                             min_diam_m=0.04, max_diam_m=0.40)["Test"]
        self.assertGreaterEqual(len(found), int(0.8 * len(self.blobs)))
        got = np.array([[r, c] for r, c, _, _ in found])
        truth = np.array(self.blobs, dtype="float64")
        d = np.sqrt(((truth[:, None, :] - got[None, :, :]) ** 2).sum(-1))
        # chaque tache réelle a une détection à moins de 3 px
        self.assertGreater(float((d.min(axis=1) < 3.0).mean()), 0.8)

    def test_save_load_roundtrip(self):
        model = self._model()
        p = self.tmp / "cm.npz"
        model.save(p)
        again = ColorModel.load(p)
        self.assertEqual(again.species(), model.species())
        self.assertAlmostEqual(again.gsd_m, model.gsd_m)
        block = self.tiler.read_window(0, 0, 64, 64)
        np.testing.assert_allclose(again.score_block(block, "Test"),
                                   model.score_block(block, "Test"), atol=1e-5)

    def test_scan_excludes_known_points(self):
        """Un balayage complet ne doit proposer que ce qui n'est pas déjà annoté."""
        model = self._model()
        params = ScanParams(min_sep_m=0.2, exclude_m=0.15, block_px=256, rerank_top=0)
        xs = np.array([p[0] for p in self.points[:20]])
        ys = np.array([p[1] for p in self.points[:20]])
        found = scan_raster(self.tiler, model, (xs, ys), params,
                            thresholds={"Test": 90.0}, log=lambda *a: None)
        self.assertTrue(found)
        known = np.stack([xs, ys], axis=1)
        for c in found:
            d = np.sqrt(((known - np.array([c.x, c.y])) ** 2).sum(axis=1)).min()
            self.assertGreaterEqual(d, params.exclude_m)


class CandidateStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="candstore_"))
        self.store = CandidateStore(self.tmp / "c.sqlite")

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cands(self):
        return [Candidate("A", 10.0, 20.0, 1.0, 0.01, score=1.0),
                Candidate("A", 40.0, 20.0, 2.0, 0.01, score=2.0),
                Candidate("B", 10.0, 50.0, 3.0, 0.01, score=3.0)]

    def test_insert_query_and_bbox(self):
        n = self.store.replace_raster("r.tif", self._cands())
        self.assertEqual(n, 3)
        got = self.store.in_bbox("r.tif", 0, 0, 20, 30)
        self.assertEqual([c.species for c in got], ["A"])
        ranked = self.store.ranked("r.tif")
        self.assertEqual([c.score for c in ranked], [3.0, 2.0, 1.0])
        self.assertEqual(self.store.ranked("r.tif", species="B")[0].species, "B")

    def test_decisions_survive_a_new_scan(self):
        self.store.replace_raster("r.tif", self._cands())
        cid = self.store.ranked("r.tif", species="A")[0].cid
        self.store.set_status(cid, "rejected")
        # nouveau balayage : le candidat rejeté ne doit pas revenir
        self.store.replace_raster("r.tif", self._cands())
        ranked = self.store.ranked("r.tif")
        self.assertEqual(len(ranked), 2)
        self.assertTrue(all(c.status == STATUS_NEW for c in ranked))

    def test_add_is_idempotent_on_the_same_spot(self):
        self.store.add("r.tif", self._cands())
        self.store.add("r.tif", self._cands())
        self.assertEqual(len(self.store.ranked("r.tif")), 3)

    def test_counts_and_new_xy(self):
        self.store.replace_raster("r.tif", self._cands())
        counts = self.store.counts("r.tif")
        self.assertEqual(counts["A"][STATUS_NEW], 2)
        xs, ys, codes = self.store.all_new_xy("r.tif")
        self.assertEqual(len(xs), 3)
        self.assertEqual(sorted(set(codes)), ["A", "B"])


class RasterTilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tiler_test_"))
        _make_raster(cls.tmp / "synth.tif", [(100, 100)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_jpeg_and_png_encoding(self):
        for fmt, magic in (("jpeg", b"\xff\xd8"), ("png", b"\x89PNG")):
            t = RasterTiler(self.tmp / "synth.tif", 2.0, 0.0, 2, tile_format=fmt)
            try:
                blob = t.read_tile_image(0)
                self.assertTrue(blob.startswith(magic), fmt)
                self.assertIn(fmt, t.media_type)
            finally:
                t.close()

    def test_decimated_read_shape_and_inside_mask(self):
        t = RasterTiler(self.tmp / "synth.tif", 2.0, 0.0, 2)
        try:
            arr = t.read_window(0, 0, SIZE, SIZE, out_shape=(50, 50))
            self.assertEqual(arr.shape[:2], (50, 50))
            # une fenêtre débordant du raster est rognée, jamais une erreur
            edge = t.read_window(SIZE - 10, SIZE - 10, 100, 100)
            self.assertEqual(edge.shape[:2], (10, 10))
            xs = np.array([ORIGIN[0] + 1.0, ORIGIN[0] - 50.0])
            ys = np.array([ORIGIN[1] - 1.0, ORIGIN[1] - 1.0])
            np.testing.assert_array_equal(t.inside_mask(xs, ys), [True, False])
        finally:
            t.close()

    def test_read_centered_matches_pixel_conversion(self):
        t = RasterTiler(self.tmp / "synth.tif", 2.0, 0.0, 2)
        try:
            x, y = t.pixel_to_utm(0, 25.0, 25.0)
            crop = t.read_centered(x, y, 21)
            self.assertEqual(crop.shape[:2], (21, 21))
            direct = t.read_window(15, 15, 21, 21)
            np.testing.assert_array_equal(crop, direct)
        finally:
            t.close()


if __name__ == "__main__":
    unittest.main()
