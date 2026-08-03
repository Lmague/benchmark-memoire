"""Tests de validation autonomes, réutilisés par la CLI et la suite de tests.

Deux validations obligatoires :

1. Conversion pixel -> UTM : convertit les pixels des coins et du centre du raster
   via ``rasterio.transform.xy`` et vérifie la correspondance avec ``src.bounds``
   à la tolérance d'un demi-pixel. Aucune lecture de pixels n'est nécessaire (on
   n'utilise que le transform et les bornes) : le raster complet n'est jamais lu.

2. Aller-retour API : pose un point au centre d'une tuile via le store, le relit
   depuis le GeoPackage de sortie et vérifie que ses coordonnées UTM tombent dans
   les bornes de la tuile.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import rasterio
from rasterio import transform as rio_transform

from .geopackage import AnnotationStore
from .raster import RasterTiler


def run_coordinate_validation(
    raster_path: Path, tile_size_m: float = 5.0, overlap_m: float = 0.0
) -> Tuple[bool, List[str]]:
    report: List[str] = []
    passed = True
    with rasterio.open(raster_path) as src:
        transform = src.transform
        width, height = src.width, src.height
        left, bottom, right, top = src.bounds
        res_x = abs(src.res[0])
        res_y = abs(src.res[1])

    half_x = 0.5 * res_x
    half_y = 0.5 * res_y
    eps = 1e-6

    checks = [
        # (nom, row, col, x_attendu, y_attendu, tol_x, tol_y)
        ("coin haut-gauche", 0, 0, left, top, half_x + eps, half_y + eps),
        ("coin bas-droit", height - 1, width - 1, right, bottom, half_x + eps, half_y + eps),
        # centre : parité de W/H -> tolérance 1 pixel
        ("centre", height // 2, width // 2,
         (left + right) / 2.0, (top + bottom) / 2.0, res_x + eps, res_y + eps),
    ]

    for name, row, col, xe, ye, tol_x, tol_y in checks:
        x, y = rio_transform.xy(transform, row, col, offset="center")
        dx, dy = abs(x - xe), abs(y - ye)
        ok = dx <= tol_x and dy <= tol_y
        passed = passed and ok
        report.append(
            f"[{'OK' if ok else 'ÉCHEC'}] {name}: pixel(row={row},col={col}) -> "
            f"({x:.6f}, {y:.6f}) ; écart=({dx:.6f}, {dy:.6f}) m ; "
            f"tol=({tol_x:.6f}, {tol_y:.6f}) m"
        )
    return passed, report


def run_roundtrip_validation(
    raster_path: Path,
    species: List[str],
    tile_size_m: float = 5.0,
    overlap_m: float = 0.0,
) -> Tuple[bool, List[str]]:
    report: List[str] = []
    passed = True
    tiler = RasterTiler(raster_path, tile_size_m, overlap_m, cache_size=4)
    try:
        epsg = tiler.crs.to_epsg()
        index = tiler.n_tiles // 2  # une tuile au milieu de la grille
        info = tiler.tile_info(index)
        px = info.width / 2.0
        py = info.height / 2.0
        x, y = tiler.pixel_to_utm(index, px, py)

        sp = species[0]
        tmpdir = Path(tempfile.mkdtemp(prefix="ortho_annot_validate_"))
        out = tmpdir / "roundtrip.gpkg"
        store = AnnotationStore(out, species, epsg)
        try:
            fid = store.add_point(sp, x, y)
        finally:
            store.close()

        gdf = gpd.read_file(out, layer=sp)
        report.append(f"couche '{sp}' relue : {len(gdf)} point(s), fid inséré={fid}")

        ok_count = len(gdf) == 1
        passed = passed and ok_count
        report.append(f"[{'OK' if ok_count else 'ÉCHEC'}] exactement 1 point relu")

        ok_crs = gdf.crs is not None and gdf.crs.to_epsg() == epsg
        passed = passed and ok_crs
        report.append(f"[{'OK' if ok_crs else 'ÉCHEC'}] CRS relu = EPSG:{gdf.crs.to_epsg() if gdf.crs else None}")

        ok_label = "Label" in gdf.columns and gdf.iloc[0]["Label"] == sp
        passed = passed and ok_label
        report.append(f"[{'OK' if ok_label else 'ÉCHEC'}] Label = {gdf.iloc[0]['Label'] if 'Label' in gdf.columns else None}")

        pt = gdf.geometry.iloc[0]
        in_bounds = (
            info.left <= pt.x <= info.right and info.bottom <= pt.y <= info.top
        )
        passed = passed and in_bounds
        report.append(
            f"[{'OK' if in_bounds else 'ÉCHEC'}] point ({pt.x:.4f}, {pt.y:.4f}) "
            f"dans les bornes de la tuile "
            f"[{info.left:.4f},{info.right:.4f}]x[{info.bottom:.4f},{info.top:.4f}]"
        )

        # nettoyage du fichier temporaire (hors Dataset_Leo)
        for p in tmpdir.iterdir():
            p.unlink()
        tmpdir.rmdir()
    finally:
        tiler.close()
    return passed, report
