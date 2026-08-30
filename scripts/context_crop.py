#!/usr/bin/env python3
"""Découpe une fenêtre de CONTEXTE spatial (512/1024/2048px) autour de chaque tuile
d'un split, pour l'expérience de self-distillation contexte→tuile (voir
``scripts/context_distill.py`` et ``scripts/context_distill_README.md``).

── Pourquoi ce script existe (problème résolu) ──────────────────────────────────
Les fichiers ``tile_XXXXX.png`` produits par ``scripts/tilerization.py`` n'encodent
PAS leur position pixel dans l'orthomosaïque mère : ``tile_count`` s'incrémente pour
chaque fenêtre glissante NON VIDE (``tile_is_empty``), mais seule une fraction de ces
fenêtres est sauvée (celles qui intersectent une annotation, cf. tilerization.py
lignes 130-164). D'où les trous observés dans la numérotation (06871 → 06894, etc.) :
tile_count encode une position dans la séquence des fenêtres non-vides, pas dans la
grille (row_off, col_off) elle-même.

La seule façon fiable de retrouver (row_off, col_off) pour un ``tile_XXXXX.png`` donné
est donc de REJOUER exactement la même boucle (même ordre, même filtre d'emptiness,
même intersection de labels) sur le raster mère. Ce script réutilise DIRECTEMENT les
fonctions de ``tilerization.py`` (``tile_is_empty``, ``get_dominant_class``, mêmes
constantes ``TILE_SIZE``/``STRIDE``/``IGNORE_THRESHOLD``) pour ne jamais diverger de la
logique originale.

Validation (2026-08-29, hors dépôt, voir scripts/context_distill_README.md §Validation) :
rejoué sur l'ortho ``20230724_alder39_m3m`` (28369 fenêtres, 2489 tuiles annotées) —
égalité d'ENSEMBLE parfaite entre les tile_count reconstruits et les tile_XXXXX.png
réellement sur disque (2489/2489), classe dominante identique pour les 2489 tuiles
communes. Le replay est fidèle SI le raster n'a pas été réécrit depuis la tuilisation
originale (le filtre ``tile_is_empty`` dépend des octets exacts de l'image).

── Choix de conception : la fenêtre de contexte est REDIMENSIONNÉE avant sauvegarde ──
Un contexte natif de 2048px en PNG pèserait ~10 Mo/tuile (train = 49433 tuiles ⇒
~500 Go) — intraitable à transférer vers Narval et à charger en RAM pendant
l'entraînement. Ce script sauve donc le contexte à une résolution FIXE ``--out-size``
(défaut 224, la résolution native attendue par le teacher DINOv3), quel que soit
``--context-size`` demandé. C'est exactement le principe des "global crops" style
DINO/DINOv2 : la fenêtre est plus grande dans l'espace pixel de l'ortho (donc GSD
effectif plus grossier après redimensionnement), mais le tenseur réseau reste de taille
constante. Le "token d'échelle" du student (cf. context_distill.py) encode précisément
ce ratio ``context_size / TILE_SIZE`` pour que le modèle sache quel GSD effectif il
regarde. Sortie : ~même poids qu'une tuile normale (~120 Ko/PNG), quel que soit
context_size → ``context_<size>.zip`` a le même ordre de grandeur que ``tiles.zip``.

── Politique de bord ──────────────────────────────────────────────────────────────
Une fenêtre de contexte centrée sur une tuile proche du bord de l'ortho déborderait
souvent des dimensions du raster (ex. contexte 2048px sur une tuile à moins de 1024px
du bord). Politique choisie : CLAMP (la fenêtre glisse pour rester entièrement dans le
raster, donc n'est plus rigoureusement centrée) plutôt que padding noir — le padding
noir introduirait un signal artificiel que ``tile_is_empty`` traite justement comme
"absence de donnée" ailleurs dans le pipeline. Chaque fenêtre réellement lue est
enregistrée dans le manifeste (``context_row0/col0``, ``clamped``) — le script de
distillation n'a pas à deviner.

── Fuite spatiale ────────────────────────────────────────────────────────────────
Le split spatial (spatial_datacurve/splits/) sépare train/val/test par ORTHOMOSAÏQUE ENTIÈRE.
Une fenêtre de contexte est découpée dans le MÊME raster que sa tuile — jamais dans un
autre — donc un contexte de train ne peut, par construction, jamais piocher des pixels
d'un ortho test. Pas de fuite additionnelle introduite par ce script.

── Usage ──────────────────────────────────────────────────────────────────────────
    python scripts/context_crop.py \\
        --split-csv spatial_datacurve/splits/frac100_seed0/train.csv \\
        --context-sizes 512,1024,2048 \\
        --out-size 224 \\
        --out-dir out/context \\
        --dataset-root "High-Resolution Arctic Vegetation Maps and Photogrammetry Data from Drone Surveys at Trail Valley Creek, Northwest Territories"

Sortie (sous ``--out-dir``) :
    context_512/arctic_vegetation/<ortho>/<CLASSE>/tile_XXXXX.png   (out_size × out_size)
    context_1024/...
    context_2048/...
    coords_manifest.json   — {relpath: {ortho, tile_count, class, tile_row_off,
                              tile_col_off, contexts: {size: {row0, col0, clamped}}}}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ (sibling import)
from tilerization import (  # noqa: E402 — réutilise la logique canonique, ne PAS réimplémenter
    IGNORE_THRESHOLD,
    LABELS_FILE,
    STRIDE,
    TILE_SIZE,
    get_dominant_class,
    tile_is_empty,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DATASET_DIR = ("High-Resolution Arctic Vegetation Maps and Photogrammetry Data "
                         "from Drone Surveys at Trail Valley Creek, Northwest Territories")


def _parse_split_csvs(paths: list[str]) -> dict[str, dict[int, tuple[str, str]]]:
    """Lit un ou plusieurs CSV ``(filepath,label)`` → ``{ortho: {tile_count: (relpath, classe)}}``.

    ``filepath`` attendu : ``arctic_vegetation/<ortho>/<CLASSE>/tile_XXXXX.png``
    (convention fixée par tilerization.py / spatial_datacurve/splits). Lève si un filepath ne
    matche pas ce format — mieux vaut échouer bruyamment que deviner.
    """
    needed: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    n_rows = 0
    for csv_path in paths:
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)  # header
            for row in r:
                if len(row) < 2:
                    continue
                fp = row[0]
                parts = fp.split("/")
                if len(parts) != 4 or parts[0] != "arctic_vegetation" or not parts[3].startswith("tile_"):
                    raise ValueError(f"filepath inattendu (format non reconnu) : {fp!r} dans {csv_path}")
                _, ortho, cls, fname = parts
                tile_count = int(fname[len("tile_"):-len(".png")])
                needed[ortho][tile_count] = (fp, cls)
                n_rows += 1
    print(f"[context_crop] {n_rows} tuiles demandées sur {len(needed)} orthos "
          f"(depuis {len(paths)} CSV).")
    return needed


def _clamp_window(center: int, size: int, dim: int) -> tuple[int, bool]:
    """Renvoie ``(start, clamped)`` pour une fenêtre ``size`` centrée sur ``center``
    dans ``[0, dim)``. Réduit ``size`` en dernier recours si ``size > dim`` (raster
    plus petit que le contexte demandé — cas limite non rencontré sur Arctic-TVC mais
    protégé explicitement plutôt que de planter sur un IndexError obscur).
    """
    if size >= dim:
        return 0, True
    start = center - size // 2
    clamped = False
    if start < 0:
        start, clamped = 0, True
    elif start + size > dim:
        start, clamped = dim - size, True
    return start, clamped


def _read_rgb_window(src, row_off: int, col_off: int, size: int, n_bands: int) -> np.ndarray:
    import rasterio.windows
    bands_to_read = [1, 2, 3] if n_bands >= 3 else [1]
    window = rasterio.windows.Window(col_off, row_off, size, size)
    tile = src.read(bands_to_read, window=window)
    tile = np.transpose(tile, (1, 2, 0))
    if tile.shape[2] == 1:
        tile = np.repeat(tile, 3, axis=2)
    return tile


def _process_ortho(ortho: str, needed_for_ortho: dict[int, tuple[str, str]],
                   dataset_root: str, out_dirs: dict[int, str], out_size: int,
                   context_sizes: list[int], manifest: dict) -> None:
    import rasterio
    import rasterio.windows
    import geopandas as gpd
    from PIL import Image
    from shapely import box
    from shapely.strtree import STRtree

    t0 = time.time()
    raster_path = os.path.join(dataset_root, "photogrammetry", ortho, f"{ortho}_rgb.cog.tif")
    if not os.path.exists(raster_path):
        raise FileNotFoundError(
            f"raster introuvable pour l'ortho '{ortho}' : {raster_path!r} — "
            "vérifier --dataset-root (les COG bruts doivent être présents localement "
            "ou rsyncés sur Narval, cf. scripts/context_distill_README.md).")

    labels_all = gpd.read_file(LABELS_FILE if os.path.isabs(LABELS_FILE)
                               else os.path.join(dataset_root, LABELS_FILE))

    needed_ids = set(needed_for_ortho.keys())
    found: dict[int, tuple[int, int, str]] = {}

    with rasterio.open(raster_path) as src:
        width, height = src.width, src.height
        crs = src.crs
        transform = src.transform
        n_bands = src.count

        rast_bounds = box(*src.bounds)
        labels_proj = labels_all.to_crs(crs)
        mask = labels_proj.intersects(rast_bounds)
        labels_local = labels_proj[mask].copy()
        geometries = labels_local.geometry.values
        tree = STRtree(geometries)

        tile_count = 0
        for row_off in range(0, height - TILE_SIZE + 1, STRIDE):
            if len(found) == len(needed_ids):
                break
            for col_off in range(0, width - TILE_SIZE + 1, STRIDE):
                tile = _read_rgb_window(src, row_off, col_off, TILE_SIZE, n_bands)
                if tile_is_empty(tile, IGNORE_THRESHOLD):
                    continue
                tile_count += 1

                left = transform[2] + col_off * transform[0] + row_off * transform[1]
                top = transform[5] + col_off * transform[3] + row_off * transform[4]
                right = left + TILE_SIZE * transform[0]
                bottom = top + TILE_SIZE * transform[4]
                tile_bbox = box(min(left, right), min(top, bottom), max(left, right), max(top, bottom))

                query_indices = tree.query(tile_bbox, predicate="intersects")
                if len(query_indices) == 0:
                    continue
                dominant_class = get_dominant_class(query_indices, labels_local)
                if dominant_class is None:
                    continue

                if tile_count in needed_ids:
                    found[tile_count] = (row_off, col_off, str(dominant_class))
            if len(found) == len(needed_ids):
                break

        missing = needed_ids - set(found.keys())
        if missing:
            raise RuntimeError(
                f"[context_crop] ortho={ortho} : {len(missing)} tuiles demandées "
                f"introuvables dans le replay (ex. tile_count={sorted(missing)[:5]}) — "
                "le raster a probablement changé depuis la tuilisation originale. "
                "Déclaré bloqué plutôt que de produire un contexte désaligné.")

        for tile_count, (row_off, col_off, replay_cls) in found.items():
            relpath, csv_cls = needed_for_ortho[tile_count]
            if replay_cls != csv_cls:
                raise RuntimeError(
                    f"[context_crop] ortho={ortho} tile_count={tile_count} : classe "
                    f"rejouée '{replay_cls}' != classe du split '{csv_cls}' — désalignement, "
                    "bloqué plutôt que silencieusement faux.")

            center_row = row_off + TILE_SIZE // 2
            center_col = col_off + TILE_SIZE // 2
            entry = {
                "ortho": ortho, "tile_count": tile_count, "class": csv_cls,
                "tile_row_off": row_off, "tile_col_off": col_off,
                "contexts": {},
            }
            for size in context_sizes:
                c_row0, clamped_r = _clamp_window(center_row, size, height)
                c_col0, clamped_c = _clamp_window(center_col, size, width)
                eff_size = min(size, height, width)
                crop = _read_rgb_window(src, c_row0, c_col0, eff_size, n_bands)
                img = Image.fromarray(crop.astype(np.uint8))
                if eff_size != out_size:
                    img = img.resize((out_size, out_size), Image.LANCZOS)
                dst = os.path.join(out_dirs[size], relpath)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                img.save(dst)
                entry["contexts"][str(size)] = {
                    "row0": c_row0, "col0": c_col0, "size_px": eff_size,
                    "clamped": bool(clamped_r or clamped_c),
                }
            manifest[relpath] = entry

    print(f"[context_crop] ortho={ortho}: {len(found)}/{len(needed_ids)} tuiles "
          f"({time.time() - t0:.1f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-csv", nargs="+", required=True,
                    help="un ou plusieurs CSV (filepath,label) du split spatial, ex. "
                         "spatial_datacurve/splits/frac100_seed0/train.csv (Design A : train UNIQUEMENT, "
                         "le contexte n'est pas utilisé à l'inférence).")
    ap.add_argument("--context-sizes", default="1024",
                    help="tailles de fenêtre en pixels natifs de l'ortho, séparées par virgule "
                         "(ex. '512,1024,2048'). Un seul passage sur chaque raster produit "
                         "TOUTES les tailles demandées (évite de relire le raster N fois).")
    ap.add_argument("--out-size", type=int, default=224,
                    help="résolution de sauvegarde (redimensionnement LANCZOS), défaut 224 "
                         "= résolution native attendue par le teacher DINOv3.")
    ap.add_argument("--out-dir", required=True,
                    help="racine de sortie ; crée out-dir/context_<size>/... par taille.")
    ap.add_argument("--dataset-root", default=os.path.join(_PROJECT_ROOT, _DEFAULT_DATASET_DIR),
                    help="racine du dataset Arctic-TVC brut (contient photogrammetry/<ortho>/"
                         "<ortho>_rgb.cog.tif et labels/2023_tvc_labels.gpkg). PAR DÉFAUT le "
                         "chemin local du dépôt — sur Narval, transférer les COG bruts sous "
                         "$SCRATCH et pointer --dataset-root là (cf. scripts/context_distill_README.md, "
                         "les COG NE SONT PAS dans tiles.zip).")
    ap.add_argument("--manifest-out", default=None,
                    help="chemin du manifeste JSON (défaut : <out-dir>/coords_manifest.json).")
    args = ap.parse_args()

    context_sizes = sorted({int(s) for s in args.context_sizes.split(",")})
    os.makedirs(args.out_dir, exist_ok=True)
    out_dirs = {size: os.path.join(args.out_dir, f"context_{size}") for size in context_sizes}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    needed = _parse_split_csvs(args.split_csv)
    manifest: dict = {}
    t0 = time.time()
    for i, (ortho, needed_for_ortho) in enumerate(sorted(needed.items())):
        print(f"[context_crop] [{i + 1}/{len(needed)}] ortho={ortho} "
              f"({len(needed_for_ortho)} tuiles)...", flush=True)
        _process_ortho(ortho, needed_for_ortho, args.dataset_root, out_dirs,
                       args.out_size, context_sizes, manifest)

    manifest_path = args.manifest_out or os.path.join(args.out_dir, "coords_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "context_sizes": context_sizes, "out_size": args.out_size,
            "tile_size": TILE_SIZE, "stride": STRIDE, "ignore_threshold": IGNORE_THRESHOLD,
            "n_tiles": len(manifest), "n_orthos": len(needed),
            "entries": manifest,
        }, f, indent=2)

    print(f"\n[context_crop] TERMINÉ : {len(manifest)} tuiles × {len(context_sizes)} tailles "
          f"({time.time() - t0:.1f}s total)")
    print(f"[context_crop] manifeste → {manifest_path}")
    for size, d in out_dirs.items():
        print(f"[context_crop] context_{size}/ → {d}")


if __name__ == "__main__":
    main()
