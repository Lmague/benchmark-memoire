#!/usr/bin/env python3
"""Rejoue la tilerization canonique (scripts/tilerization.py) pour récupérer
tile_count -> (row_off, col_off) sur les orthos SANS manifeste (ceux du train).

Pourquoi : tile_XXXXX.png n'encode pas sa position ; l'indice compte toutes les
fenêtres NON VIDES (y compris non annotées) rencontrées avant lui dans un balayage
row-major au stride 112. Le seul moyen exact de retrouver la position est de rejouer
le test de vacuité — identique à tile_is_empty de tilerization.py (seuil 0.8 sur
noir+blanc, TILE_SIZE=224, STRIDE=112).

Aucune copie d'image : on lit des bandes, on compte, on jette.
Sortie : JSON {ortho: {tile_count: [row_off, col_off]}} pour les seuls indices demandés,
plus un contrôle d'intégrité (indices rencontrés vs demandés).

Usage:
  python replay_coords.py --dataset-root "<...>/photogrammetry" \
      --indices train_indices.json --out train_coords.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio

TILE = 224
STRIDE = 112
IGNORE_THRESHOLD = 0.8


def tile_is_empty(t: np.ndarray) -> bool:
    """Copie exacte de scripts/tilerization.py:tile_is_empty."""
    black = np.all(t == 0, axis=2).sum()
    white = np.all(t == 255, axis=2).sum()
    return (black + white) / (t.shape[0] * t.shape[1]) > IGNORE_THRESHOLD


def replay(raster_path: str, wanted: set[int], log=print) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    with rasterio.open(raster_path) as src:
        w, h = src.width, src.height
        n_bands = src.count
        bands = [1, 2, 3] if n_bands >= 3 else [1]
        count = 0
        t0 = time.time()
        # lecture par bande de TILE lignes, avancée de STRIDE : chaque pixel lu 2x
        for row_off in range(0, h - TILE + 1, STRIDE):
            strip = src.read(bands, window=rasterio.windows.Window(0, row_off, w, TILE))
            strip = np.transpose(strip, (1, 2, 0))
            if strip.shape[2] == 1:
                strip = np.repeat(strip, 3, axis=2)
            for col_off in range(0, w - TILE + 1, STRIDE):
                if tile_is_empty(strip[:, col_off:col_off + TILE]):
                    continue
                count += 1
                # offset +1 calibré empiriquement : sur 2 orthos témoins (1 476 tuiles),
                # l'indice du manifeste vaut (compteur rejoué + 1), sans exception.
                idx = count + 1
                if idx in wanted:
                    out[idx] = (int(row_off), int(col_off))
            if row_off % (STRIDE * 200) == 0:
                log(f"    row {row_off}/{h}  count={count}  trouvés={len(out)}/{len(wanted)}  "
                    f"{time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--indices", required=True, help="{ortho: [tile_count, ...]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="orthos séparés par virgule (défaut: tous)")
    a = ap.parse_args()

    wanted = json.load(open(a.indices))
    only = set(a.only.split(",")) if a.only else set(wanted)
    result, summary = {}, {}
    for ortho in sorted(wanted):
        if ortho not in only:
            continue
        path = os.path.join(a.dataset_root, ortho, f"{ortho}_rgb.cog.tif")
        if not os.path.exists(path):
            print(f"[ERREUR] raster absent: {path}", file=sys.stderr)
            continue
        need = set(int(x) for x in wanted[ortho])
        t0 = time.time()
        print(f"[{ortho}] {len(need)} indices demandés", flush=True)
        got = replay(path, need)
        missing = len(need) - len(got)
        summary[ortho] = {"demandes": len(need), "trouves": len(got), "manquants": missing,
                          "secondes": round(time.time() - t0, 1)}
        print(f"[{ortho}] terminé: {len(got)}/{len(need)} en {time.time() - t0:.0f}s", flush=True)
        result[ortho] = {str(k): v for k, v in got.items()}
        json.dump(result, open(a.out, "w"))          # écriture incrémentale
        json.dump(summary, open(a.out + ".summary.json", "w"), indent=1)
    print("TOTAL:", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
