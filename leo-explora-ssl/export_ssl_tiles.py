#!/usr/bin/env python3
"""Export BRUT (pixels natifs, SANS stretch) des tuiles 224x224 pour le SSL.

Différence avec export_tiles.py (affichage stretché) : ici on veut la vraie
distribution du capteur pour l'extended pre-training ExPLoRA. Même fenêtrage
(tr*224, borné au raster), même convention d'idx globaux.

Sorties :
  - $OUT/ssl_tiles/{site}/{idx}.jpg   (JPEG q95, pixels bruts)
  - $OUT/ssl_manifest_{site}.json     {tiles: {idx: {tr,tc,part,green,mean}},
                                       meta: {site, parts, tile: 224}}

Le score `green` (Excess Green moyen 2G-R-B) sert au sur-échantillonnage des
tuiles végétalisées côté entraînement (le fond représente ~97% des tuiles).

Usage :
    ~/miniconda3/bin/python3 export_ssl_tiles.py            # tout
    ~/miniconda3/bin/python3 export_ssl_tiles.py --site maison --limit 500
"""
from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = Path("/home/erazal/annotations_leo")
D = Path("/home/erazal/Documents/Mémoire/Dataset_Leo/Orthomosaiques")
OUT = Path(__file__).resolve().parent / "data"  # relatif au dépôt (déplaçable)

SITES = {
    "clairiere": [D / "Orthom_Clairiere_9Aout23_WGS84UTM18N.tif"],
    "maison": [D / "Orthom_Maison_9Aout23_WGS84UTM18N.tif"],
    "trail": [D / "Orthom_TrailErable_9Aout23_WGS84UTM18N.tif"],
    "rousseau": [
        D / "Orthom_Rousseau_9Aout23_WGS84UTM18N_0-0.tif",
        D / "Orthom_Rousseau_9Aout23_WGS84UTM18N_0-1.tif",
        D / "Orthom_Rousseau_9Aout23_WGS84UTM18N_1-0.tif",
        D / "Orthom_Rousseau_9Aout23_WGS84UTM18N_1-1.tif",
    ],
}


def encode_jpeg(img, quality=95):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def export_site(site, rasters, out_root, manifest_path, quality, workers, limit):
    import rasterio
    from rasterio.windows import Window

    tiles_meta: dict[str, dict] = {}
    if manifest_path.exists():
        tiles_meta = json.loads(manifest_path.read_text()).get("tiles", {})
        print(f"reprise {site}: {len(tiles_meta)} tuiles déjà exportées.")

    parts, offset = [], 0
    # dimensions d'abord (offsets globaux stables)
    dims = []
    for rp in rasters:
        with rasterio.open(rp) as src:
            n_tc, n_tr = src.width // 224, src.height // 224
        dims.append((n_tr, n_tc))
        parts.append({"raster": Path(rp).name, "n_tr": n_tr, "n_tc": n_tc,
                      "offset": offset})
        offset += n_tr * n_tc

    pool = ThreadPoolExecutor(max_workers=workers)
    pending, stats = [], {"saved": 0, "blank": 0, "t0": time.time()}

    def flush():
        for fut, idx, meta, dest in pending:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(fut.result())
            tiles_meta[str(idx)] = meta
            stats["saved"] += 1
        pending.clear()

    out_dir = out_root / site
    for part, rp in enumerate(rasters):
        n_tr, n_tc = dims[part]
        off = parts[part]["offset"]
        src = rasterio.open(rp)
        H, W = src.height, src.width
        print(f"{site} morceau {part}: {n_tr}x{n_tc} ({n_tr*n_tc} tuiles, off {off})",
              flush=True)
        for tr in range(n_tr):
            if limit and stats["saved"] + len(pending) >= limit:
                break
            row_tiles = []
            for tc in range(n_tc):
                idx = off + tr * n_tc + tc
                if str(idx) in tiles_meta:
                    continue
                row_tiles.append((idx, tr, tc))
            if not row_tiles:
                continue
            r = max(0, min(H - 224, tr * 224))
            band = src.read([1, 2, 3], window=Window(0, r, W, min(224, H - r)))
            band = np.transpose(band, (1, 2, 0)).astype(np.uint8)
            for idx, tr_, tc in row_tiles:
                c = max(0, min(W - 224, tc * 224))
                w = band[0:224, c:c + 224]
                if w.shape[:2] != (224, 224):
                    continue
                m = float(w.mean())
                if m > 252 or m < 3:
                    stats["blank"] += 1
                    continue
                f = w.astype(np.float32)
                green = float((2 * f[:, :, 1] - f[:, :, 0] - f[:, :, 2]).mean())
                meta = {"part": part, "tr": tr_, "tc": tc, "green": round(green, 2),
                        "mean": round(m, 1), "f": f"{site}/{idx}.jpg"}
                pending.append((pool.submit(encode_jpeg, w, quality),
                                idx, meta, out_dir / f"{idx}.jpg"))
                if limit and stats["saved"] + len(pending) >= limit:
                    break
            if len(pending) > 400:
                flush()
        src.close()
        flush()
    pool.shutdown()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"meta": {"site": site, "parts": parts, "tile": 224, "raw": True},
         "tiles": tiles_meta}))
    print(f"TERMINÉ {site}: {stats['saved']} jpg (+reprise), "
          f"{stats['blank']} vides ({time.time()-stats['t0']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=None, help="un seul site (défaut: tous)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="test : max tuiles/site")
    args = ap.parse_args()

    out_root = Path(args.out) / "ssl_tiles"
    sites = [args.site] if args.site else list(SITES)
    for site in sites:
        export_site(site, SITES[site], out_root,
                    Path(args.out) / f"ssl_manifest_{site}.json",
                    args.quality, args.workers, args.limit)


if __name__ == "__main__":
    main()
