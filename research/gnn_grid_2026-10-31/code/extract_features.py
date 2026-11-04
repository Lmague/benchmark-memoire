#!/usr/bin/env python3
"""Extraction des features frozen pour la grille pilote (SimDINOv2 ViT-B/16 iNat).

Deux vues par tuile, même backbone, dim 768 :
  - tile    : la tuile 224 px ;
  - ctx512  : le crop 512 px centré, redimensionné à 224 (convention clean baseline).

Écrit dans --out-dir : features_tile.npy, features_ctx512.npy, index.json,
features_manifest.csv (+ sha256), meta.json.

Préflight : --limit N permet un test court sur N tuiles (ordre déterministe).

Usage :
  python extract_features.py --manifest .../coords_manifest.json \
      --tiles-root $SLURM_TMPDIR/tiles --ctx-root $SLURM_TMPDIR/context_512 \
      --out-dir $SCRATCH/gnn_grid_2026-10-31/pilot --repo ~/benchmark-memoire
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RUBC", "SEDG", "TUSS", "WILL"]


def sha256_file(p: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tiles-root", required=True,
                    help="racine contenant <ortho>/<classe>/tile_*.png (vue tuile)")
    ap.add_argument("--ctx-root", required=True,
                    help="racine contenant <ortho>/<classe>/tile_*.png (vue 512)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default=os.path.expanduser("~/benchmark-memoire"))
    ap.add_argument("--backbone", default="simdinov2_vitb16")
    ap.add_argument("--ckpt", default=str(Path(os.environ.get("SCRATCH", "")) / "checkpoints/simdinov2_vitb_inat21plantae.pth"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--views", default="tile,ctx512")
    a = ap.parse_args()

    # --- BLAS mono-thread (les features doivent être reproductibles) ---
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, os.path.abspath(a.repo))
    from src.data import build_transforms
    from src.models import build_frozen_extractor
    from src.utils import get_normalization

    man = json.load(open(a.manifest))
    entries = man["entries"]
    keys = sorted(entries)
    if a.limit:
        keys = keys[: a.limit]
    print(f"[extract] {len(keys)} tuiles, vues={a.views}", flush=True)

    # --- vérification d'existence avant de charger le modèle ---
    missing = []
    for k in keys[:2000]:
        for root in (a.tiles_root, a.ctx_root):
            p = os.path.join(root, k)
            if not os.path.exists(p):
                missing.append(p)
                if len(missing) > 5:
                    break
    if missing:
        raise SystemExit(f"fichiers manquants (ex.) : {missing}")

    views = [v.strip() for v in a.views.split(",") if v.strip()]
    for v in views:
        if v not in ("tile", "ctx512"):
            raise SystemExit(f"vue inconnue : {v}")

    class TileDataset(Dataset):
        def __init__(self, keys, roots, tf):
            self.keys, self.roots, self.tf = keys, roots, tf

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, i):
            k = self.keys[i]
            out = []
            for root in self.roots:
                with Image.open(os.path.join(root, k)) as im:
                    out.append(self.tf(im.convert("RGB")))
            return i, torch.stack(out)

    roots = [a.tiles_root if v == "tile" else a.ctx_root for v in views]
    mean, std = get_normalization("simdino_inat")
    tf = build_transforms("eval", mean, std, 224)
    ds = TileDataset(keys, roots, tf)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    num_workers=a.workers, pin_memory=True,
                    persistent_workers=(a.workers > 0))

    model, fwd, dim, norm = build_frozen_extractor(a.backbone, checkpoint=a.ckpt)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda demandé mais CUDA indisponible")
    model.eval().to(dev)
    print(f"[extract] backbone={a.backbone} dim={dim} norm={norm} ckpt={a.ckpt}", flush=True)

    feats = {v: np.zeros((len(keys), dim), dtype=np.float32) for v in views}
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for bi, (idx, batch) in enumerate(dl):
            batch = batch.to(dev, non_blocking=True)
            for j, v in enumerate(views):
                with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
                    y = fwd(model, batch[:, j])
                feats[v][idx.numpy()] = y.float().cpu().numpy()
            done += batch.shape[0]
            if bi % 20 == 0:
                el = time.time() - t0
                print(f"[extract] {done}/{len(keys)}  {done / max(el, 1e-9):.1f} img/s", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    ckpt_hash = sha256_file(a.ckpt) if os.path.exists(a.ckpt) else "MISSING"
    out = {}
    for v in views:
        p = os.path.join(a.out_dir, f"features_{v}.npy")
        np.save(p, feats[v])
        out[v] = {"path": p, "shape": list(feats[v].shape), "dtype": str(feats[v].dtype),
                  "sha256": sha256_file(p)}
        print(f"[extract] écrit {p} {feats[v].shape}", flush=True)

    index = {"keys": keys, "views": views, "dim": dim, "backbone": a.backbone,
             "ckpt": a.ckpt, "ckpt_sha256": ckpt_hash, "norm": norm,
             "order": "sorted(coords_manifest keys)"}
    with open(os.path.join(a.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=1)

    with open(os.path.join(a.out_dir, "features_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "key", "ortho", "class", "label11", "tile_row_off",
                    "tile_col_off", "split", "feature_file", "sha256"])
        for i, k in enumerate(keys):
            e = entries[k]
            w.writerow([i, k, e["ortho"], e["class"], CLASS_NAMES_11.index(e["class"]),
                        e["tile_row_off"], e["tile_col_off"], "",
                        out[views[0]]["path"], out[views[0]]["sha256"]])

    meta = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_tiles": len(keys), "views": views, "dim": dim,
        "source_manifest": os.path.abspath(a.manifest),
        "source_manifest_sha256": sha256_file(a.manifest),
        "torch": torch.__version__, "numpy": np.__version__,
        "device": dev,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
        "elapsed_s": round(time.time() - t0, 1),
        "batch_size": a.batch_size, "workers": a.workers,
        "amp": "float16 autocast",
        "features": out,
    }
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print("[extract] terminé", json.dumps({k: v["shape"] for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()
