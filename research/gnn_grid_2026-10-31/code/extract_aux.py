#!/usr/bin/env python3
"""Extrait les features des fenêtres AUXILIAIRES (non annotées mais non vides)
qui complètent le voisinage 3x3 des tuiles annotées.

Motif : le tuilage canonique ne conserve que les fenêtres annottées, donc 47 % des
3x3 sont incomplètes — pas seulement aux bords d'ortho, mais partout où l'annotation
est en patches. Un GATv2 est handicapé par ces trous ; on lui donne le graphe complet.

Les features sont extraites avec EXACTEMENT le même pipeline que les tuiles annotées
(même checkpoint, même normalisation simdino_inat, même transform eval 224), donc
elles vivent dans le même espace.

Sortie : aux_features.npy (N,768) float16 + aux_index.json [{ortho,row,col}]
Aucune image n'est écrite sur disque : lecture -> tenseur -> feature -> jeté.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio


def tile_is_empty(t: np.ndarray) -> bool:
    black = np.all(t == 0, axis=2).sum()
    white = np.all(t == 255, axis=2).sum()
    return (black + white) / (t.shape[0] * t.shape[1]) > 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--manifest", default="out/coords/coords_manifest_35.json")
    ap.add_argument("--out-dir", default="out/aux")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parents[3] / "checkpoints" / "simdinov2_vitb_inat21plantae.pth"))
    ap.add_argument("--backbone", default="simdinov2_vitb16")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke test : nb max de fenêtres")
    ap.add_argument("--orthos", default="", help="liste séparée par virgule (défaut: tous)")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from rasterio.windows import Window

    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.abspath(a.repo))
    from src.data import build_transforms
    from src.models import build_frozen_extractor
    from src.utils import get_normalization

    T, S = 224, 112
    man = json.load(open(a.manifest))["entries"]
    lab: dict[str, set] = {}
    for v in man.values():
        lab.setdefault(v["ortho"], set()).add((int(v["tile_row_off"]), int(v["tile_col_off"])))
    only = set(a.orthos.split(",")) if a.orthos else set(lab)

    # bbox des rasters (en-tête seulement)
    import PIL.Image as PImg
    PImg.MAX_IMAGE_PIXELS = None

    cands: list[tuple[str, int, int]] = []
    for o in sorted(lab):
        if o not in only:
            continue
        w, h = PImg.open(os.path.join(a.dataset_root, o, f"{o}_rgb.cog.tif")).size
        c = set()
        for (r, cc_) in lab[o]:
            for dr in (-S, 0, S):
                for dc in (-S, 0, S):
                    rr, ccc = r + dr, cc_ + dc
                    if 0 <= rr <= h - T and 0 <= ccc <= w - T:
                        c.add((rr, ccc))
        c -= lab[o]
        cands += [(o, r, cc2) for (r, cc2) in sorted(c)]
    if a.limit:
        cands = cands[: a.limit]
    print(f"[aux] {len(cands)} fenêtres candidates sur {len(only)} orthos", flush=True)

    model, fwd, dim, norm = build_frozen_extractor(a.backbone, checkpoint=a.ckpt)
    model.eval()
    mean, std = get_normalization("simdino_inat")
    tf = build_transforms("eval", mean, std, T)

    feats = np.zeros((len(cands), dim), dtype=np.float16)
    kept: list[dict] = []
    buf, buf_pos = [], []
    t0 = time.time()
    n_read = 0

    def flush() -> None:
        nonlocal buf, buf_pos
        if not buf:
            return
        x = torch.stack(buf)
        with torch.no_grad():
            y = fwd(model, x)
        y = y.float().numpy().astype(np.float16)
        start = len(kept)
        feats[start:start + len(y)] = y
        kept.extend(buf_pos)
        buf, buf_pos = [], []

    rasterio.Env(GDAL_CACHEMAX=1024)
    for o in sorted({c[0] for c in cands}):
        sub = [c for c in cands if c[0] == o]
        path = os.path.join(a.dataset_root, o, f"{o}_rgb.cog.tif")
        with rasterio.open(path) as src:
            nb = src.count
            bands = [1, 2, 3] if nb >= 3 else [1]
            for (_, r, c) in sub:
                arr = src.read(bands, window=Window(c, r, T, T))
                arr = np.transpose(arr, (1, 2, 0))
                if arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                n_read += 1
                if arr.shape[:2] != (T, T) or tile_is_empty(arr):
                    continue
                buf.append(tf(Image.fromarray(arr.astype(np.uint8))))
                buf_pos.append({"ortho": o, "row": int(r), "col": int(c)})
                if len(buf) >= a.batch_size:
                    flush()
                    if len(kept) % 5120 < a.batch_size:
                        el = time.time() - t0
                        print(f"[aux] {len(kept)} gardées / {n_read} lues  "
                              f"{100 * len(kept) / max(el, 1e-9):.1f} img/s  {el:.0f}s", flush=True)
        flush()
    flush()

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "aux_features.npy"), feats[: len(kept)])
    json.dump(kept, open(os.path.join(a.out_dir, "aux_index.json"), "w"))
    print(f"[aux] terminé : {len(kept)} fenêtres gardées sur {n_read} lues "
          f"({100 * len(kept) / max(n_read, 1):.1f} % non vides) en {time.time() - t0:.0f}s",
          flush=True)
    print(f"[aux] {os.path.join(a.out_dir, 'aux_features.npy')} "
          f"{np.load(os.path.join(a.out_dir, 'aux_features.npy')).shape}")


if __name__ == "__main__":
    main()
