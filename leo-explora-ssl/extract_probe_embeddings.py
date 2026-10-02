#!/usr/bin/env python3
"""Extrait les embeddings (Narval, GPU) des tuiles LABELLISÉES avec l'encodeur
adapté, pour le probe comparatif local.

Entrée : JPEG bruts transférés (mêmes fichiers que l'entraînement) + listes
d'idx labellisés (bank tr/tc + tile_labels*.csv traduits en idx).
Sortie : probe_emb.npz {idx, emb, src} où src in {bank, tile}.

Le probe lui-même (régression logistique AUPRC holdout spatial) tourne en
local CPU via eval_probe.py.

Usage (Narval) :
    python extract_probe_embeddings.py --ckpt $SCRATCH/leo_ssl/runs/.../last.pth \\
        --tiles $SCRATCH/leo_ssl/ssl_tiles --out probe_emb_adapted.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from simdinov2.models.vision_transformer import vit_base
from ssl_train import load_backbone  # noqa: E402  (réutilise le remap exact)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


@torch.inference_mode()
def embed_files(enc, files, batch=256):
    outs, idxs = [], []
    buf, bidx = [], []
    for idx, f in files:
        img = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
        buf.append((img - IMAGENET_MEAN) / IMAGENET_STD)
        bidx.append(idx)
        if len(buf) == batch:
            x = torch.tensor(np.stack(buf)).float().cuda()
            outs.append(enc(x).cpu().numpy())
            idxs.extend(bidx)
            buf, bidx = [], []
    if buf:
        x = torch.tensor(np.stack(buf)).float().cuda()
        outs.append(enc(x).cpu().numpy())
        idxs.extend(bidx)
    return np.concatenate(outs), np.array(idxs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="last.pth adapté (clé teacher)")
    ap.add_argument("--tiles", required=True, help="dossier ssl_tiles/")
    ap.add_argument("--labels", required=True,
                    help="json {idx: label} des tuiles labellisées (cf. eval_probe --dump)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc = vit_base(patch_size=16, img_size=224, layerscale=1e-5,
                   block_chunks=0, num_register_tokens=4).cuda().eval()
    # charge via le même remap que l'entraînement
    import re, tempfile
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = raw["teacher"]
    out = {}
    for k, v in sd.items():
        k2 = k.removeprefix("backbone.")
        out[re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", k2)] = v
    enc.load_state_dict(out, strict=True)

    lab = json.loads(Path(args.labels).read_text())
    # lab: {key: {"f": "{site}/{idx}.jpg", "label": ...}} (cf. eval_probe --dump)
    files = [(k, str(Path(args.tiles) / v["f"])) for k, v in lab.items()
             if (Path(args.tiles) / v["f"]).exists()]
    print(f"{len(files)}/{len(lab)} tuiles labellisées retrouvées")
    emb, keys = embed_files(enc, files)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    np.savez(args.out, key=np.array(keys),
             emb=emb.astype(np.float32))
    print("écrit:", args.out, emb.shape)


if __name__ == "__main__":
    main()
