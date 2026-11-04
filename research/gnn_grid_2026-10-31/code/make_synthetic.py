#!/usr/bin/env python3
"""Test de fumée (hors science) : features synthétiques + split réduit pour valider
run_arms.py de bout en bout (grille, sonde, entraînement, bootstrap) sans GPU.

Usage : python code/make_synthetic.py --out-dir /tmp/smoke --n-orthos 6 --n-per-ortho 300
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RUBC", "SEDG", "TUSS", "WILL"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-orthos", type=int, default=6)
    ap.add_argument("--n-per-ortho", type=int, default=300)
    ap.add_argument("--dim", type=int, default=768)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.RandomState(0)

    # grille : 20x20 par ortho, stride 112 ; 300 tuiles tirées sans remise
    tiles, keys = [], []
    ortho_ids = [f"ortho{i}" for i in range(a.n_orthos)]
    for oi, o in enumerate(ortho_ids):
        cells = rng.choice(20 * 20, size=a.n_per_ortho, replace=False)
        for t, c in enumerate(cells):
            r, cc = divmod(int(c), 20)
            lab = int(rng.randint(0, 11))
            key = f"arctic_vegetation/{o}/{CLASS_NAMES_11[lab]}/tile_{t:05d}.png"
            tiles.append({"key": key, "ortho": o, "class": CLASS_NAMES_11[lab], "label": lab,
                          "tile_row_off": r * 112, "tile_col_off": cc * 112,
                          "split": "train" if oi < a.n_orthos - 2 else
                                   ("val" if oi == a.n_orthos - 2 else "test")})
            keys.append(key)
    order = np.argsort(keys)
    tiles = [tiles[i] for i in order]
    keys = [keys[i] for i in order]

    # features : signal de classe + un peu de voisinage (pour que la grille ait un sens)
    y = np.array([t["label"] for t in tiles])
    proto = rng.randn(11, a.dim).astype(np.float32) * 3
    X = proto[y] + rng.randn(len(y), a.dim).astype(np.float32)
    X2 = proto[y] + rng.randn(len(y), a.dim).astype(np.float32)
    np.save(os.path.join(a.out_dir, "features_tile.npy"), X)
    np.save(os.path.join(a.out_dir, "features_ctx512.npy"), X2)
    json.dump({"keys": keys, "views": ["tile", "ctx512"], "dim": a.dim,
               "backbone": "synthetic"}, open(os.path.join(a.out_dir, "index.json"), "w"))
    json.dump({"protocol": "synthetic", "n_tiles": len(tiles),
               "orthos": {"train": ortho_ids[:-2], "val": [ortho_ids[-2]],
                          "test": [ortho_ids[-1]]},
               "report": {}, "tiles": tiles},
              open(os.path.join(os.path.dirname(a.out_dir), "smoke_split.json"), "w"))
    print("synthetic:", len(tiles), "tiles", X.shape)


if __name__ == "__main__":
    main()
