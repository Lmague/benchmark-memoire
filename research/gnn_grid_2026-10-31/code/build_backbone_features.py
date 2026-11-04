#!/usr/bin/env python3
"""Build a canonical-order feature matrix for any backbone.

Concatenates the train / val / test embeddings in the exact order of
``out/canonical_split.json``, drops RHOL and remaps 12-class labels when the
embeddings use that numbering, and refuses to write anything if the labels do
not match the split.

The output (``features_tile.npy``) is what ``decoding_ablation.py`` expects.

Usage:
    python build_backbone_features.py --name dinov3_vitl16_lvd \
        --emb-dir $SCRATCH/embeddings \
        --split out/canonical_split.json --out-dir out/mb/dinov3_vitl16_lvd
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

RHOL12 = 7


def files_for(emb_dir: str, name: str, split: str) -> tuple[str, str]:
    flat_x = os.path.join(emb_dir, f"{name}_{split}.npy")
    if os.path.exists(flat_x):
        return flat_x, os.path.join(emb_dir, f"{name}_{split}_labels.npy")
    return os.path.join(emb_dir, f"{split}.npy"), os.path.join(emb_dir, f"{split}_labels.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True,
                    help="flat dir with <name>_<split>.npy, or one dir holding <split>.npy")
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    tiles = json.load(open(a.split))["tiles"]
    y_ref = np.array([t["label"] for t in tiles], np.int64)
    s_ref = np.array([t["split"] for t in tiles])

    mats, schema = {}, None
    for split in ("train", "val", "test"):
        fx, fy = files_for(a.emb_dir, a.name, split)
        if not (os.path.exists(fx) and os.path.exists(fy)):
            raise SystemExit(f"[features] missing {fx} or {fy}")
        X = np.load(fx).astype(np.float32)
        y = np.load(fy).astype(np.int64)
        if schema is None:
            schema = "12cls" if int(y.max()) >= 11 else "11cls"
        if schema == "12cls":
            keep = y != RHOL12
            X, y = X[keep], y[keep]
            y = y.copy()
            y[y > RHOL12] -= 1
        m = s_ref == split
        if len(y) != int(m.sum()) or not np.array_equal(y, y_ref[m]):
            raise SystemExit(
                f"[features] {a.name}/{split}: labels do not match the canonical split "
                f"({len(y)} rows vs {int(m.sum())} expected). Order or schema differs; refusing.")
        mats[split] = X
        print(f"[features] {split}: {X.shape} ok", flush=True)

    # The canonical split is ordered by orthomosaic, not by split: write each split's
    # rows back into the global positions where that split appears.
    X = np.empty((len(s_ref), mats["train"].shape[1]), dtype=np.float32)
    for split in ("train", "val", "test"):
        pos = np.where(s_ref == split)[0]
        if len(pos) != len(mats[split]):
            raise SystemExit(f"[features] {split}: {len(pos)} positions vs {len(mats[split])} rows")
        X[pos] = mats[split]
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "features_tile.npy"), X)
    json.dump({"name": a.name, "schema": schema, "n": int(len(X)), "dim": int(X.shape[1]),
               "keys": [t["key"] for t in tiles]},
              open(os.path.join(a.out_dir, "index.json"), "w"))
    print(f"[features] wrote {a.out_dir}/features_tile.npy {X.shape} ({schema})")


if __name__ == "__main__":
    main()
