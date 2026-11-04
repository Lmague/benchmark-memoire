#!/usr/bin/env python3
"""Descripteurs non appris par tuile : couleur, indices RGB, texture, structure.

Motivation : les features de fondation gelées saturent (~0,47 sur la tuile seule) et
l'agrégation de voisinage ne gagne presque rien. On teste l'autre canal d'information
disponible sans GPU : la radiométrie et la texture brutes, que le ViT iNat peut avoir
sous-représentées.

Aucune image n'est écrite : seuls les descripteurs (≈57 dims) sont sauvegardés.
Ordre de sortie = ordre du split trié par (ortho, ligne, colonne) pour une lecture
séquentielle des COG.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

TILE = 224
EPS = 1e-6


def lbp_hist(gray_u8: np.ndarray, bins: int = 10) -> np.ndarray:
    """LBP 8-voisins, histogramme de 10 classes (uniforme simplifié)."""
    c = gray_u8[1:-1, 1:-1].astype(np.int16)
    code = np.zeros(c.shape, np.int16)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for k, (dr, dc) in enumerate(offs):
        n = gray_u8[1 + dr:gray_u8.shape[0] - 1 + dr, 1 + dc:gray_u8.shape[1] - 1 + dc]
        code |= ((n.astype(np.int16) >= c) << k).astype(np.int16)
    # 10 classes : nombre de transitions 0->1 (0..8) ramené à 10 bins
    b = np.unpackbits(code.astype(np.uint8).reshape(-1, 1), axis=1)[:, -8:]
    bb = np.concatenate([b, b[:, :1]], axis=1).astype(np.int8)  # uint8 -> int8 : pas de bouclage
    trans = np.abs(np.diff(bb, axis=1)).sum(1)
    h = np.bincount(trans // 2, minlength=bins).astype(np.float32)
    return h / max(h.sum(), 1.0)


def glcm_feats(gray_u8: np.ndarray, levels: int = 16) -> np.ndarray:
    """Contraste, homogénéité, énergie, corrélation moyennés sur 4 orientations."""
    q = (gray_u8.astype(np.uint16) * levels // 256).astype(np.int32)
    out = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = q[max(0, dr):q.shape[0] + min(0, dr), max(0, dc):q.shape[1] + min(0, dc)]
        b = q[max(0, -dr):q.shape[0] - max(0, dr), max(0, -dc):q.shape[1] - max(0, dc)]
        m = np.bincount(a.ravel() * levels + b.ravel(), minlength=levels * levels)
        P = m.reshape(levels, levels).astype(np.float64)
        P /= max(P.sum(), 1.0)
        i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        mu = (P * i).sum()
        muj = (P * j).sum()
        si = np.sqrt((P * (i - mu) ** 2).sum())
        sj = np.sqrt((P * (j - muj) ** 2).sum())
        out.append([(P * (i - j) ** 2).sum(), (P / (1.0 + np.abs(i - j))).sum(),
                    (P ** 2).sum(),
                    ((P * (i - mu) * (j - muj)).sum() / max(si * sj, EPS))])
    return np.mean(np.array(out), axis=0).astype(np.float32)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.where(d < EPS, 0.0, np.where(mx == r, ((g - b) / (d + EPS)) % 6,
                                       np.where(mx == g, (b - r) / (d + EPS) + 2,
                                                (r - g) / (d + EPS) + 4)))
    return np.stack([h * 60.0, np.where(mx > EPS, d / (mx + EPS), 0.0), mx], -1)


def descriptors(rgb_u8: np.ndarray) -> np.ndarray:
    rgb = rgb_u8.astype(np.float32) / 255.0
    f = []
    for c in range(3):
        ch = rgb[..., c]
        f += [ch.mean(), ch.std(), *np.percentile(ch, [10, 50, 90])]
    hsv = rgb_to_hsv(rgb)
    ang = np.deg2rad(hsv[..., 0])
    f += [np.cos(ang).mean(), np.sin(ang).mean(), hsv[..., 0].std(),
          hsv[..., 1].mean(), hsv[..., 1].std(), hsv[..., 2].mean(), hsv[..., 2].std()]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    idx = {"exg": 2 * g - r - b,
           "vari": (g - r) / (g + r - b + EPS),
           "gli": (2 * g - r - b) / (2 * g + r + b + EPS),
           "tgi": g - 0.39 * r - 0.61 * b}
    for v in idx.values():
        f += [v.mean(), v.std(), *np.percentile(v, [10, 90])]
    gray = rgb.mean(-1)
    gu8 = (gray * 255).astype(np.uint8)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    f += [mag.mean(), mag.std(), *np.percentile(mag, [90, 99])]
    f.append(np.abs(np.diff(gray, 2, axis=0)).mean())
    hist = np.bincount((gu8.ravel() >> 3), minlength=32).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    f.append(-(p[p > 0] * np.log(p[p > 0])).sum())
    f += lbp_hist(gu8).tolist()
    f += glcm_feats(gu8).tolist()
    return np.asarray(f, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out-dir", default="out/color")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    import rasterio
    from rasterio.windows import Window

    os.makedirs(a.out_dir, exist_ok=True)
    tiles = sorted(json.load(open(a.split))["tiles"],
                   key=lambda t: (t["ortho"], t["tile_row_off"], t["tile_col_off"]))
    if a.limit:
        tiles = tiles[a.start:a.start + a.limit]
    elif a.start:
        tiles = tiles[a.start:]

    feats = None
    idx = []
    t0 = time.time()
    cur = None
    src = None
    for i, t in enumerate(tiles):
        o = t["ortho"]
        if o != cur:
            if src is not None:
                src.close()
            path = os.path.join(a.dataset_root, o, f"{o}_rgb.cog.tif")
            src = rasterio.open(path)
            cur = o
        win = Window(t["tile_col_off"], t["tile_row_off"], TILE, TILE)
        arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        rgb = np.transpose(arr, (1, 2, 0))
        d = descriptors(rgb)
        if feats is None:
            feats = np.zeros((len(tiles), d.size), dtype=np.float32)
        feats[i] = d
        idx.append({"ortho": o, "row": t["tile_row_off"], "col": t["tile_col_off"]})
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            print(f"[color] {i + 1}/{len(tiles)}  {len(tiles) / el:.1f} img/s  {el:.0f}s",
                  flush=True)
    if src is not None:
        src.close()

    np.save(os.path.join(a.out_dir, "color_features.npy"), feats)
    json.dump(idx, open(os.path.join(a.out_dir, "color_index.json"), "w"))
    print(f"[color] terminé : {feats.shape} en {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
