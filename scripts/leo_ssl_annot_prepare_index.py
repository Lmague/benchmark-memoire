#!/usr/bin/env python3
"""Prépare l'INDEX des tuiles annotées (local, sans GPU) — pour l'extraction sur Narval.

Pourquoi ce script
------------------
Les annotations (`session.gpkg`) donnent des coordonnées UTM, pas des indices de tuile.
Le passage UTM → (tr, tc) → idx web exige le **transform du raster**, qui n'existe que
localement (les COG pèsent 131 Go). On fait donc ici le travail « raster » — quelques
lectures d'en-tête et de l'arithmétique — et on expédie à Narval un fichier d'INDEX de
quelques centaines de ko : Narval n'a plus qu'à faire les forward passes sur les JPEG
déjà présents dans `$SCRATCH/web/tiles/clairiere/`.

Vérifié : les 6037 points annotés sont **tous dans l'ortho Clairière**
(`idx = tr*260 + tc`, part 0, offset 0 — même convention que `web/manifest_clairiere.json`
et que `leo_extract_big.py --bank-npz`).

Sortie : `<out>/annot_index.npz`
    bank : species, fid, x, y, tr, tc, idx, tile_exists(à vérifier sur Narval)
    bg   : tr, tc, idx, x, y            (fond aléatoire, rng seed 0, exclu ±3 tuiles)
    rej  : species, tr, tc, idx, x, y   (candidats REJETÉS de candidates.sqlite)

Les négatifs sont **appariés** (mêmes tuiles pour tous les modèles) : la liste ne dépend
que des annotations et du rng, jamais du checkpoint.

Usage :
    python scripts/leo_ssl_annot_prepare_index.py --out ~/annot_index.npz
Puis (voir scripts/leo_ssl_annot_README.md §3) :
    rsync -avP ~/annot_index.npz ~/annotations_leo/session.gpkg \\
               ~/annotations_leo/prospect/candidates.sqlite \\
               ~/annotations_leo/web/manifest_clairiere.json \\
               narval:$SCRATCH/annot_leo/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_ANNOT = Path("/home/erazal/annotations_leo")
TILE = 224
N_BG = 3000
BUF = 3


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annot-dir", default=str(DEFAULT_ANNOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bg", type=int, default=N_BG)
    ap.add_argument("--buf", type=int, default=BUF)
    args = ap.parse_args()

    annot = Path(args.annot_dir)
    sys.path.insert(0, str(annot))
    try:
        import extract_embeddings as E          # noqa: E402 — lecture gpkg + constantes
    except ImportError as e:
        raise SystemExit(
            f"[env] import impossible dans {sys.executable}\n      {e}\n"
            "      → mauvais interpréteur. Utiliser l'env miniconda3 (qui a rasterio) :\n"
            "        /home/erazal/miniconda3/bin/python scripts/leo_ssl_annot_prepare_index.py ...\n"
            "      ou :  source /home/erazal/miniconda3/etc/profile.d/conda.sh && conda activate base") from e
    import rasterio
    from rasterio import transform as rt

    with rasterio.open(E.RASTER) as src:
        W, H, trf = src.width, src.height, src.transform
        n_tr, n_tc = H // TILE, W // TILE
    print(f"raster {W}x{H} → grille {n_tr} x {n_tc} (n_tc={n_tc})")

    # ---- bank (points annotés) ----
    pts = []
    for sp, fid, x, y in E._read_points():
        r, c = rt.rowcol(trf, float(x), float(y))
        tr_i, tc_i = int(r) // TILE, int(c) // TILE
        inb = 0 <= tr_i < n_tr and 0 <= tc_i < n_tc
        pts.append((sp, fid, float(x), float(y), tr_i, tc_i, tr_i * n_tc + tc_i, int(inb)))
    sp_a = np.array([p[0] for p in pts])
    print(f"bank : {len(pts)} points | {len({p[6] for p in pts if p[7]})} tuiles uniques")
    out_pts = {k: np.array(v) for k, v in zip(
        ("species", "fid", "x", "y", "tr", "tc", "idx", "inb"),
        zip(*[(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]) for p in pts]))}

    # ---- fond aléatoire (hors ±buf tuiles autour d'une annotation) ----
    ann = {(p[4], p[5]) for p in pts if p[7]}
    rng = np.random.default_rng(0)
    bg, tries = [], 0
    while len(bg) < args.n_bg and tries < args.n_bg * 40:
        tries += 1
        tr_i, tc_i = int(rng.integers(0, n_tr)), int(rng.integers(0, n_tc))
        if any((tr_i + dtr, tc_i + dtc) in ann
               for dtr in range(-args.buf, args.buf + 1)
               for dtc in range(-args.buf, args.buf + 1)):
            continue
        bg.append((tr_i, tc_i, tr_i * n_tc + tc_i))
    bg = np.asarray(bg)
    # centres UTM (nécessaires aux folds spatiaux du banc d'évaluation côté Narval)
    bg_xy = np.array([rasterio.transform.xy(trf, tr_i * TILE + TILE / 2,
                                            tc_i * TILE + TILE / 2)
                      for tr_i, tc_i in bg[:, :2]])
    print(f"fond  : {len(bg)} tuiles")

    # ---- candidats rejetés ----
    cand = annot / "prospect" / "candidates.sqlite"
    rej = []
    if cand.exists():
        import sqlite3
        con = sqlite3.connect(f"file:{cand}?mode=ro", uri=True)
        try:
            rows = con.execute(
                "SELECT species, x, y FROM candidates WHERE status='rejected'").fetchall()
        finally:
            con.close()
        for sp, x, y in rows:
            r, c = rt.rowcol(trf, float(x), float(y))
            tr_i, tc_i = int(r) // TILE, int(c) // TILE
            if 0 <= tr_i < n_tr and 0 <= tc_i < n_tc:
                rej.append((sp, tr_i, tc_i, tr_i * n_tc + tc_i, float(x), float(y)))
    rej = np.asarray(rej, dtype=object) if rej else np.zeros((0, 6), dtype=object)
    print(f"rejet : {len(rej)} points | tuiles uniques totales "
          f"{len(set(bg[:, 2].tolist()) | set(rej[:, 3].tolist()) if len(rej) else set(bg[:, 2].tolist()))}")

    np.savez_compressed(
        args.out, **{f"bank_{k}": v for k, v in out_pts.items()},
        bg_tr=bg[:, 0].astype(np.int32), bg_tc=bg[:, 1].astype(np.int32),
        bg_idx=bg[:, 2].astype(np.int32),
        bg_x=bg_xy[:, 0], bg_y=bg_xy[:, 1],
        rej_species=rej[:, 0].astype(str) if len(rej) else np.zeros(0, dtype="<U7"),
        rej_tr=rej[:, 1].astype(np.int32) if len(rej) else np.zeros(0, np.int32),
        rej_tc=rej[:, 2].astype(np.int32) if len(rej) else np.zeros(0, np.int32),
        rej_idx=rej[:, 3].astype(np.int32) if len(rej) else np.zeros(0, np.int32),
        rej_x=rej[:, 4].astype(float) if len(rej) else np.zeros(0),
        rej_y=rej[:, 5].astype(float) if len(rej) else np.zeros(0),
        n_tc=np.array([n_tc]), n_tr=np.array([n_tr]),
        raster=np.array([str(E.RASTER)]),
    )
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
