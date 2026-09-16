#!/usr/bin/env python3
"""Extraction des features de texture par tuile (GLCM/GLRLM/GLSZM/GLDM/NGTDM + DWT + Fourier).

── Pourquoi ce script existe ────────────────────────────────────────────────────────
Arctic-TVC est à 2,2 mm/px : le patch d'un ViT (16 px) couvre 3,5 cm, donc la texture fine
(moucheture du lichen, tapis de mousse, linaires de linaigrette) est lissée avant d'entrer
dans le réseau. Les descripteurs de texture classiques sont l'outil standard de la
télédétection de végétation pour cette échelle (Kulich et al. 2026,
doi:10.3389/fpls.2026.1841696 ; Deng et al. 2022, doi:10.1038/s41598-022-17620-2).

Ces features sont destinées à être CONCATÉNÉES à l'embedding avant la sonde
(scripts/texture_ablation.py). Elles ne doivent JAMAIS être empilées en canaux d'entrée
d'un ViT gelé : le patch embedding attend 3 canaux normalisés, et une pile 224×224×K
casserait les filtres pré-entraînés (cf. DEFLECT/UPE, arXiv:2504.17397, pour la seule
façon correcte de faire entrer des canaux supplémentaires dans un ViT).

── Repartabilité ────────────────────────────────────────────────────────────────────
Le calcul se fait par CHUNKS. Chaque chunk est écrit dans ``<out>.partNNNN.npy`` et un
redémarrage saute les chunks déjà présents (même convention « skip-if-done, repartable »
que scripts/context_crop.py). Un job SLURM interrompu reprend où il s'était arrêté. Les
parts sont fusionnées puis supprimées à la fin (``--keep-parts`` pour les garder).

── Coût ─────────────────────────────────────────────────────────────────────────────
~20–25 ms/tuile en mono-processus (mesuré sur 224×224). Train+val+test = 80 088 tuiles
⇒ ~30 min mono-cœur, quelques minutes avec ``--n-jobs 8``.

── Usage ────────────────────────────────────────────────────────────────────────────
    # Sur Narval / Colab, là où les tuiles existent (ordre = celui des CSV de split)
    python3 scripts/texture_features.py --split train --n-jobs 8
    python3 scripts/texture_features.py --split val   --n-jobs 8
    python3 scripts/texture_features.py --split test  --n-jobs 8

    # Smoke test local (les tuiles natives sont absentes de la machine de dev)
    python3 scripts/texture_features.py --split test --tiles-dir out/context/context_512 \
        --out /tmp/tex_test.npy --limit 200

Sorties : ``<out>.npy`` (float32, n_tuiles × n_features) + ``<out>.json`` (contrat de
colonnes, paramètres, provenance, empreinte du CSV).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))


# ── paramètres partagés avec les workers (initialisés une fois par processus) ─────────
_W: dict = {}


def _init_worker(tiles_root: str, fallback: str, params: dict, feature_names: list[str],
                 strict: bool = False) -> None:
    from PIL import Image  # noqa: F401  (importé ici : un seul import par worker)
    _W["tiles_root"] = tiles_root
    _W["fallback"] = fallback
    _W["params"] = params
    _W["names"] = feature_names
    _W["strict"] = strict
    from src.texture import tile_features  # noqa: F401
    _W["fn"] = tile_features


def _extract_one(filepath: str) -> np.ndarray:
    """Features d'une tuile, dans l'ORDRE DU CONTRAT (jamais celui du dict interne).

    Une tuile manquante ou illisible renvoie une ligne NaN au lieu de lever : sur 80 000
    fichiers lus depuis un zip de cluster, un PNG corrompu ne doit pas coûter 8 h de GPU.
    Le compte de NaN et la liste des fichiers fautifs sont écrits dans le JSON de
    provenance, et ``--strict`` rétablit le comportement levant.
    """
    from PIL import Image
    from src.texture import to_gray
    root = _W["tiles_root"]
    path = os.path.join(root, filepath)
    if not os.path.exists(path) and _W["fallback"]:
        path = os.path.join(_W["fallback"], filepath)
    with Image.open(path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.uint8)
    feats = _W["fn"](to_gray(arr), **_W["params"])
    return np.asarray([feats[n] for n in _W["names"]], dtype=np.float32)


def _extract_safe(filepath: str) -> tuple[np.ndarray | None, str]:
    """Enveloppe : ``(features, "")`` ou ``(None, message_d_erreur)``."""
    try:
        return _extract_one(filepath), ""
    except Exception as exc:                                   # noqa: BLE001
        if _W.get("strict"):
            raise
        return None, f"{filepath}\t{type(exc).__name__}: {exc}"


def _read_split(csv_path: str, limit: int | None) -> list[str]:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[:limit]
    return [r["filepath"] for r in rows]


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--csv", default=None,
                    help="CSV du split (défaut : splits/<split>_11cls.csv, ordre des embeddings)")
    ap.add_argument("--tiles-dir", default="out/tiles",
                    help="racine des tuiles (défaut out/tiles ; sur Narval $SLURM_TMPDIR/tiles)")
    ap.add_argument("--out", default=None, help="chemin .npy (défaut results/texture/<split>.npy)")
    ap.add_argument("--levels", type=int, default=16, help="niveaux de quantification (défaut 16)")
    ap.add_argument("--distances", default="1,2,4", help="distances GLCM en px (défaut 1,2,4)")
    ap.add_argument("--no-dwt", action="store_true")
    ap.add_argument("--no-fourier", action="store_true")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--dwt-level", type=int, default=3)
    ap.add_argument("--gldm-alpha", type=int, default=1)
    ap.add_argument("--gldm-delta", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=1, help="processus (PAS de threads BLAS)")
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que les N premières tuiles")
    ap.add_argument("--force", action="store_true", help="recalculer même si la sortie existe")
    ap.add_argument("--keep-parts", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="lever sur une tuile manquante/illisible (défaut : ligne NaN + "
                         "fichier listé dans le JSON de provenance)")
    args = ap.parse_args()

    csv_path = args.csv or str(PROJ / "splits" / f"{args.split}_11cls.csv")
    out_base = Path(args.out) if args.out else (PROJ / "results" / "texture" / f"{args.split}.npy")
    out_npy = out_base if out_base.suffix == ".npy" else out_base.with_suffix(".npy")
    out_json = out_npy.with_suffix(".json")
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    params = dict(levels=args.levels, distances=distances,
                  with_dwt=not args.no_dwt, with_fourier=not args.no_fourier,
                  gldm_alpha=args.gldm_alpha, gldm_delta=args.gldm_delta,
                  wavelet=args.wavelet, dwt_level=args.dwt_level)
    # Copie JSON-safe : JSON n'a pas de tuples, donc comparer directement ``params``
    # (qui contient ``distances`` en tuple) à ce qui a été relu échouerait toujours et
    # la reprise recalculerait tout.
    params_json = json.loads(json.dumps(params))

    from src.texture import expected_feature_names
    names = list(expected_feature_names(**params))
    filepaths = _read_split(csv_path, args.limit)
    n = len(filepaths)
    print(f"[texture] split={args.split}  tuiles={n}  features={len(names)}  "
          f"csv={csv_path}", flush=True)
    print(f"[texture] tiles_dir={args.tiles_dir}  out={out_npy}", flush=True)

    # ── CSV de provenance : empreinte + chemin, pour détecter un désalignement futur
    provenance = dict(split=args.split, csv=csv_path, csv_sha1=_sha1(csv_path),
                      n_tiles=n, tiles_dir=args.tiles_dir, feature_names=names,
                      params=params_json, limit=args.limit)

    if out_npy.exists() and out_json.exists() and not args.force:
        prev = json.loads(out_json.read_text())
        if (prev.get("n_tiles") == n and prev.get("csv_sha1") == provenance["csv_sha1"]
                and prev.get("feature_names") == names and prev.get("params") == params_json):
            print("[texture] sortie déjà complète et cohérente → rien à faire "
                  "(utiliser --force pour recalculer)", flush=True)
            return 0
        print("[texture] sortie présente mais incohérente (CSV/params/noms ont changé) → recalcul",
              flush=True)

    # ── calcul par chunks, repartable
    chunks = [(i, min(i + args.chunk_size, n)) for i in range(0, n, args.chunk_size)]
    t_start = time.time()
    failures: list[str] = []
    init = (args.tiles_dir, os.environ.get("ARCTIC_TILES_FALLBACK", ""), params, names,
            args.strict)
    for k, (lo, hi) in enumerate(chunks, 1):
        part = out_npy.with_name(f"{out_npy.stem}.part{k - 1:04d}.npy")
        if part.exists() and part.with_suffix(".done").exists() and not args.force:
            continue
        sub = filepaths[lo:hi]
        t0 = time.time()
        if args.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker,
                                     initargs=init) as ex:
                res = list(ex.map(_extract_safe, sub, chunksize=8))
        else:
            _init_worker(*init)
            res = [_extract_safe(fp) for fp in sub]
        rows = []
        for fp, (arr, err) in zip(sub, res):
            if arr is None:
                rows.append(np.full(len(names), np.nan, dtype=np.float32))
                failures.append(err)
            else:
                rows.append(arr)
        arr = np.stack(rows).astype(np.float32)
        np.save(part, arr)
        part.with_suffix(".done").touch()
        rate = (hi - lo) / max(time.time() - t0, 1e-9)
        suffix = f"  ({len(failures)} tuile(s) illisible(s) depuis le début)" if failures else ""
        print(f"  chunk {k}/{len(chunks)}  [{lo}:{hi}]  {arr.shape}  "
              f"{rate:.0f} tuiles/s  (total {time.time() - t_start:.0f}s){suffix}", flush=True)

    parts = [out_npy.with_name(f"{out_npy.stem}.part{k - 1:04d}.npy") for k in range(1, len(chunks) + 1)]
    missing = [p for p in parts if not p.exists()]
    if missing:
        print(f"[texture] ERREUR : {len(missing)} chunks manquants — relancer le script", file=sys.stderr)
        return 1
    total = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in parts)
    if total != n:
        print(f"[texture] ERREUR : {total} lignes assemblées pour {n} attendues", file=sys.stderr)
        return 1
    X = np.empty((n, len(names)), dtype=np.float32)
    pos = 0
    for p in parts:
        a = np.load(p, mmap_mode="r")
        X[pos:pos + a.shape[0]] = a
        pos += a.shape[0]
    np.save(out_npy, X)
    provenance["shape"] = list(X.shape)
    provenance["n_non_finis"] = int((~np.isfinite(X)).sum())
    provenance["n_tuiles_illisibles"] = len(failures)
    provenance["tuiles_illisibles"] = failures[:200]
    provenance["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_json.write_text(json.dumps(provenance, ensure_ascii=False, indent=1))
    if not args.keep_parts:
        for p in parts:
            p.unlink(missing_ok=True)
            p.with_suffix(".done").unlink(missing_ok=True)
    print(f"[texture] OK → {out_npy}  {X.shape}  en {time.time() - t_start:.0f}s", flush=True)
    if provenance["n_non_finis"]:
        print(f"[texture] ATTENTION : {provenance['n_non_finis']} valeurs non finies "
              f"({len(failures)} tuile(s) illisible(s), liste dans {out_json.name}) — "
              f"la sonde REFUSERA ce cache (StandardScaler propagerait les NaN).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
