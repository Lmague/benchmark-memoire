#!/usr/bin/env python3
"""CLI extraction des features frozen (+ couche-par-couche).

    python extract.py --config configs/frozen_dinov3_lvd.yaml
    python extract.py --config configs/frozen_dinov3_lvd.yaml --layerwise
    python extract.py --config configs/frozen_simdinov2_vitb16.yaml --dry-run

Le modèle extrait = ``cfg.model.name`` (clé d'extracteur frozen). Cache fp16 dans
``cfg.paths.emb_dir``, réutilisable par probe / k-NN / analyse latente.
"""
from __future__ import annotations

import argparse

from src import utils
from src.config import load_config
from src.features import SPLITS, extract_features, extract_layerwise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML frozen_* (configs/...)")
    ap.add_argument("--layerwise", action="store_true",
                    help="extraire la représentation CLS de chaque bloc transformer")
    ap.add_argument("--splits", nargs="+", default=None, metavar="SPLIT",
                    help="splits à extraire (défaut: train val test ; layerwise: test seul)")
    ap.add_argument("--dry-run", action="store_true",
                    help="résout la config et affiche le plan sans extraire (ni clone, ni .pth)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.set_seed(cfg.train.seed)
    key = cfg.model.name
    ckpt = cfg.raw.get("checkpoint")
    layerwise = args.layerwise or cfg.features.layerwise

    # Résolution des splits : --splits prime ; sinon test-seul en layerwise, sinon les 3.
    # L'extraction frozen normale (sans --layerwise, sans --splits) reste sur train/val/test.
    if args.splits is not None:
        splits = tuple(args.splits)
        bad = [s for s in splits if s not in SPLITS]
        if bad:
            ap.error(f"--splits invalide(s) {bad} (attendu parmi {list(SPLITS)})")
    elif layerwise:
        splits = ("test",)
    else:
        splits = SPLITS

    print(f"[extract] modèle={key} env={cfg.env} emb_dir={cfg.paths.emb_dir} "
          f"norm={cfg.model.norm} layerwise={layerwise} splits={list(splits)}")

    if args.dry_run:
        needs_ckpt = key.startswith("simdinov2")
        print(f"[dry-run] checkpoint={ckpt!r}")
        if needs_ckpt and not ckpt:
            print("[dry-run] ⚠ SimDINOv2 : champ `checkpoint:` vide — à renseigner avant extraction.")
        print(f"[dry-run] plan : {'layerwise CLS/bloc' if layerwise else 'frozen CLS/pooled'} "
              f"sur splits={list(splits)} — extraction NON lancée.")
        return

    utils.maybe_mount_drive(cfg.env)
    print(f"[extract] démarrage extraction {key}")

    if layerwise:
        n = extract_layerwise(cfg, key, splits=splits)
        print(f"[extract] couche-par-couche : {n} couches cachées pour {key}")
    else:
        out = extract_features(cfg, key, splits=splits)
        for s, (E, _L) in out.items():
            print(f"[extract]   {s}: {E.shape}")


if __name__ == "__main__":
    main()
