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
from src.features import extract_features, extract_layerwise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML frozen_* (configs/...)")
    ap.add_argument("--layerwise", action="store_true",
                    help="extraire la représentation CLS de chaque bloc transformer")
    ap.add_argument("--dry-run", action="store_true",
                    help="résout la config et affiche le plan sans extraire (ni clone, ni .pth)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.set_seed(cfg.train.seed)
    key = cfg.model.name
    ckpt = cfg.raw.get("checkpoint")
    print(f"[extract] modèle={key} env={cfg.env} emb_dir={cfg.paths.emb_dir} "
          f"norm={cfg.model.norm} layerwise={args.layerwise or cfg.features.layerwise}")

    if args.dry_run:
        needs_ckpt = key.startswith("simdinov2")
        print(f"[dry-run] checkpoint={ckpt!r}")
        if needs_ckpt and not ckpt:
            print("[dry-run] ⚠ SimDINOv2 : champ `checkpoint:` vide — à renseigner avant extraction.")
        print("[dry-run] OK — config résolue, extraction NON lancée.")
        return

    utils.maybe_mount_drive(cfg.env)
    print(f"[extract] démarrage extraction {key}")

    if args.layerwise or cfg.features.layerwise:
        n = extract_layerwise(cfg, key)
        print(f"[extract] couche-par-couche : {n} couches cachées pour {key}")
    else:
        out = extract_features(cfg, key)
        for s, (E, _L) in out.items():
            print(f"[extract]   {s}: {E.shape}")


if __name__ == "__main__":
    main()
