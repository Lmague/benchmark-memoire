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
import os

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
    ap.add_argument("--force-reextract", action="store_true",
                    help="ignore le cache .npy existant et ré-extrait depuis les poids "
                         "(à utiliser si la provenance d'un .npy est douteuse)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.set_seed(cfg.train.seed)
    key = cfg.model.name
    ckpt = cfg.raw.get("checkpoint")
    # Résolution : chemin relatif → relatif à ckpt_dir (local ou colab selon env auto-détecté).
    # On RÉÉCRIT cfg.raw["checkpoint"] pour que extract_features (qui lit cfg.raw) utilise le
    # chemin résolu — sinon le .pth ne serait cherché que relativement au CWD.
    if ckpt and not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg.paths.ckpt_dir, ckpt)
        cfg.raw["checkpoint"] = ckpt
    # --force-reextract : désactive le cache-hit (extract_features ré-extrait et écrase).
    if args.force_reextract:
        cfg.features.cache = False
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
        needs_ckpt = (key.startswith("simdinov2") or key.startswith("satmae")
                      or key.endswith("_arctic"))
        print(f"[dry-run] checkpoint={ckpt!r} force_reextract={args.force_reextract}")
        if needs_ckpt and not ckpt:
            print(f"[dry-run] ⚠ {key} : champ `checkpoint:` vide — à renseigner avant extraction.")
        elif needs_ckpt and not os.path.exists(ckpt):
            print(f"[dry-run] ⚠ {key} : checkpoint {ckpt!r} introuvable (normal en local si "
                  "les poids sont sur Drive — sera résolu à l'exécution sur Colab).")
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
        # Garde-fou split v3 : un n différent signale un .npy issu d'un autre split (cache
        # douteux) → relancer avec --force-reextract. Non bloquant (warning seulement).
        v3_counts = {"train": 49433, "val": 13209, "test": 17598}
        for s, (E, _L) in out.items():
            exp = v3_counts.get(s)
            warn = "" if exp is None or E.shape[0] == exp else \
                f"  ⚠ ATTENDU {exp} tuiles (split ≠ v3 ? → relancer avec --force-reextract)"
            print(f"[extract]   {s}: {E.shape}{warn}")


if __name__ == "__main__":
    main()
