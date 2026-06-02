#!/usr/bin/env python3
"""CLI analyse de l'espace latent : RankMe + anisotropie (+ couche-par-couche).

    python analyze.py --config configs/frozen_eval.yaml
    python analyze.py --config configs/frozen_eval.yaml --layerwise --layerwise-model dinov3_vitb16_lvd

Deux passes (with/without RHOL). Écrit ``results_dir/<pass>/latent_metrics.json`` et,
si ``--layerwise``, ``results_dir/layerwise/<model>_layerwise.json``.
"""
from __future__ import annotations

import argparse
import os

from src import utils
from src.config import load_config
from src.features import load_features, load_layerwise
from src.latent import drop_class, layerwise_curves, metrics_on


def _model_keys(cfg):
    keys = list(cfg.models)
    if cfg.probe.include_finetuned:
        keys += list(cfg.finetuned_models)
    return keys


def _infer_n_layers(model_key: str) -> int:
    """Nb de blocs transformer selon l'architecture (ViT-B/16=12, ViT-L/16=24).

    Évite le défaut figé à 12 qui tronquait les ViT-L/16 (DINOv3 L/16, SimDINOv2 L/16).
    Passer ``--n-layers`` explicitement court-circuite cette inférence.
    """
    k = model_key.lower()
    if "vitl" in k or "vit_l" in k or "vitl16" in k:
        return 24
    if "vitb" in k or "vit_b" in k or "vitb16" in k:
        return 12
    raise ValueError(
        f"Nombre de blocs indéterminé pour '{model_key}' : passez --n-layers explicitement "
        "(ViT-B/16=12, ViT-L/16=24).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layerwise", action="store_true",
                    help="courbes RankMe/anisotropie couche-par-couche (features layerwise requises)")
    ap.add_argument("--layerwise-model", default=None, help="modèle pour les courbes (défaut: 1er)")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="nb de blocs (défaut: inféré du modèle — ViT-B/16=12, ViT-L/16=24)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.maybe_mount_drive(cfg.env)
    keys = _model_keys(cfg)
    rk_split, an_split = cfg.latent.rankme_split, cfg.latent.anisotropy_split

    for tag, drop, names in utils.rhol_passes():
        res = {}
        for key in keys:
            feats = load_features(cfg, key)
            if drop is not None:
                feats = {s: drop_class(*feats[s], drop) for s in feats}
            res[key] = metrics_on(feats[rk_split][0], feats[an_split][0],
                                  n_pairs=cfg.latent.n_pairs,
                                  subsample_n=cfg.latent.rankme_subsample)
            m = res[key]
            print(f"[analyze:{tag}] {key:24s} dim={m['dim']:<5} "
                  f"RankMe={m['rankme']:>8.2f} Aniso={m['anisotropy']:+.4f}")
        out = os.path.join(cfg.paths.results_dir, tag, "latent_metrics.json")
        utils.save_json({"latent_metrics": res}, out)
        print(f"[analyze:{tag}] -> {out}")

    if args.layerwise:
        key = args.layerwise_model or keys[0]
        n_layers = args.n_layers if args.n_layers is not None else _infer_n_layers(key)
        feats = load_layerwise(cfg, key, rk_split, n_layers)
        curves = layerwise_curves(feats, n_pairs=cfg.latent.n_pairs,
                                  subsample_n=cfg.latent.rankme_subsample)
        out = os.path.join(cfg.paths.results_dir, "layerwise", f"{key}_layerwise.json")
        utils.save_json({"model": key, "split": rk_split, "curves": curves}, out)
        print(f"[analyze] couche-par-couche {key} ({len(curves)} couches) -> {out}")


if __name__ == "__main__":
    main()
