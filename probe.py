#!/usr/bin/env python3
"""CLI linear probe + k-NN sur features cachées.

    python probe.py --config configs/frozen_eval.yaml
    python probe.py --config configs/frozen_eval.yaml --only dinov3_vitb16_lvd

Deux passes (with/without RHOL). Écrit ``results_dir/<pass>/probe_knn.json``.
"""
from __future__ import annotations

import argparse
import os

from src import utils
from src.config import load_config
from src.features import load_features
from src.latent import drop_class
from src.probe import knn, linear_probe


def _model_keys(cfg, only):
    if only:
        return list(only)
    keys = list(cfg.models)
    if cfg.probe.include_finetuned:
        keys += list(cfg.finetuned_models)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restreindre à certains modèles")
    ap.add_argument("--output-tag", default=None,
                    help="suffixe du fichier de sortie : probe_knn_T.json (défaut: probe_knn.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.maybe_mount_drive(cfg.env)
    keys = _model_keys(cfg, args.only)
    print(f"[probe] modèles={keys}")

    for tag, drop, names in utils.rhol_passes():
        n_classes = len(names)
        probe_res, knn_res = {}, {}
        for key in keys:
            feats = load_features(cfg, key)
            if drop is not None:
                feats = {s: drop_class(*feats[s], drop) for s in feats}
            probe_res[key] = linear_probe(feats, n_classes, names,
                                          tuple(cfg.probe.C_grid), cfg.probe.max_iter,
                                          cfg.train.seed, cfg.probe.selection_metric)
            knn_res[key] = knn(feats, n_classes, tuple(cfg.probe.knn_k))
            p = probe_res[key]["test"]
            print(f"[probe:{tag}] {key:24s} C={probe_res[key]['best_C']} "
                  f"(sel={cfg.probe.selection_metric}) "
                  f"testF1m(all)={p['f1_macro_all']:.4f} acc={p['accuracy']:.4f}")
        fname = f"probe_knn_{args.output_tag}.json" if args.output_tag else "probe_knn.json"
        out = os.path.join(cfg.paths.results_dir, tag, fname)
        utils.save_json({"probe": probe_res, "knn": knn_res}, out)
        print(f"[probe:{tag}] -> {out}")


if __name__ == "__main__":
    main()
