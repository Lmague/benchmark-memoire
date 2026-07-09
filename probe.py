#!/usr/bin/env python3
"""CLI linear probe + k-NN sur features cachées.

    python probe.py --config configs/frozen_eval.yaml
    python probe.py --config configs/frozen_eval.yaml --only dinov3_vitb16_lvd

Deux passes (with/without RHOL). Écrit ``results_dir/<pass>/probe_knn.json``.
"""
from __future__ import annotations

import argparse
import json
import os

from src import utils
from src.config import load_config
from src.features import load_features
from src.latent import drop_class
from src.probe import knn, linear_probe


def _sota_keys(cfg):
    """Énumère les runs SOTA présents sous ``cfg.paths.sota_dir``.

    Grille : ``sota_regimes × sota_fractions × sota_seeds`` (config probe). Ne retient
    que les runs dont le dossier d'embeddings existe (un run manquant → SKIP + warning,
    pour ne pas faire échouer tout le corpus sur un run absent/incomplet)."""
    keys = []
    for regime in cfg.probe.sota_regimes:
        for frac in cfg.probe.sota_fractions:
            for seed in cfg.probe.sota_seeds:
                key = f"vitb16_{regime}_frac{frac}_seed{seed}"
                run_dir = os.path.join(cfg.paths.sota_dir, regime, "embeddings", key)
                if os.path.isdir(run_dir):
                    keys.append(key)
                else:
                    print(f"[probe] run SOTA absent → SKIP : {run_dir}")
    return keys


def _model_keys(cfg, only, include_sota=False):
    if only:
        return list(only)
    keys = list(cfg.models)
    if cfg.probe.include_finetuned:
        keys += list(cfg.finetuned_models)
    if include_sota or cfg.probe.include_sota:
        keys += _sota_keys(cfg)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="restreindre à certains modèles")
    ap.add_argument("--output-tag", default=None,
                    help="suffixe du fichier de sortie : probe_knn_T.json (défaut: probe_knn.json)")
    ap.add_argument("--force", action="store_true",
                    help="re-calcule même si l'output existe déjà (défaut: skip si présent)")
    ap.add_argument("--merge", action="store_true",
                    help="fusionne dans le JSON existant (met à jour les modèles calculés, "
                         "garde les autres) au lieu d'écraser — pour ajouter/recalculer un sous-ensemble")
    ap.add_argument("--include-sota", action="store_true",
                    help="ajoute les runs SOTA (vitb16_{regime}_frac{XXX}_seed{N}) énumérés "
                         "sous cfg.paths.sota_dir (schéma 11cls_no_rhol)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.maybe_mount_drive(cfg.env)
    keys = _model_keys(cfg, args.only, include_sota=args.include_sota)
    print(f"[probe] {len(keys)} modèles={keys}")

    # Passe-major : pour chaque passe, chaque modèle est réduit selon SON schéma source
    # (12cls canonique / 11cls SOTA) via utils.pass_drops. Les deux sources aboutissent au
    # MÊME référentiel de classes par passe (mêmes noms/indices finaux) → coexistence dans
    # le même JSON de sortie. with_rhol n'est pas applicable aux runs SOTA (RHOL déjà absent).
    for tag, names in utils.probe_passes():
        n_classes = len(names)
        fname = f"probe_knn_{args.output_tag}.json" if args.output_tag else "probe_knn.json"
        out = os.path.join(cfg.paths.results_dir, tag, fname)
        # Skip si l'output existe déjà et pas --force (idempotence Makefile)
        if not args.force and os.path.exists(out):
            print(f"[probe:{tag}] {out} existe déjà → SKIP (utiliser --force pour re-calculer).")
            continue
        probe_res, knn_res = {}, {}
        for key in keys:
            schema = utils.source_schema(key)
            drops = utils.pass_drops(tag, schema)
            if drops is None:
                print(f"[probe:{tag}] {key:24s} passe non applicable (source {schema}) → SKIP")
                continue
            feats = load_features(cfg, key)
            for d in drops:  # cascade DÉCROISSANTE (ordre imposé par pass_drops)
                feats = {s: drop_class(*feats[s], d) for s in feats}
            probe_res[key] = linear_probe(feats, n_classes, names,
                                          tuple(cfg.probe.C_grid), cfg.probe.max_iter,
                                          cfg.train.seed, cfg.probe.selection_metric)
            knn_res[key] = knn(feats, n_classes, tuple(cfg.probe.knn_k))
            p = probe_res[key]["test"]
            print(f"[probe:{tag}] {key:28s} C={probe_res[key]['best_C']} "
                  f"(sel={cfg.probe.selection_metric}) "
                  f"testF1m(all)={p['f1_macro_all']:.4f} acc={p['accuracy']:.4f}")
        if not probe_res and not knn_res:
            print(f"[probe:{tag}] aucun modèle applicable → pas d'écriture.")
            continue
        if args.merge and os.path.exists(out):
            with open(out) as f:
                prev = json.load(f)
            prev.setdefault("probe", {}).update(probe_res)
            prev.setdefault("knn", {}).update(knn_res)
            payload = prev
            print(f"[probe:{tag}] fusion : {list(probe_res)} dans {len(prev['probe'])} modèles existants")
        else:
            payload = {"probe": probe_res, "knn": knn_res}
        utils.save_json(payload, out)
        print(f"[probe:{tag}] -> {out}")


if __name__ == "__main__":
    main()
