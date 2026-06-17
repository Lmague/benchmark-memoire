#!/usr/bin/env python3
"""CLI génération des figures PNG depuis les JSON de résultats (+ features cachées).

    # figures frozen (barplot F1, scatter anisotropie vs F1) pour la passe with_rhol :
    python make_figures.py --config configs/frozen_eval.yaml
    # + t-SNE (features cachées) :
    python make_figures.py --config configs/frozen_eval.yaml --tsne
    # figures d'un run de fine-tuning (matrice de confusion + courbes) :
    python make_figures.py --config configs/vitb16_mhsa.yaml --training-tag vitb16_mhsa
    # courbes couche-par-couche :
    python make_figures.py --config configs/frozen_eval.yaml --layerwise-model dinov3_vitb16_lvd

Lit le format combiné historique (``<pass>/latent_results.json``) ou les sorties séparées
(``latent_metrics.json`` + ``probe_knn.json``). Écrit dans ``results_dir/figures/``.
"""
from __future__ import annotations

import argparse
import os

from src import figures, utils
from src.config import load_config
from src.utils import load_json


def _load_frozen(results_dir: str, pass_tag: str) -> dict:
    """Charge probe + latent + knn pour la passe demandée.

    Préfère le JSON CANONIQUE ``probe_knn_cgrid.json`` (grille étendue
    C∈{1e-4..10}, F1≈0.4789 pour le co-leader).  L'ancien ``probe_knn.json``
    (grille restreinte, F1≈0.4675 périmé) est marqué .deprecated et n'est
    lu que si le canonique est absent.
    """
    pdir = os.path.join(results_dir, pass_tag)
    data = {"latent_metrics": {}, "probe": {}, "knn": {}}
    combined = os.path.join(pdir, "latent_results.json")
    if os.path.exists(combined):
        d = load_json(combined)
        for k in data:
            data[k] = d.get(k, {})
        return data
    lm = os.path.join(pdir, "latent_metrics.json")
    if os.path.exists(lm):
        data["latent_metrics"] = load_json(lm).get("latent_metrics", {})
    # Canonique d'abord (grille étendue) ; fallback sur l'ancien si nécessaire.
    pk_canon = os.path.join(pdir, "probe_knn_cgrid.json")
    pk_old = os.path.join(pdir, "probe_knn.json")
    pk = pk_canon if os.path.exists(pk_canon) else pk_old
    if os.path.exists(pk):
        d = load_json(pk)
        data["probe"], data["knn"] = d.get("probe", {}), d.get("knn", {})
        if pk == pk_old:
            print(f"[make_figures] WARN: {pk} est PÉRIMÉ (F1≈0.4675, grille restreinte). "
                  f"Utilise probe_knn_cgrid.json (F1≈0.4789) si disponible.")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pass", dest="pass_tag", default="with_rhol",
                    choices=["with_rhol", "without_rhol"])
    ap.add_argument("--training-tag", default=None,
                    help="tag {model}_{regime} pour CM + courbes d'un run de fine-tuning")
    ap.add_argument("--tsne", action="store_true", help="t-SNE depuis les features cachées")
    ap.add_argument("--layerwise-model", default=None, help="courbes couche-par-couche")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.maybe_mount_drive(cfg.env)
    fig_dir = os.path.join(cfg.paths.results_dir, "figures")
    made = []

    # --- figures d'un run de fine-tuning ---
    if args.training_tag:
        rj = os.path.join(cfg.paths.results_dir, f"{args.training_tag}_results.json")
        res = load_json(rj)
        if res.get("confusion_matrix"):
            made.append(figures.plot_confusion_matrix(
                res["confusion_matrix"], res.get("classes", utils.CLASS_NAMES),
                os.path.join(fig_dir, f"{args.training_tag}_confusion.png"),
                title=f"Confusion — {args.training_tag}"))
        if res.get("history"):
            made.append(figures.plot_training_curves(
                res["history"], os.path.join(fig_dir, f"{args.training_tag}_curves.png"),
                title=args.training_tag))

    # --- figures frozen (probe + latent) ---
    fr = _load_frozen(cfg.paths.results_dir, args.pass_tag)
    probe, lat = fr["probe"], fr["latent_metrics"]
    if probe:
        f1 = {k: v["test"]["f1_macro_all"] for k, v in probe.items()}
        made.append(figures.plot_f1_barplot(
            f1, os.path.join(fig_dir, f"f1_barplot_{args.pass_tag}.png"),
            title=f"Probe F1-Macro (test, {args.pass_tag})"))
        if lat:
            pts = [(k, lat[k]["anisotropy"], probe[k]["test"]["f1_macro_pres"])
                   for k in probe if k in lat]
            made.append(figures.plot_anisotropy_vs_f1(
                pts, os.path.join(fig_dir, f"anisotropy_vs_f1_{args.pass_tag}.png"),
                title=f"Anisotropie vs F1 ({args.pass_tag})"))

    # --- courbes couche-par-couche ---
    if args.layerwise_model:
        lj = os.path.join(cfg.paths.results_dir, "layerwise", f"{args.layerwise_model}_layerwise.json")
        d = load_json(lj)
        made.append(figures.plot_layerwise_curves(
            d["curves"], os.path.join(fig_dir, f"{args.layerwise_model}_layerwise.png"),
            model=args.layerwise_model))

    # --- t-SNE (features cachées) ---
    if args.tsne:
        from src.features import load_features
        keys = list(cfg.models) + (list(cfg.finetuned_models) if cfg.probe.include_finetuned else [])
        feats_test = {}
        for k in keys:
            try:
                feats_test[k] = load_features(cfg, k)["test"]
            except Exception as e:  # noqa: BLE001
                print(f"[tsne] skip {k}: {e}")
        subt = {k: f"F1m={probe[k]['test']['f1_macro_pres']:.3f}" for k in feats_test if k in probe}
        made.append(figures.plot_tsne(
            feats_test, os.path.join(fig_dir, f"tsne_{args.pass_tag}.png"),
            subtitles=subt or None))

    if made:
        print("[make_figures] PNG écrits :")
        for p in made:
            print("   ", p)
    else:
        print("[make_figures] aucune figure produite — vérifier les JSON de résultats.")


if __name__ == "__main__":
    main()
