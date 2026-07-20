#!/usr/bin/env python3
"""Linear probe F1 *par couche* + géométrie par couche + CKA inter-couches.

``analyze.py --layerwise`` ne calcule que la GÉOMÉTRIE par couche (RankMe/anisotropie).
Ici on ajoute, pour chaque bloc transformer :
  - le **F1 du linear probe** (sonde entraînée sur la représentation CLS de ce bloc) ;
  - le **gain de F1 vs le bloc précédent** (``f1_gain_from_prev``) — où l'information
    utile à la tâche apparaît le long du réseau ;
  - la **CKA linéaire vs le bloc précédent** (``cka_with_prev``) — mesure de changement
    représentationnel : CKA basse = le bloc transforme fortement la représentation ;
    CKA haute = le bloc ne fait que propager (candidat à contraindre plus fort / geler).

Objectif diagnostic (au-delà de la corrélation géométrie↔F1) : identifier EMPIRIQUEMENT
quels blocs portent l'adaptation, pour informer le choix de ``full_ft_block_indices`` en
ExPLoRA (p.ex. tester si le "profil en U" — extrémités actives, milieu inerte — est réel
et propre à un backbone, ou général aux deux SSL). Le diagnostic se fait sur le modèle
FROZEN : il mesure où l'info est déjà utile / où la représentation change, PAS directement
le gain qu'apporterait le fine-tuning (proxy, pas mesure directe).

Prérequis : embeddings layerwise train/val/test extraits via
``extract.py --layerwise --splits train val test`` pour chaque modèle (ViT uniquement ;
les MAE ne sont pas pertinents en CLS — à exclure).

Usage :
  python scripts/layerwise_probe.py \\
      --config configs/frozen_eval.yaml \\
      --models dinov3_vitb16_lvd simdinov2_vitb16 \\
      --output results/layerwise_probe.json \\
      --fig results/figures/layerwise_probe_pooled.png \\
      --fig-depth results/figures/layerwise_depth_profile.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import count_layerwise_layers, load_layerwise
from src.latent import anisotropy, rankme, subsample
from src.probe import linear_probe
from src.utils import CLASS_NAMES, N_CLASSES

DEFAULT_MODELS = ["dinov3_vitl16_lvd", "dinov3_vitl16_sat", "dinov3_vitb16_lvd",
                  "simdinov2_vitl16", "simdinov2_vitb16", "vitb16_imagenet"]


def _spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def _kendall(x, y):
    from scipy.stats import kendalltau
    return float(kendalltau(x, y)[0])


def linear_cka(X, Y):
    """CKA linéaire (Kornblith et al. 2019) entre deux matrices de représentation
    aux MÊMES lignes (échantillons), dimensions de features éventuellement différentes.

    Retourne une similarité dans [0, 1] : proche de 1 = représentations quasi identiques
    (bloc "redondant") ; proche de 0 = fort changement représentationnel.

    Formulation feature-space : HSIC_lin(X, Y) = ||X^T Y||_F^2 ; efficace quand d << n.
    Les deux matrices doivent être alignées ligne-à-ligne (mêmes tuiles dans le même ordre).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = float(((X.T @ Y) ** 2).sum())
    hsic_xx = float(((X.T @ X) ** 2).sum())
    hsic_yy = float(((Y.T @ Y) ** 2).sum())
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else float("nan")


def run_model(cfg, key: str, subsample_n: int = 20000) -> dict:
    n_layers = count_layerwise_layers(cfg, key, "train")
    lw = {s: load_layerwise(cfg, key, s, n_layers) for s in ("train", "val", "test")}
    C_grid = tuple(cfg.probe.C_grid)

    # Indices de sous-échantillonnage PARTAGÉS entre couches, réservés à la CKA :
    # la CKA exige des lignes alignées (mêmes tuiles test, même ordre) d'une couche à
    # l'autre. Le test set est fixe → un seul tirage d'indices, réutilisé à chaque bloc.
    # (La géométrie RankMe/aniso conserve, elle, l'appel subsample() historique inchangé,
    #  pour reproduire bit-à-bit les valeurs des runs déjà effectués.)
    n_test = int(np.asarray(lw["test"][0][0]).shape[0])
    _rng = np.random.RandomState(cfg.train.seed)
    if n_test > subsample_n:
        cka_idx = _rng.choice(n_test, subsample_n, replace=False)
    else:
        cka_idx = np.arange(n_test)

    layers = {}
    prev_cka_emb = None
    prev_f1 = None
    for li in range(n_layers):
        feats = {"train": lw["train"][li], "val": lw["val"][li], "test": lw["test"][li]}
        res = linear_probe(feats, N_CLASSES, CLASS_NAMES, C_grid,
                           cfg.probe.max_iter, cfg.train.seed, cfg.probe.selection_metric)
        Ete = np.asarray(lw["test"][li][0], np.float32)
        Es = subsample(Ete, subsample_n, cfg.train.seed)   # inchangé : géométrie
        rm = rankme(Es)
        dim = int(Ete.shape[1])
        f1 = res["test"]["f1_macro_pres"]

        # Changement représentationnel vs bloc précédent (lignes alignées via cka_idx).
        cka_emb = Ete[cka_idx]
        cka_prev = linear_cka(prev_cka_emb, cka_emb) if prev_cka_emb is not None else None
        f1_gain = (f1 - prev_f1) if prev_f1 is not None else None

        layers[li] = {
            "f1_macro_pres": f1,
            "f1_macro_all": res["test"]["f1_macro_all"],
            "best_C": res["best_C"],
            "rankme": rm,
            "rankme_normalized": rm / dim,
            "anisotropy": anisotropy(Es, seed=cfg.train.seed),
            "dim": dim,
            "cka_with_prev": cka_prev,
            "f1_gain_from_prev": f1_gain,
        }
        prev_cka_emb = cka_emb
        prev_f1 = f1

        msg = (f"[layerwise-probe] {key:20} L{li:02d}  F1={f1:.4f}  "
               f"RankMe={rm:7.1f}  Aniso={layers[li]['anisotropy']:+.3f}")
        if cka_prev is not None:
            msg += f"  CKA(prev)={cka_prev:.3f}  ΔF1={f1_gain:+.4f}"
        print(msg)

    # corrélation intra-modèle (sur les couches)
    f1s = [layers[li]["f1_macro_pres"] for li in range(n_layers)]
    rmn = [layers[li]["rankme_normalized"] for li in range(n_layers)]
    ani = [layers[li]["anisotropy"] for li in range(n_layers)]
    within = {
        "n_layers": n_layers,
        "spearman_rankme_norm_vs_f1": _spearman(rmn, f1s),
        "spearman_anisotropy_vs_f1": _spearman(ani, f1s),
    }
    # Résumés diagnostiques : où la représentation change le plus, où le F1 saute le plus.
    cka_pairs = [(li, layers[li]["cka_with_prev"]) for li in range(1, n_layers)
                 if layers[li]["cka_with_prev"] is not None]
    gain_pairs = [(li, layers[li]["f1_gain_from_prev"]) for li in range(1, n_layers)
                  if layers[li]["f1_gain_from_prev"] is not None]
    if cka_pairs:
        li_min = min(cka_pairs, key=lambda t: t[1])
        within["min_cka_transition"] = {"to_layer": li_min[0], "cka": li_min[1]}
    if gain_pairs:
        li_max = max(gain_pairs, key=lambda t: t[1])
        within["max_f1_gain_layer"] = {"to_layer": li_max[0], "f1_gain": li_max[1]}

    return {"n_layers": n_layers, "layers": layers, "within_model_correlation": within}


def make_figure(results: dict, fig_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, r in results.items():
        xs = [r["layers"][li]["rankme_normalized"] for li in range(r["n_layers"])]
        ys = [r["layers"][li]["f1_macro_pres"] for li in range(r["n_layers"])]
        ax.plot(xs, ys, marker="o", ms=3, linewidth=1.0, alpha=0.8, label=key)
    ax.set_xlabel("RankMe normalisé (par couche)", fontsize=10)
    ax.set_ylabel("f1_macro_pres du probe (par couche)", fontsize=10)
    ax.set_title("Géométrie ↔ F1, intra-modèle couche-par-couche", fontsize=11,
                 fontweight="bold")
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {fig_path}")


def make_depth_figure(results: dict, fig_path: str) -> None:
    """Profil en profondeur : F1, anisotropie et CKA(consécutive) vs index de couche.

    C'est LA figure qui révèle un éventuel "profil en U" (extrémités vs milieu) et
    localise les transitions à fort changement représentationnel — support direct au
    choix des blocs à dégeler.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax_f1, ax_ani, ax_cka = axes

    for key, r in results.items():
        n = r["n_layers"]
        xs = list(range(n))
        f1s = [r["layers"][li]["f1_macro_pres"] for li in xs]
        ani = [r["layers"][li]["anisotropy"] for li in xs]
        ax_f1.plot(xs, f1s, marker="o", ms=3, linewidth=1.0, alpha=0.85, label=key)
        ax_ani.plot(xs, ani, marker="o", ms=3, linewidth=1.0, alpha=0.85, label=key)
        # CKA définie pour li >= 1 (transition li-1 → li), tracée à l'abscisse li.
        cka_x = [li for li in range(1, n) if r["layers"][li]["cka_with_prev"] is not None]
        cka_y = [r["layers"][li]["cka_with_prev"] for li in cka_x]
        ax_cka.plot(cka_x, cka_y, marker="o", ms=3, linewidth=1.0, alpha=0.85, label=key)

    ax_f1.set_ylabel("F1 probe (par couche)", fontsize=10)
    ax_f1.set_title("Profil en profondeur — diagnostic pour le choix des blocs",
                    fontsize=11, fontweight="bold")
    ax_ani.set_ylabel("Anisotropie", fontsize=10)
    ax_cka.set_ylabel("CKA vs couche préc.\n(bas = fort changement)", fontsize=10)
    ax_cka.set_xlabel("Index de couche (bloc transformer)", fontsize=10)
    for ax in axes:
        ax.grid(ls=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    print(f"[saved] {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--output", default="results/layerwise_probe.json")
    ap.add_argument("--fig", default="results/figures/layerwise_probe_pooled.png")
    ap.add_argument("--fig-depth", default="results/figures/layerwise_depth_profile.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = args.models or DEFAULT_MODELS

    results = {}
    for key in models:
        try:
            results[key] = run_model(cfg, key)
        except FileNotFoundError as e:
            print(f"[skip] {key}: {e}")

    # nuage poolé : toutes les paires (géométrie, F1) couche × modèle
    pooled_f1, pooled_rmn, pooled_ani = [], [], []
    for r in results.values():
        for li in range(r["n_layers"]):
            L = r["layers"][li]
            pooled_f1.append(L["f1_macro_pres"])
            pooled_rmn.append(L["rankme_normalized"])
            pooled_ani.append(L["anisotropy"])

    pooled = {}
    if len(pooled_f1) >= 3:
        pooled = {
            "n_points": len(pooled_f1),
            "spearman_rankme_norm_vs_f1": _spearman(pooled_rmn, pooled_f1),
            "kendall_rankme_norm_vs_f1": _kendall(pooled_rmn, pooled_f1),
            "spearman_anisotropy_vs_f1": _spearman(pooled_ani, pooled_f1),
            "kendall_anisotropy_vs_f1": _kendall(pooled_ani, pooled_f1),
        }

    out = {
        "metric": "f1_macro_pres",
        "selection_metric": cfg.probe.selection_metric,
        "models": results,
        "pooled_correlation": pooled,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output}  ({len(results)} modèles)")
    if pooled:
        print(f"\nNuage poolé ({pooled['n_points']} points couche×modèle) :")
        print(f"  RankMe norm. ↔ F1 : ρ={pooled['spearman_rankme_norm_vs_f1']:+.3f}  "
              f"τ={pooled['kendall_rankme_norm_vs_f1']:+.3f}")
        print(f"  Anisotropie  ↔ F1 : ρ={pooled['spearman_anisotropy_vs_f1']:+.3f}  "
              f"τ={pooled['kendall_anisotropy_vs_f1']:+.3f}")
    if results:
        make_figure(results, args.fig)
        make_depth_figure(results, args.fig_depth)


if __name__ == "__main__":
    main()