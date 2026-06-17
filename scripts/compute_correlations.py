"""Corrélations Spearman & Kendall entre métriques géométriques et F1.

Sources :
  --latent-json  results/with_rhol/latent_metrics.json  (rankme, anisotropy, dim)
  --probe-json   results/with_rhol/probe_knn_cgrid.json  (f1_macro_pres test)

Calcule dinov3_vitl16_lvd latent metrics depuis embeddings/dinov3_vitl16_lvd_test.npy
si absents du latent-json.

Groupes :
  (a) ViT-L  : les 5 ViT-L/16 (dim 1024) — dinov3_sat, dinov3_lvd, simdinov2_l,
               satmae, scalemae. Architecture contrôlée, pré-entraînement varié. IC95 bootstrap.
  (b) dim-768 : vitb16_imagenet, dinov3_vitb16_lvd, simdinov2_vitb16
               n=3, point estimate uniquement.
  (c) global  : les 9 modèles. IC95 bootstrap (n=1000), n reste faible.
  (c') global avec rankme_normalized = rankme/dim.

Usage :
  python scripts/compute_correlations.py \\
      --config configs/frozen_eval.yaml \\
      --probe-json results/with_rhol/probe_knn_cgrid.json \\
      --latent-json results/with_rhol/latent_metrics.json \\
      --output results/correlations.json
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

from src.latent import (alpha_req, anisotropy, nesum, rankme,
                        rankme_normalized, subsample)
from src.transfer import logme_score


# ------------------------------------------------------------------ helpers

def _load_f1(probe_json: str) -> dict[str, float]:
    """f1_macro_pres test par modèle depuis probe_knn_cgrid.json."""
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["test"]["f1_macro_pres"])
            for m, d in probe.items()
            if "test" in d and "f1_macro_pres" in d["test"]}


def _load_labels(csv_path: str) -> np.ndarray:
    """Labels entiers (colonne 2) du CSV, dans l'ordre des lignes = ordre des embeddings."""
    import csv
    ys = []
    with open(csv_path) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if len(row) >= 2:
                ys.append(int(row[1]))
    return np.asarray(ys, dtype=np.int64)


def _load_latent(latent_json: str) -> dict[str, dict]:
    with open(latent_json) as f:
        data = json.load(f)
    return data.get("latent_metrics", data)


def _compute_latent_from_emb(emb_path: str, subsample_n: int = 20000,
                              n_pairs: int = 10000, seed: int = 42) -> dict:
    E = np.load(emb_path).astype(np.float32)
    E_sub = subsample(E, subsample_n, seed)
    return {
        "rankme": rankme(E_sub),
        "anisotropy": anisotropy(E_sub, n_pairs=n_pairs, seed=seed),
        "dim": int(E.shape[1]),
    }


def _compute_spectrum_from_emb(emb_path: str, subsample_n: int = 20000,
                                n_pairs: int = 10000, seed: int = 42) -> dict:
    """Spectre complet (rankme, rankme_normalized, alpha_req, nesum, anisotropy).

    Sur le MÊME sous-échantillon (n=20000, seed=42) que le benchmark nightly
    pour garantir des valeurs identiques (cf. spectrum_metrics.json).
    """
    E = np.load(emb_path).astype(np.float32)
    dim = int(E.shape[1])
    E_sub = subsample(E, subsample_n, seed)
    rm = rankme(E_sub)
    return {
        "dim": dim,
        "rankme": rm,
        "rankme_normalized": rm / dim,
        "alpha_req": alpha_req(E_sub),
        "nesum": nesum(E_sub),
        "anisotropy": anisotropy(E_sub, n_pairs=n_pairs, seed=seed),
    }


def _compute_logme(emb_train_path: str, labels: np.ndarray) -> float:
    """LogME (You et al. ICML 2021) sur embeddings train + labels — outil de RANG."""
    F = np.load(emb_train_path)
    if F.shape[0] != labels.shape[0]:
        raise ValueError(f"désalignement embeddings/labels : {F.shape[0]} vs {labels.shape[0]}"
                         f" ({emb_train_path})")
    return logme_score(F, labels)


# ------------------------------------------------------------------ correlation

def _spearman_r(x, y) -> float:
    """Spearman ρ via rangs (pur numpy, pas de scipy pour le point estimate)."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(x, y)
    return float(r)


def _kendall_tau(x, y) -> float:
    from scipy.stats import kendalltau
    tau, _ = kendalltau(x, y)
    return float(tau)


def _corr_point(x, y, label: str) -> dict:
    """Point estimates Spearman + Kendall sur (x, y) de longueur n."""
    n = len(x)
    if n < 3:
        return {
            "n": n,
            "status": f"INSUFFICIENT n={n} — corrélation non calculable (besoin n≥3)",
            "spearman_r": None,
            "kendall_tau": None,
        }
    return {
        "n": n,
        "spearman_r": _spearman_r(x, y),
        "kendall_tau": _kendall_tau(x, y),
    }


def _corr_bootstrap(x, y, n_boot: int = 1000, seed: int = 42) -> dict:
    """Point estimates + IC95 bootstrap (percentile) pour Spearman et Kendall."""
    n = len(x)
    if n < 3:
        return {
            "n": n,
            "status": f"INSUFFICIENT n={n} — corrélation non calculable",
            "spearman_r": None,
            "kendall_tau": None,
        }
    rng = np.random.RandomState(seed)
    sp_boot = np.empty(n_boot)
    kt_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        xb, yb = np.asarray(x)[idx], np.asarray(y)[idx]
        try:
            sp_boot[b] = _spearman_r(xb, yb)
            kt_boot[b] = _kendall_tau(xb, yb)
        except Exception:
            sp_boot[b] = np.nan
            kt_boot[b] = np.nan
    return {
        "n": n,
        "spearman_r": _spearman_r(x, y),
        "spearman_ci95": [float(np.nanpercentile(sp_boot, 2.5)),
                          float(np.nanpercentile(sp_boot, 97.5))],
        "kendall_tau": _kendall_tau(x, y),
        "kendall_ci95": [float(np.nanpercentile(kt_boot, 2.5)),
                         float(np.nanpercentile(kt_boot, 97.5))],
        "n_note": "n reste faible — IC95 à titre indicatif",
    }


# ------------------------------------------------------------------ groups

def _group_corr(models: list[str], latent: dict, f1: dict,
                metric: str, ci: bool, n_boot: int = 1000) -> dict:
    """Corrélation {metric} ↔ f1_macro_pres pour les modèles donnés."""
    available = [m for m in models if m in latent and m in f1]
    missing_latent = [m for m in models if m not in latent]
    missing_f1 = [m for m in models if m in latent and m not in f1]

    x = [latent[m][metric] for m in available]
    y = [f1[m] for m in available]

    base = {
        "models_requested": models,
        "models_available": available,
        "missing_latent": missing_latent,
        "missing_f1": missing_f1,
        "metric_x": metric,
        "metric_y": "f1_macro_pres",
    }
    if ci:
        base.update(_corr_bootstrap(x, y, n_boot=n_boot))
    else:
        base.update(_corr_point(x, y, metric))
        base["ci_note"] = "CI non pertinent à n=3"
    return base


# ------------------------------------------------------- transférabilité n=12

# Métriques additionnelles (nightly 2026-06-14) : α-ReQ, NESum (spectre test) et
# LogME (train). Évaluées sur les 12 modèles (9 frozen + 3 fine-tunés) pour mesurer
# si elles prédisent le F1 test mieux que RankMe — qui s'effondre dès qu'on ajoute
# les fine-tunés (F1 élevé sans RankMe élevé).
TRANSFER_MODELS = [
    "resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd", "dinov3_vitl16_sat",
    "dinov3_vitl16_lvd", "simdinov2_vitb16", "simdinov2_vitl16", "satmae_vitl16",
    "scalemae_vitl16", "resnet50_arctic", "vitb16_arctic", "vitb16_fulft_arctic",
]
TRANSFER_METRICS = ["rankme", "rankme_normalized", "alpha_req", "nesum", "anisotropy", "logme"]


def _transferability_block(emb_dir: str, csv_dir: str, f1: dict[str, float],
                            n_boot: int, seed: int, do_logme: bool) -> dict:
    """Calcule spectre (test) + LogME (train) pour les 12 modèles, puis corrèle
    chaque métrique ↔ F1 (n=12, bootstrap IC95)."""
    labels = _load_labels(os.path.join(csv_dir, "train.csv")) if do_logme else None
    per_model: dict[str, dict] = {}
    available = []
    for m in TRANSFER_MODELS:
        test_p = os.path.join(emb_dir, f"{m}_test.npy")
        if not os.path.exists(test_p) or m not in f1:
            print(f"[transfer] skip {m} (emb test ou F1 manquant)")
            continue
        rec = _compute_spectrum_from_emb(test_p, seed=seed)
        if do_logme:
            train_p = os.path.join(emb_dir, f"{m}_train.npy")
            if os.path.exists(train_p):
                print(f"[transfer] LogME {m} ...", flush=True)
                rec["logme"] = _compute_logme(train_p, labels)
            else:
                print(f"[transfer] {m}_train.npy absent — LogME ignoré pour ce modèle")
        rec["f1_macro_pres"] = f1[m]
        per_model[m] = rec
        available.append(m)

    corr = {}
    for metric in TRANSFER_METRICS:
        usable = [m for m in available if metric in per_model[m]]
        if len(usable) < 3:
            continue
        x = [per_model[m][metric] for m in usable]
        y = [per_model[m]["f1_macro_pres"] for m in usable]
        c = _corr_bootstrap(x, y, n_boot=n_boot, seed=seed)
        c["models"] = usable
        corr[metric] = c

    return {
        "models": available,
        "f1_metric": "f1_macro_pres",
        "subsample_n": 20000,
        "seed": seed,
        "per_model": per_model,
        "correlations": corr,
        "note": ("Métriques nightly 2026-06-14 (α-ReQ, NESum, LogME) sur les 12 modèles. "
                 "F1 = même source que le rapport (probe étendu). LogME = outil de RANG."),
    }


def _plot_logme_vs_f1(block: dict, out_png: str) -> None:
    """Scatter LogME (train) vs F1 (test), n=12 — cohérent avec le F1 du rapport."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = block["per_model"]
    models = [m for m in block["models"] if "logme" in pm[m]]
    if not models:
        print("[fig] aucun LogME disponible — figure non générée")
        return
    x = [pm[m]["logme"] for m in models]
    y = [pm[m]["f1_macro_pres"] for m in models]
    c = block["correlations"].get("logme", {})
    rho = c.get("spearman_r")
    tau = c.get("kendall_tau")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, s=110, color="#3b76af", edgecolor="black", zorder=3)
    for m, xi, yi in zip(models, x, y):
        ax.annotate(m, (xi, yi), textcoords="offset points", xytext=(7, 4), fontsize=9)
    title = "LogME (train) vs F1-macro(11) test — 12 modèles"
    if rho is not None:
        title += f"\nSpearman $\\rho$={rho:+.3f}   Kendall $\\tau$={tau:+.3f}"
    ax.set_title(title)
    ax.set_xlabel("LogME (train, 49 433 tuiles) — score")
    ax.set_ylabel("F1-macro-présent (test, 17 598 tuiles)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig] {out_png}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--probe-json", default="results/with_rhol/probe_knn_cgrid.json")
    ap.add_argument("--latent-json", default="results/with_rhol/latent_metrics.json")
    ap.add_argument("--output", default="results/correlations.json")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emb-dir", default=None,
                    help="répertoire des embeddings .npy (défaut: <cfg.paths.emb_dir>)")
    ap.add_argument("--probe-json-all",
                    default="results/with_rhol/probe_knn_cgrid.json",
                    help="probe 12 modèles (F1 canonique du rapport) pour le bloc transférabilité. "
                         "Lit la source canonique (grille étendue, F1≈0.4789), "
                         "JAMAIS probe_knn.json (F1≈0.4675 périmé).")
    ap.add_argument("--csv-dir", default=None,
                    help="répertoire des CSV (train.csv pour les labels LogME) "
                         "(défaut: <cfg.paths.csv_dir>)")
    ap.add_argument("--no-logme", action="store_true",
                    help="désactive le calcul LogME (train) dans le bloc transférabilité")
    ap.add_argument("--fig", default="docs/figures/logme_vs_f1.png",
                    help="chemin de la figure LogME↔F1 (vide pour ne pas la générer)")
    args = ap.parse_args()

    # Charge la config et résout les chemins par défaut (jamais hardcodés).
    from src.config import load_config
    cfg = load_config(args.config)
    if args.emb_dir is None:
        args.emb_dir = cfg.paths.emb_dir
    if args.csv_dir is None:
        args.csv_dir = cfg.paths.csv_dir

    f1 = _load_f1(args.probe_json)
    latent = _load_latent(args.latent_json)

    # Calcul latent pour dinov3_vitl16_lvd si absent du JSON
    lvd_key = "dinov3_vitl16_lvd"
    if lvd_key not in latent:
        emb_path = os.path.join(args.emb_dir, f"{lvd_key}_test.npy")
        if os.path.exists(emb_path):
            print(f"[latent] Calcul {lvd_key} depuis {emb_path} ...")
            latent[lvd_key] = _compute_latent_from_emb(emb_path)
            print(f"  rankme={latent[lvd_key]['rankme']:.2f}  "
                  f"anisotropy={latent[lvd_key]['anisotropy']:.4f}  "
                  f"dim={latent[lvd_key]['dim']}")
        else:
            print(f"[warn] {lvd_key}_test.npy introuvable — latent metrics manquants")

    # Ajouter rankme_normalized dans la copie locale
    for m, v in latent.items():
        v["rankme_normalized"] = v["rankme"] / v["dim"]

    # ViT-L/16 (dim 1024) : architecture contrôlée, 5 pré-entraînements (2 satellite-MAE,
    # 1 satellite-DINO, 1 naturel-DINO, 1 plante-DINO) — le groupe le mieux contrôlé.
    VITL = ["dinov3_vitl16_sat", "dinov3_vitl16_lvd", "simdinov2_vitl16",
            "satmae_vitl16", "scalemae_vitl16"]
    DIM768 = ["vitb16_imagenet", "dinov3_vitb16_lvd", "simdinov2_vitb16"]
    ALL_GLOBAL = ["resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
                  "simdinov2_vitb16", "dinov3_vitl16_sat", "dinov3_vitl16_lvd",
                  "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16"]

    results = {}

    # (a) ViT-L — n=5, architecture contrôlée, IC95 bootstrap
    results["vit_l"] = {
        "rankme": _group_corr(VITL, latent, f1, "rankme", ci=True, n_boot=args.n_bootstrap),
        "anisotropy": _group_corr(VITL, latent, f1, "anisotropy", ci=True, n_boot=args.n_bootstrap),
    }
    for metric_key in ("rankme", "anisotropy"):
        results["vit_l"][metric_key]["n_note"] = "n=5 (ViT-L, archi contrôlée) — IC95 indicatif"

    # (b) dim-768 — n=3, point estimate
    results["dim_768"] = {
        "rankme": _group_corr(DIM768, latent, f1, "rankme", ci=False),
        "anisotropy": _group_corr(DIM768, latent, f1, "anisotropy", ci=False),
    }

    # (c) global 9 modèles — IC95 bootstrap
    results["global"] = {
        "rankme": _group_corr(ALL_GLOBAL, latent, f1, "rankme",
                              ci=True, n_boot=args.n_bootstrap),
        "anisotropy": _group_corr(ALL_GLOBAL, latent, f1, "anisotropy",
                                  ci=True, n_boot=args.n_bootstrap),
    }

    # (c') global avec rankme_normalized
    results["global_rankme_normalized"] = {
        "rankme_normalized": _group_corr(ALL_GLOBAL, latent, f1, "rankme_normalized",
                                         ci=True, n_boot=args.n_bootstrap),
        "anisotropy": _group_corr(ALL_GLOBAL, latent, f1, "anisotropy",
                                  ci=True, n_boot=args.n_bootstrap),
    }

    # (d) transférabilité n=12 — α-ReQ, NESum, LogME (+ rankme/anisotropie pour mémoire)
    print("\n[transfer] bloc 12 modèles (α-ReQ, NESum, LogME) ...")
    f1_all = _load_f1(args.probe_json_all)
    transfer = _transferability_block(args.emb_dir, args.csv_dir, f1_all,
                                      n_boot=args.n_bootstrap, seed=args.seed,
                                      do_logme=not args.no_logme)
    results["transferability_all12"] = transfer

    # (e) Paliers compétitifs top-K par F1, sur les 12 modèles.
    # Pour CHAQUE K ∈ {8, 6}, on restreint aux K premiers modèles (tri F1 desc.)
    # et on recalcule les corrélations sur ce sous-ensemble.  Les métriques sont
    # celles du bloc transferability_all12 (cohérence n=12 vs palier).
    paliers: dict[str, dict] = {}
    per_model = transfer["per_model"]
    all12_sorted_by_f1 = sorted(per_model.keys(),
                                 key=lambda m: per_model[m]["f1_macro_pres"],
                                 reverse=True)
    palier_metrics = ["logme", "rankme", "rankme_normalized", "alpha_req",
                      "nesum", "anisotropy"]
    for K in (8, 6):
        K_eff = min(K, len(all12_sorted_by_f1))
        top_k = all12_sorted_by_f1[:K_eff]
        sub = {m: per_model[m] for m in top_k}
        paliers[f"top_{K_eff}"] = {
            "models_used": top_k,
            "K": K_eff,
            "f1_range": [sub[top_k[-1]]["f1_macro_pres"],
                         sub[top_k[0]]["f1_macro_pres"]],
        }
        per_metric = {}
        for metric in palier_metrics:
            avail = [m for m in top_k if metric in sub[m]]
            if len(avail) < 3:
                continue
            x = [sub[m][metric] for m in avail]
            y = [sub[m]["f1_macro_pres"] for m in avail]
            per_metric[metric] = _corr_bootstrap(x, y, n_boot=args.n_bootstrap, seed=args.seed)
            per_metric[metric]["models"] = avail
        paliers[f"top_{K_eff}"]["correlations"] = per_metric
    results["competitive_tiers"] = paliers

    out = {
        "probe_source": args.probe_json,
        "probe_source_all12": args.probe_json_all,
        "latent_source": args.latent_json,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "latent_values_used": {
            m: {k: v for k, v in vals.items()}
            for m, vals in latent.items()
        },
        "f1_values_used": f1,
        "vit_l": results["vit_l"],
        "dim_768": results["dim_768"],
        "global": results["global"],
        "global_rankme_normalized": results["global_rankme_normalized"],
        "transferability_all12": transfer,
        "competitive_tiers": paliers,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f_out:
        json.dump(out, f_out, indent=2)
    print(f"[saved] {args.output}")

    # Résumé terminal
    print("\n=== CORRÉLATIONS GÉOMÉTRIE ↔ F1 ===")
    for group_name, group in [("ViT-L (n=5, archi contrôlée)", results["vit_l"]),
                               ("dim-768 (n=3)", results["dim_768"]),
                               ("global (n=9)", results["global"]),
                               ("global rankme_norm (n=9)", results["global_rankme_normalized"])]:
        print(f"\n{group_name} :")
        for mx, gr in group.items():
            n = gr.get("n", "?")
            if gr.get("spearman_r") is None:
                print(f"  {mx:<22} n={n}  INSUFFICIENT")
            else:
                sp = gr["spearman_r"]
                kt = gr["kendall_tau"]
                ci_sp = gr.get("spearman_ci95", "—")
                print(f"  {mx:<22} n={n}  ρ={sp:+.3f}  τ={kt:+.3f}  "
                      f"IC95ρ={ci_sp}")

    print(f"\ntransférabilité (n={len(transfer['models'])}, F1 du rapport) :")
    for mx, gr in transfer["correlations"].items():
        sp, kt = gr["spearman_r"], gr["kendall_tau"]
        ci = gr.get("spearman_ci95", "—")
        print(f"  {mx:<22} n={gr['n']}  ρ={sp:+.3f}  τ={kt:+.3f}  IC95ρ={ci}")

    # Paliers compétitifs
    for tier_name, tier in paliers.items():
        print(f"\npalier {tier_name} ({tier['K']} modèles, F1∈[{tier['f1_range'][0]:.4f}, "
              f"{tier['f1_range'][1]:.4f}]) :")
        for mx, gr in tier["correlations"].items():
            sp, kt = gr["spearman_r"], gr["kendall_tau"]
            ci = gr.get("spearman_ci95", "—")
            print(f"  {mx:<22} n={gr['n']}  ρ={sp:+.3f}  τ={kt:+.3f}  IC95ρ={ci}")

    # Figure LogME ↔ F1 (cohérente avec le F1 canonique du rapport)
    if args.fig:
        _plot_logme_vs_f1(transfer, args.fig)


if __name__ == "__main__":
    main()
