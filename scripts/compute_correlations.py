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

from src.latent import rankme, anisotropy, rankme_normalized, subsample


# ------------------------------------------------------------------ helpers

def _load_f1(probe_json: str) -> dict[str, float]:
    """f1_macro_pres test par modèle depuis probe_knn_cgrid.json."""
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["test"]["f1_macro_pres"])
            for m, d in probe.items()
            if "test" in d and "f1_macro_pres" in d["test"]}


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
    ap.add_argument("--emb-dir",
                    default="/home/erazal/Documents/Mémoire/embeddings",
                    help="répertoire des embeddings .npy")
    args = ap.parse_args()

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

    out = {
        "probe_source": args.probe_json,
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


if __name__ == "__main__":
    main()
