#!/usr/bin/env python3
"""
silhouette_top8_test.py — Test de robustesse de la silhouette (et 10 autres métriques)
sur le palier compétitif (top-8, top-6).

Population définie par rang de F1 (pas de seuil codé en dur).
Nécessite : all_scores_consolidated.json + old consolidated (f7bc3e4^) pour les 5
métriques spectrales perdues à la régénération.

Schéma : EXCLUSIVEMENT manuscrit (f1_macro_pres_11cls, provenance without_rhol).

Auteur : Hermes (eo-researcher), 2026-07-15
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, kendalltau

REPO = Path(__file__).resolve().parent.parent
SEED = 42
N_BOOTSTRAP = 2000
rng = np.random.RandomState(SEED)

# ── Chargement ──────────────────────────────────────────────────────────

def load_consolidated():
    """Charge le fichier canonique actuel (F1 + 6 métriques cluster)."""
    path = REPO / "results" / "all_scores_consolidated.json"
    with open(path) as f:
        return json.load(f)

def load_old_consolidated():
    """Charge l'ancienne version (f7bc3e4^) pour les 5 métriques spectrales perdues."""
    path = REPO / "results" / "all_scores_consolidated.json"
    # Récupère via git
    import subprocess
    r = subprocess.run(
        ["git", "show", "f7bc3e4^:results/all_scores_consolidated.json"],
        capture_output=True, text=True, cwd=str(REPO)
    )
    if r.returncode != 0:
        print(f"ERREUR git: {r.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(r.stdout)


# ── Assemblage du dataset ───────────────────────────────────────────────

def build_dataset():
    """Construit un dict {model_name: {f1_11cls, metric1, metric2, ...}}."""
    new = load_consolidated()
    old = load_old_consolidated()

    old_by = {m["model"]: m for m in old["models"]}
    new_by = {m["model"]: m for m in new["models"]}

    models = sorted(new_by.keys())
    dataset = {}

    for name in models:
        n = new_by[name]
        o = old_by.get(name, {})

        # F1 du manuscrit (11cls, without_rhol)
        f1_11 = n["f1_macro_pres_11cls"]
        f1_12 = n["f1_macro_pres_12cls"]

        # Métriques cluster (6) — du nouveau fichier
        row = {
            "f1_11cls": f1_11,
            "f1_12cls": f1_12,
            "silhouette_score": n["silhouette_score"],
            "davies_bouldin_index": n["davies_bouldin_index"],
            "calinski_harabasz_index": n["calinski_harabasz_index"],
            "nc1": n["nc1"],
            "dim_mle_mean": n["dim_mle_mean"],
            "dim_mle_median": n["dim_mle_median"],
        }

        # Métriques spectrales (5) — de l'ancien fichier (calculées sur embeddings,
        # confirmées identiques entre old/new pour les champs communs)
        row["global_effective_rank"] = o.get("global_effective_rank", None)
        row["global_stable_rank"] = o.get("global_stable_rank", None)
        row["global_participation_ratio"] = o.get("global_participation_ratio", None)
        row["global_alpha"] = o.get("global_alpha", None)

        # NC2 deviation ETF — calculée à partir de nc2_mean_cos
        # 11cls: ideal_cos = -1/(11-1) = -0.1
        nc2_mean = n["nc2_mean_cos"]
        row["nc2_deviation_etf_11cls"] = abs(nc2_mean - (-0.1))

        # 12cls version (pour référence)
        row["nc2_deviation_etf_12cls"] = abs(nc2_mean - (-1.0 / 11.0))

        dataset[name] = row

    return dataset


# ── Métriques à tester ──────────────────────────────────────────────────

METRIC_KEYS = [
    ("silhouette_score",           "Silhouette"),
    ("davies_bouldin_index",       "Davies-Bouldin"),
    ("calinski_harabasz_index",    "Calinski-Harabasz"),
    ("nc1",                        "NC1"),
    ("nc2_deviation_etf_11cls",    "NC2 dév. ETF"),
    ("dim_mle_mean",               "Dim. MLE (moy.)"),
    ("dim_mle_median",             "Dim. MLE (méd.)"),
    ("global_effective_rank",      "Rang effectif"),
    ("global_stable_rank",         "Stable rank"),
    ("global_participation_ratio", "Participation ratio"),
    ("global_alpha",               "α spectral"),
]

# Vérifier disponibilité
def check_availability(dataset):
    missing = []
    available = []
    for key, label in METRIC_KEYS:
        vals = [dataset[m].get(key) for m in dataset]
        if any(v is None for v in vals):
            missing.append((key, label))
        else:
            available.append((key, label))
    return available, missing


# ── Définition des populations ──────────────────────────────────────────

def define_populations(dataset, f1_key="f1_11cls"):
    """Définit n12, top8, top6 par rang de F1 décroissant."""
    models_sorted = sorted(dataset.keys(), key=lambda m: dataset[m][f1_key], reverse=True)
    f1s = [dataset[m][f1_key] for m in models_sorted]

    n12 = models_sorted
    top8 = models_sorted[:8]
    top6 = models_sorted[:6]

    excluded_8 = models_sorted[8:]
    excluded_6 = models_sorted[6:]

    # Vérification des seuils implicites
    threshold_top8 = f1s[7]  # F1 du 8ème
    threshold_top6 = f1s[5]  # F1 du 6ème

    return {
        "n12": {"models": n12, "n": 12},
        "top8": {"models": top8, "n": 8, "excluded": excluded_8,
                 "threshold_f1": threshold_top8},
        "top6": {"models": top6, "n": 6, "excluded": excluded_6,
                 "threshold_f1": threshold_top6},
    }


# ── Calculs statistiques ────────────────────────────────────────────────

def compute_correlations(metric_vals, f1_vals):
    """Spearman rho + p, Kendall tau."""
    rho, p_spearman = spearmanr(metric_vals, f1_vals)
    tau, p_kendall = kendalltau(metric_vals, f1_vals)
    return {"spearman_rho": float(rho), "spearman_p": float(p_spearman),
            "kendall_tau": float(tau), "kendall_p": float(p_kendall)}


def bootstrap_ci(metric_vals, f1_vals, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """IC95 bootstrap sur Spearman rho."""
    rng_local = np.random.RandomState(seed)
    n = len(metric_vals)
    idx = np.arange(n)
    rhos = []
    for _ in range(n_bootstrap):
        boot_idx = rng_local.choice(idx, size=n, replace=True)
        rho, _ = spearmanr(metric_vals[boot_idx], f1_vals[boot_idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    ci_low = float(np.percentile(rhos, 2.5))
    ci_high = float(np.percentile(rhos, 97.5))
    return {"ci95_low": ci_low, "ci95_high": ci_high, "bootstrap_n": n_bootstrap}


def leave_one_out(metric_vals, f1_vals, model_names):
    """Leave-one-out pour une métrique."""
    n = len(metric_vals)
    rhos = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho, _ = spearmanr(metric_vals[mask], f1_vals[mask])
        rhos.append((model_names[i], float(rho)))
    rhos_only = [r[1] for r in rhos]
    most_influential = min(rhos, key=lambda x: abs(x[1] - np.mean(rhos_only)))
    return {
        "all": rhos,
        "min": float(np.min(rhos_only)),
        "max": float(np.max(rhos_only)),
        "delta": float(np.max(rhos_only) - np.min(rhos_only)),
        "most_influential_model": most_influential[0],
        "most_influential_rho": most_influential[1],
    }


def benjamini_hochberg(p_values):
    """Correction BH. Retourne p_BH pour chaque test."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    p_bh = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        p_bh[orig_idx] = min(p * n / rank, 1.0)
    return p_bh


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    print("=== Chargement du dataset ===")
    dataset = build_dataset()
    models_all = sorted(dataset.keys())
    print(f"Modèles chargés : {len(models_all)}")

    available, missing = check_availability(dataset)
    print(f"Métriques disponibles : {len(available)}/{len(METRIC_KEYS)}")
    if missing:
        print(f"Métriques MANQUANTES : {[label for _, label in missing]}")

    # Populations (11cls)
    print("\n=== Définition des populations (11cls) ===")
    pops = define_populations(dataset, "f1_11cls")

    for pop_name, pop_info in pops.items():
        f1s = [dataset[m]["f1_11cls"] for m in pop_info["models"]]
        print(f"\n{pop_name} (n={pop_info['n']}):")
        for m in pop_info["models"]:
            print(f"  {m:25s}  F1={dataset[m]['f1_11cls']:.4f}")
        if "excluded" in pop_info:
            print(f"  --- exclus (F1 < {pop_info['threshold_f1']:.4f}) ---")
            for m in pop_info["excluded"]:
                print(f"  {m:25s}  F1={dataset[m]['f1_11cls']:.4f}")

    # ── Calculs ────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "script": "scripts/silhouette_top8_test.py",
            "date": "2026-07-15",
            "f1_source": "results/all_scores_consolidated.json (f1_macro_pres_11cls, without_rhol)",
            "spectral_source": "git show f7bc3e4^:results/all_scores_consolidated.json",
            "nc2_source": "abs(nc2_mean_cos - (-0.1)) — 11cls ideal ETF cos",
            "schema": "11cls manuscrit (without_rhol)",
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "available_metrics": len(available),
            "total_metrics": len(METRIC_KEYS),
            "missing_metrics": [label for _, label in missing],
        },
        "populations": {},
        "leave_one_out": {},
    }

    for pop_name, pop_info in pops.items():
        models = pop_info["models"]
        f1_vals = np.array([dataset[m]["f1_11cls"] for m in models])
        n = len(models)

        pop_results = {
            "models": models,
            "n": n,
            "f1_values": {m: dataset[m]["f1_11cls"] for m in models},
        }

        # Corrélations par métrique
        metrics_results = []
        all_p_values = []

        for key, label in available:
            metric_vals = np.array([dataset[m][key] for m in models])

            corr = compute_correlations(metric_vals, f1_vals)
            ci = bootstrap_ci(metric_vals, f1_vals)
            all_p_values.append(corr["spearman_p"])

            metrics_results.append({
                "metric_key": key,
                "metric_label": label,
                **corr,
                **ci,
            })

        # Corrections multiples (au sein de cette population seulement)
        p_vals = [m["spearman_p"] for m in metrics_results]
        n_tests = len(p_vals)
        p_bonf = [min(p * n_tests, 1.0) for p in p_vals]
        p_bh = benjamini_hochberg(p_vals)

        for i, mr in enumerate(metrics_results):
            mr["n_tests"] = n_tests
            mr["p_bonferroni"] = p_bonf[i]
            mr["p_bh"] = p_bh[i]
            # Δ vs n12 (pour top8/top6)
            if pop_name != "n12":
                n12_rho = None
                for n12_mr in results["populations"].get("n12", {}).get("metrics", []):
                    if n12_mr["metric_key"] == mr["metric_key"]:
                        n12_rho = n12_mr["spearman_rho"]
                        break
                if n12_rho is not None:
                    mr["delta_vs_n12"] = mr["spearman_rho"] - n12_rho

        pop_results["metrics"] = metrics_results
        results["populations"][pop_name] = pop_results

    # ── Leave-one-out (silhouette seulement) ──────────────────────────
    print("\n=== Leave-one-out silhouette ===")
    for pop_name, pop_info in pops.items():
        models = pop_info["models"]
        f1_vals = np.array([dataset[m]["f1_11cls"] for m in models])
        sil_vals = np.array([dataset[m]["silhouette_score"] for m in models])

        loo = leave_one_out(sil_vals, f1_vals, models)
        results["leave_one_out"][pop_name] = loo

        full_rho, _ = spearmanr(sil_vals, f1_vals)
        print(f"  {pop_name} (n={len(models)}): ρ_full={full_rho:.4f}, "
              f"LOO min={loo['min']:.4f}, max={loo['max']:.4f}, "
              f"Δ={loo['delta']:.4f}, influent={loo['most_influential_model']}")

    # ── Colinéarité inter-métriques (Tâche 2) ──────────────────────────
    print("\n=== Colinéarité inter-métriques (n12, 11cls) ===")
    n12_models = pops["n12"]["models"]
    colinearity = {}
    for i, (key_i, label_i) in enumerate(available):
        for j, (key_j, label_j) in enumerate(available):
            if i >= j:
                continue
            vals_i = np.array([dataset[m][key_i] for m in n12_models])
            vals_j = np.array([dataset[m][key_j] for m in n12_models])
            rho, p = spearmanr(vals_i, vals_j)
            colinearity[f"{label_i} vs {label_j}"] = {
                "rho": float(rho), "p": float(p)
            }
            if abs(rho) > 0.9:
                print(f"  |ρ|={abs(rho):.3f}: {label_i} vs {label_j}")

    results["colinearity"] = colinearity

    # ── Sauvegarde ─────────────────────────────────────────────────────
    out_path = REPO / "results" / "silhouette_robustness_CANONICAL.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRésultats sauvegardés : {out_path}")
    print("Terminé.")


if __name__ == "__main__":
    main()
