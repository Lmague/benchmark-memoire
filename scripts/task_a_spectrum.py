#!/usr/bin/env python3
"""TÂCHE A — α-ReQ (Agrawal NeurIPS 2022) + NESum (He & Ozay ICML 2022) sur les 12 modèles.

Réutilise le MÊME sous-échantillon test (20k, seed 42) que ``rankme`` (cf.
``base.yaml`` : ``rankme_split: test``, ``rankme_subsample: 20000``) pour
que les 4 métriques soient comparables entre elles.

Formules :
- α-ReQ : pente (en valeur absolue) d'une régression log-log sur le spectre
  décroissant de la covariance empirique centrée.  Référence : Agrawal et al.,
  NeurIPS 2022, Section 3.1 (https://papers.neurips.cc/paper_files/paper/
  2022/file/70596d70542c51c8d9b4e423f4bf2736-Paper-Conference.pdf).
- NESum : Σᵢ λᵢ / λ₁ sur le spectre trié de la covariance centrée.
  Référence : He & Ozay, ICML 2022, Déf. 4.1 (https://proceedings.mlr.press/
  v162/he22c/he22c.pdf).

Sorties :
  - results/transfer/spectrum_metrics.json  (par modèle)
  - results/transfer/spectrum_metrics.csv    (flat)
  - results/transfer/correlations_with_f1.json  (Spearman, Kendall)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.latent import (anisotropy, nesum, rankme, rankme_normalized, alpha_req,
                        subsample)

EMB = PROJ / "embeddings"
OUT = PROJ / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)

# 12 modèles, dans l'ordre canonique des configs
MODELS = [
    "resnet50_imagenet",
    "vitb16_imagenet",
    "dinov3_vitb16_lvd",
    "dinov3_vitl16_sat",
    "dinov3_vitl16_lvd",
    "simdinov2_vitb16",
    "simdinov2_vitl16",
    "satmae_vitl16",
    "scalemae_vitl16",
    "resnet50_arctic",
    "vitb16_arctic",
    "vitb16_fulft_arctic",
]


def _load_f1_canonical() -> dict[str, float]:
    """F1 depuis la source canonique ``probe_knn_cgrid.json`` (grille étendue,
    F1≈0.4789 pour le co-leader).  L'ancien ``probe_knn.json`` (grille
    restreinte, F1≈0.4675) est PÉRIMÉ et marqué .deprecated.
    """
    pk = PROJ / "results" / "with_rhol" / "probe_knn_cgrid.json"
    with open(pk) as f:
        d = json.load(f)
    probe = d.get("probe", d)
    return {m: float(probe[m]["test"]["f1_macro_pres"])
            for m in probe if "test" in probe[m]}


F1_PRES = _load_f1_canonical()


def compute_one(key: str, subsample_n: int = 20000, seed: int = 42) -> dict:
    """Calcule les 4 métriques latentes sur le sous-échantillon TEST du modèle."""
    E = np.load(EMB / f"{key}_test.npy").astype(np.float32)
    Z = subsample(E, subsample_n, seed)
    return {
        "model": key,
        "dim": int(E.shape[1]),
        "n_test": int(E.shape[0]),
        "subsample_n": int(Z.shape[0]),
        "rankme": float(rankme(Z)),
        "rankme_normalized": float(rankme_normalized(Z)),
        "alpha_req": float(alpha_req(Z)),
        "nesum": float(nesum(Z)),
        "anisotropy": float(anisotropy(Z, n_pairs=10000, seed=seed)),
        "f1_macro_pres_test": F1_PRES[key],
    }


def correlate(metrics: list[dict], metric_keys: list[str], f1_key: str) -> dict:
    """Spearman + Kendall entre chaque métrique latente et le F1 test."""
    out = {}
    f1 = np.array([m[f1_key] for m in metrics])
    for mk in metric_keys:
        x = np.array([m[mk] for m in metrics])
        if np.isnan(x).any():
            out[mk] = {"spearman_r": None, "kendall_tau": None, "n": int(len(x)),
                       "note": "NaN present"}
            continue
        rs, ps = spearmanr(x, f1)
        kt, kp = kendalltau(x, f1)
        out[mk] = {
            "spearman_r": float(rs), "spearman_p": float(ps),
            "kendall_tau": float(kt), "kendall_p": float(kp),
            "n": int(len(x)),
        }
    return out


def main() -> None:
    print("=" * 70)
    print("TÂCHE A — α-ReQ + NESum (12 modèles, sous-échantillon test 20k seed 42)")
    print("=" * 70)
    results = []
    for k in MODELS:
        try:
            r = compute_one(k)
            results.append(r)
            print(f"[{k:24s}] dim={r['dim']:5d}  RankMe={r['rankme']:7.2f}  "
                  f"α-ReQ={r['alpha_req']:5.3f}  NESum={r['nesum']:7.3f}  "
                  f"Aniso={r['anisotropy']:+.4f}  F1={r['f1_macro_pres_test']:.4f}")
        except Exception as exc:
            print(f"[{k:24s}] ERREUR : {exc!r}")

    # CSV plat
    csv_path = OUT / "spectrum_metrics.csv"
    keys = ["model", "dim", "n_test", "subsample_n", "rankme", "rankme_normalized",
            "alpha_req", "nesum", "anisotropy", "f1_macro_pres_test"]
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\n[csv] -> {csv_path}")

    # JSON complet
    json_path = OUT / "spectrum_metrics.json"
    with open(json_path, "w") as f:
        json.dump({"per_model": results,
                   "f1_source": "results/with_rhol/probe_knn_cgrid.json (f1_macro_pres canonique)",
                   "subsample_n": 20000, "seed": 42,
                   "split": "test", "embedding_pooler": True,
                   "formulae": {
                       "alpha_req": "Agrawal et al. NeurIPS 2022 — log(λⱼ) = -α log(j) + c ; α = -pente",
                       "nesum":     "He & Ozay ICML 2022 Déf. 4.1 — Σᵢ λᵢ/λ₁ sur spectre de cov. centrée",
                   }}, f, indent=2)
    print(f"[json] -> {json_path}")

    # Corrélations
    corr = correlate(results,
                     ["rankme", "rankme_normalized", "alpha_req", "nesum", "anisotropy"],
                     "f1_macro_pres_test")
    corr_path = OUT / "correlations_with_f1.json"
    with open(corr_path, "w") as f:
        json.dump({"metrics_vs_f1_macro_pres_test": corr,
                   "n_models": len(results),
                   "seed_subsample": 42}, f, indent=2)
    print(f"\n[corr] Spearman & Kendall (n={len(results)}) :")
    for mk, v in corr.items():
        if v.get("spearman_r") is not None:
            print(f"  {mk:22s}  ρ={v['spearman_r']:+.3f} (p={v['spearman_p']:.3g})  "
                  f"τ={v['kendall_tau']:+.3f} (p={v['kendall_p']:.3g})")

    print(f"\n[json] -> {corr_path}")
    print("=" * 70)
    print("TÂCHE A terminée.")
    print("=" * 70)


if __name__ == "__main__":
    main()
