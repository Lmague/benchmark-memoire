"""Bootstrap apparié toutes paires parmi les modèles frozen disponibles.

Lit best_C depuis --probe-json (probe_knn_cgrid.json), re-fitte + bootstrappe chaque
modèle (même infrastructure que bootstrap_ci.py), puis calcule toutes les paires N*(N-1)/2.

Sorties :
  --output-json  results/significance_matrix.json
  --output-png   results/significance_matrix.png  (heatmap P(A>B), matplotlib pur)

Usage :
  python scripts/significance_matrix.py \\
      --config configs/frozen_eval.yaml \\
      --probe-json results/with_rhol/probe_knn_cgrid.json \\
      --n-bootstrap 1000
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import load_features
from src.probe import _standardize
from src.utils import make_canonical_lr

N_CLASSES = 12


# ------------------------------------------------------------------ bootstrap utils

def _load_best_C(probe_json: str) -> dict[str, float]:
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["best_C"]) for m, d in probe.items() if "best_C" in d}


def _refit_predict(cfg, model_key: str, best_C: float, seed: int = 42):
    feats = load_features(cfg, model_key)
    ytr = np.asarray(feats["train"][1])
    yte = np.asarray(feats["test"][1])
    xtr, _xva, xte = _standardize(feats)
    clf = make_canonical_lr(C=best_C, max_iter=cfg.probe.max_iter, random_state=seed)
    clf.fit(xtr, ytr)
    return yte, clf.predict(xte)


def _bootstrap_f1_pres(y_true, y_pred, n_boot: int, seed: int = 42) -> np.ndarray:
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    rng = np.random.RandomState(seed)
    f1_pres = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        present = sorted({int(v) for v in yt})
        f1_pres[b] = f1_score(yt, yp, average="macro", zero_division=0, labels=present)
    return f1_pres


def _observed_f1_pres(y_true, y_pred) -> float:
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    present = sorted({int(v) for v in y_true})
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present))


def _summary(samples: np.ndarray, observed: float) -> dict:
    return {
        "observed": float(observed),
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


# ------------------------------------------------------------------ core logic

def run_bootstrap(cfg, probe_json: str, n_boot: int, seed: int = 42):
    """Refit + bootstrap pour chaque modèle disponible dans probe_json."""
    best_C = _load_best_C(probe_json)
    emb_dir = cfg.paths.emb_dir
    results: dict[str, dict] = {}
    samples: dict[str, np.ndarray] = {}

    for model_key, c in sorted(best_C.items()):
        train_emb = os.path.join(emb_dir, f"{model_key}_train.npy")
        if not os.path.exists(train_emb):
            print(f"[skip] {model_key}: train embeddings absents")
            continue
        print(f"[bootstrap] {model_key}: re-fit (C={c}) + {n_boot} tirages")
        y_true, y_pred = _refit_predict(cfg, model_key, c, seed=seed)
        assert y_true.shape[0] == 17598, f"{model_key}: n_test={y_true.shape[0]} != 17598"
        obs = _observed_f1_pres(y_true, y_pred)
        f1_pres = _bootstrap_f1_pres(y_true, y_pred, n_boot, seed=seed)
        samples[model_key] = f1_pres
        results[model_key] = _summary(f1_pres, obs)

    return results, samples


def compute_all_pairs(results: dict, samples: dict) -> dict:
    """Toutes paires (A, B) avec A < B (ordre alphabétique) : bootstrap apparié."""
    models = sorted(results.keys())
    pairs: dict[str, dict] = {}
    for a, b in itertools.combinations(models, 2):
        key = f"{a}:{b}"
        sa, sb = samples[a], samples[b]
        assert sa.shape == sb.shape, f"shapes incompatibles : {sa.shape} vs {sb.shape}"
        a_s = results[a]
        b_s = results[b]
        delta = a_s["observed"] - b_s["observed"]
        p_a_gt_b = float(np.mean(sa > sb))
        disjoint = (a_s["ci95_low"] > b_s["ci95_high"]) or \
                   (b_s["ci95_low"] > a_s["ci95_high"])
        pairs[key] = {
            "model_a": a,
            "model_b": b,
            "delta_observed_a_minus_b": float(delta),
            "p_a_gt_b": p_a_gt_b,
            "ci95_a": [a_s["ci95_low"], a_s["ci95_high"]],
            "ci95_b": [b_s["ci95_low"], b_s["ci95_high"]],
            "ci95_disjoint": disjoint,
        }
    return pairs


# ------------------------------------------------------------------ heatmap

def _short_name(m: str) -> str:
    mapping = {
        "vitb16_imagenet": "ViT-B16\nImageNet",
        "dinov3_vitb16_lvd": "DINOv3\nViT-B16\nLVD",
        "simdinov2_vitb16": "SimDINOv2\nViT-B16",
        "dinov3_vitl16_lvd": "DINOv3\nViT-L16\nLVD",
        "simdinov2_vitl16": "SimDINOv2\nViT-L16",
        "resnet50_imagenet": "ResNet50\nImageNet",
        "dinov3_vitl16_sat": "DINOv3\nViT-L16\nSAT",
    }
    return mapping.get(m, m)


def plot_heatmap(models: list[str], pairs: dict, output_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    n = len(models)
    mat = np.full((n, n), np.nan)
    disjoint_mat = np.zeros((n, n), dtype=bool)
    idx = {m: i for i, m in enumerate(models)}

    for key, v in pairs.items():
        a, b = v["model_a"], v["model_b"]
        i, j = idx[a], idx[b]
        mat[i, j] = v["p_a_gt_b"]
        mat[j, i] = 1.0 - v["p_a_gt_b"]
        disjoint_mat[i, j] = v["ci95_disjoint"]
        disjoint_mat[j, i] = v["ci95_disjoint"]

    labels = [_short_name(m) for m in models]
    fig, ax = plt.subplots(figsize=(max(7, n * 1.6), max(6, n * 1.5)))
    im = ax.imshow(mat, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("modèle B", fontsize=10)
    ax.set_ylabel("modèle A", fontsize=10)
    ax.set_title("P(A > B)  |  f1_macro_pres  |  bootstrap apparié n=1000\n"
                 "* = IC95 disjoints (distinguishable)", fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="lightgrey", zorder=2))
                continue
            if np.isnan(mat[i, j]):
                continue
            val = mat[i, j]
            txt = f"{val:.2f}"
            if disjoint_mat[i, j]:
                txt += "*"
            color = "white" if abs(val - 0.5) > 0.3 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=color, fontweight="bold" if disjoint_mat[i, j] else "normal")

    plt.colorbar(im, ax=ax, label="P(ligne > colonne)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_png}")


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--probe-json", default="results/with_rhol/probe_knn_cgrid.json")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", default="results/significance_matrix.json")
    ap.add_argument("--output-png", default="results/significance_matrix.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    results, samples = run_bootstrap(cfg, args.probe_json, args.n_bootstrap, args.seed)

    if len(results) < 2:
        print("[STOP] Moins de 2 modèles bootstrappés — aucune paire possible.")
        return

    models = sorted(results.keys())
    pairs = compute_all_pairs(results, samples)

    out = {
        "models": models,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "metric": "f1_macro_pres",
        "probe_source": args.probe_json,
        "model_stats": results,
        "pairs": pairs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output_json}  ({len(models)} modèles, {len(pairs)} paires)")

    print("\nRésumé des paires IC95 disjoints :")
    for key, v in pairs.items():
        if v["ci95_disjoint"]:
            print(f"  {key}  Δ={v['delta_observed_a_minus_b']:+.4f}  "
                  f"P(A>B)={v['p_a_gt_b']:.3f}  DISJOINTS")

    plot_heatmap(models, pairs, args.output_png)


if __name__ == "__main__":
    main()
