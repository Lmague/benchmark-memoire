#!/usr/bin/env python3
"""TÂCHE B — LogME (You et al., ICML 2021) sur les 12 modèles.

Réutilise les embeddings TRAIN (mêmes pooler output que le linear probe) +
labels d'entraînement en entiers.  Calcule le score LogME pour chaque
modèle, puis la corrélation Spearman + Kendall entre LogME (rang sur
train) et F1-macro-present (rang sur test) déjà mesuré.

Formule / source : You, Liu, Wang, Long. "LogME: Practical Assessment of
Pre-trained Models for Transfer Learning." ICML 2021.  Code vendored depuis
https://raw.githubusercontent.com/thuml/LogME/master/LogME.py (MIT) dans
``src/_vendor/LogME.py`` (numba dépouillé, formulation fixed-point inchangée).
API officielle (cf. README thuml/LogME) : ``LogME(regression=False).fit(F, y)``
avec F [N, D] float64 et y [N] entiers.

Sorties :
  - results/transfer/logme_scores.json  (par modèle)
  - results/transfer/logme_scores.csv   (flat)
  - results/transfer/logme_vs_f1.json   (corrélations)
  - results/transfer/logme_vs_f1.png|pdf  (scatter)

NOTE F1 : les valeurs de ``F1_PRES`` ci-dessous sont lues depuis la SOURCE
CANONIQUE ``results/with_rhol/probe_knn_cgrid.json`` (grille étendue,
F1≈0.4789 pour le co-leader).  L'ancien ``probe_knn.json`` (F1≈0.4675) est
périmé — NE PLUS L'UTILISER.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from src.transfer import logme_score

EMB = PROJ / "embeddings"
OUT = PROJ / "results" / "transfer"
FIG = PROJ / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    "resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
    "dinov3_vitl16_sat", "dinov3_vitl16_lvd", "simdinov2_vitb16",
    "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16",
    "resnet50_arctic", "vitb16_arctic", "vitb16_fulft_arctic",
]


def _load_f1_canonical() -> dict[str, float]:
    """Lit le F1 depuis la source canonique ``probe_knn_cgrid.json``.

    C'est la source UNIQUE de F1 dans le repo (méthodo canonique).  L'ancien
    ``probe_knn.json`` (grille restreinte C∈{0.01,0.1,1,10}, F1≈0.4675) est
    marqué .deprecated et NE DOIT PAS être utilisé pour les corrélations.
    """
    pk = PROJ / "results" / "with_rhol" / "probe_knn_cgrid.json"
    with open(pk) as f:
        d = json.load(f)
    probe = d.get("probe", d)
    return {m: float(probe[m]["test"]["f1_macro_pres"])
            for m in probe if "test" in probe[m]}


F1_PRES = _load_f1_canonical()


def compute_logme(key: str) -> dict:
    """Charge F_train, y_train, calcule le score LogME et le temps CPU."""
    F = np.load(EMB / f"{key}_train.npy").astype(np.float32)
    y = np.load(EMB / f"{key}_train_labels.npy").astype(np.int64)
    t0 = time.time()
    s = logme_score(F, y, regression=False)
    dt = time.time() - t0
    return {
        "model": key, "dim": int(F.shape[1]), "n_train": int(F.shape[0]),
        "logme": float(s), "logme_time_s": round(dt, 2),
        "f1_macro_pres_test": F1_PRES[key],
    }


def scatter_logme_vs_f1(rows: list[dict]) -> None:
    """Scatter LogME (x) vs F1-macro-present (y), annotés par modèle."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = np.array([r["logme"] for r in rows])
    ys = np.array([r["f1_macro_pres_test"] for r in rows])
    ax.scatter(xs, ys, s=70, c="#1f77b4", edgecolor="black", alpha=0.85, zorder=3)
    for r in rows:
        ax.annotate(r["model"], (r["logme"], r["f1_macro_pres_test"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points", alpha=0.85)
    rs, ps = spearmanr(xs, ys)
    kt, kp = kendalltau(xs, ys)
    ax.set_xlabel("LogME (train, 49 433 tuiles) — score", fontsize=11)
    ax.set_ylabel(r"F1-macro-present (test, 17 598 tuiles)", fontsize=11)
    ax.set_title(f"LogME vs F1 test — 12 modèles\n"
                 f"Spearman ρ={rs:+.3f} (p={ps:.2e})  "
                 f"Kendall τ={kt:+.3f} (p={kp:.2e})", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "logme_vs_f1.png", dpi=150)
    fig.savefig(FIG / "logme_vs_f1.pdf")
    plt.close(fig)
    print(f"[fig] -> {FIG/'logme_vs_f1.png'}")


def main() -> None:
    print("=" * 70)
    print("TÂCHE B — LogME (12 modèles, embeddings train, label-entiers)")
    print("=" * 70)
    rows = []
    for k in MODELS:
        try:
            r = compute_logme(k)
            rows.append(r)
            print(f"[{k:24s}] dim={r['dim']:5d}  LogME={r['logme']:+.4f}  "
                  f"({r['logme_time_s']:5.1f}s)  F1={r['f1_macro_pres_test']:.4f}")
        except Exception as exc:
            print(f"[{k:24s}] ERREUR : {exc!r}")

    # CSV
    csv_path = OUT / "logme_scores.csv"
    with open(csv_path, "w") as f:
        f.write("model,dim,n_train,logme,logme_time_s,f1_macro_pres_test\n")
        for r in rows:
            f.write(f"{r['model']},{r['dim']},{r['n_train']},"
                    f"{r['logme']},{r['logme_time_s']},{r['f1_macro_pres_test']}\n")
    print(f"\n[csv] -> {csv_path}")

    # JSON
    json_path = OUT / "logme_scores.json"
    with open(json_path, "w") as f:
        json.dump({"per_model": rows,
                   "source_code": "src/_vendor/LogME.py (vendored thuml/LogME@master)",
                   "source_paper": "You et al. ICML 2021, JMLR 2022 (fixed-point formulation)",
                   "split": "train", "labels_int": True, "regression": False,
                   "f1_source": "results/with_rhol/probe_knn_cgrid.json (f1_macro_pres canonique)"}, f, indent=2)
    print(f"[json] -> {json_path}")

    # Corrélations
    xs = np.array([r["logme"] for r in rows])
    ys = np.array([r["f1_macro_pres_test"] for r in rows])
    rs, ps = spearmanr(xs, ys)
    kt, kp = kendalltau(xs, ys)
    corr_path = OUT / "logme_vs_f1.json"
    corr = {
        "n": len(rows),
        "spearman_r": float(rs), "spearman_p": float(ps),
        "kendall_tau": float(kt), "kendall_p": float(kp),
        "logme_top3": [r["model"] for r in sorted(rows, key=lambda r: -r["logme"])[:3]],
        "f1_top3":     [r["model"] for r in sorted(rows, key=lambda r: -r["f1_macro_pres_test"])[:3]],
        "note": "LogME est un outil de RANG (valeur absolue non interprétable)."
    }
    with open(corr_path, "w") as f:
        json.dump(corr, f, indent=2)
    print(f"\n[corr] Spearman ρ={rs:+.3f} (p={ps:.2e})  "
          f"Kendall τ={kt:+.3f} (p={kp:.2e})  (n={len(rows)})")
    print(f"[json] -> {corr_path}")

    # Scatter
    scatter_logme_vs_f1(rows)

    # Concordance top-3
    print("\nLogME top-3  :", corr["logme_top3"])
    print("F1    top-3  :", corr["f1_top3"])

    print("=" * 70)
    print("TÂCHE B terminée.")
    print("=" * 70)


if __name__ == "__main__":
    main()
