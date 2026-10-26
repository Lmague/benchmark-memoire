#!/usr/bin/env python3
"""Phase 0.d — Géométrie couche par couche (profondeur du réseau).

Embeddings disponibles localement :
  - `dinov3_vitl16_lvd` et `dinov3_vitl16_sat` : 24 couches, **test uniquement**
    → géométrie seulement (pas de probe possible sans embeddings train)
  - `vitb16_imagenet` : 12 couches train/val/test, mais le probe couche-par-couche
    a déjà été calculé (`_anciennes_experiences/v1_2026-06-12/layerwise_probe.json`)
    → on le recopie tel quel, étiqueté « archive 2026-06-12 », sans recalcul

Sortie : results/rapport_data/layerwise.csv
    (colonnes : model, layer, rankme_sigma_raw, eff_rank_sigma2, aniso_cosine,
     silhouette, knn_purity, nc1_canonical, f1_macro_pres, best_C, source)

    python3 scripts/rapport/layerwise_geometry.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from sklearn.metrics import silhouette_score

from registry import OUT, ensure_out, p
from src.latent import anisotropy as aniso_cosine
from scripts.geometry_extended import fisher_ratio, knn_purity
from scripts.compute_all_clustering_logme import nc1_metrics

SEED = 42
LAYERWISE_MODELS = [("dinov3_vitl16_lvd", "DINOv3 ViT-L/16 LVD", 24),
                    ("dinov3_vitl16_sat", "DINOv3 ViT-L/16 SAT", 24)]
ARCHIVE = "_anciennes_experiences/v1_2026-06-12/layerwise_probe.json"


def _entropy_rank(w, eps=1e-12):
    q = np.asarray(w, dtype=np.float64)
    q = q / (q.sum() + eps)
    return float(np.exp(-(q * np.log(q + eps)).sum()))


def geometry_layers():
    rows = []
    t0 = time.time()
    for key, display, n_layers in LAYERWISE_MODELS:
        lab_path = p("embeddings", f"{key}_layerwise_test_labels.npy")
        if not os.path.exists(lab_path):
            lab_path = p("embeddings", f"{key}_test_labels.npy")
        L = np.load(lab_path).astype(np.int64)
        for li in range(n_layers):
            fp = p("embeddings", f"{key}_test_layer{li:02d}.npy")
            if not os.path.exists(fp):
                continue
            E = np.load(fp).astype(np.float32)
            E64 = E.astype(np.float64)
            sig_raw = np.linalg.svd(E64, compute_uv=False)
            sig_cen = np.linalg.svd(E64 - E64.mean(0, keepdims=True), compute_uv=False)
            rows.append({
                "model": key, "display": display, "layer": li, "dim": int(E.shape[1]),
                "rankme_sigma_raw": _entropy_rank(sig_raw),
                "eff_rank_sigma2": _entropy_rank(sig_cen ** 2),
                "aniso_cosine": aniso_cosine(E, seed=SEED),
                "silhouette": float(silhouette_score(E64, L, random_state=SEED)),
                "knn_purity": knn_purity(E, L, seed=SEED),
                "fisher_ratio": fisher_ratio(E64, L),
                "nc1_canonical": nc1_metrics(E64, L),
                "f1_macro_pres": "", "best_C": "",
                "source": "recalcul 2026-07-25 (test, n=17598)",
            })
            print(f"  {key:20s} L{li:02d}  rk(σ)={rows[-1]['rankme_sigma_raw']:6.1f} "
                  f"aniso={rows[-1]['aniso_cosine']:.3f} "
                  f"sil={rows[-1]['silhouette']:+.4f} ({time.time() - t0:.0f}s)")
    return rows


def archive_layers():
    fp = p(ARCHIVE)
    if not os.path.exists(fp):
        print(f"  [SKIP] archive absente : {ARCHIVE}")
        return []
    d = json.load(open(fp))
    rows = []
    for key, entry in d["models"].items():
        for ls, v in sorted(entry["layers"].items(), key=lambda kv: int(kv[0])):
            rows.append({
                "model": key, "display": "ViT-B/16 IN-1k", "layer": int(ls),
                "dim": v.get("dim", ""),
                "rankme_sigma_raw": v.get("rankme", ""),
                "eff_rank_sigma2": "", "aniso_cosine": v.get("anisotropy", ""),
                "silhouette": "", "knn_purity": "", "fisher_ratio": "",
                "nc1_canonical": "",
                "f1_macro_pres": v.get("f1_macro_pres", ""),
                "best_C": v.get("best_C", ""),
                "source": "archive _anciennes_experiences/v1_2026-06-12",
            })
    print(f"  archive : {len(rows)} couches ({', '.join(d['models'])})")
    return rows


def main():
    ensure_out()
    print("== Géométrie par couche (recalcul) ==")
    rows = geometry_layers()
    print("\n== Probe par couche (archive) ==")
    rows += archive_layers()
    cols = ["model", "display", "layer", "dim", "rankme_sigma_raw", "eff_rank_sigma2",
            "aniso_cosine", "silhouette", "knn_purity", "fisher_ratio",
            "nc1_canonical", "f1_macro_pres", "best_C", "source"]
    with open(os.path.join(OUT, "layerwise.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n[OK] layerwise.csv ({len(rows)} lignes)")


if __name__ == "__main__":
    main()
