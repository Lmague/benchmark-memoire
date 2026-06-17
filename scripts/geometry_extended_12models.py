#!/usr/bin/env python3
"""Géométrie de l'espace latent recalculée sur les 12 modèles (9 frozen + 3 FT).

Les embeddings des 3 modèles fine-tunés (resnet50_arctic, vitb16_arctic,
vitb16_fulft_arctic) sont désormais disponibles localement : on applique donc le
MÊME protocole que pour les frozen (``scripts.geometry_extended.compute_all`` :
sous-échantillon 20 000 pour le spectre, 5 000 pour la k-NN purity, graine 42).

Validation : la colonne ``stable_rank`` produite ici doit coïncider à 1e-3 près avec
la colonne NESum du Tableau « Métriques spectrales » du rapport (stable rank ≡ NESum
= Σλ_i/λ_1). Écrit ``results/geometry_extended_12models.json``.

    python scripts/geometry_extended_12models.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.geometry_extended import compute_all

EMB = os.path.join(_ROOT, "embeddings")
MODELS = [
    "resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd", "simdinov2_vitb16",
    "dinov3_vitl16_sat", "dinov3_vitl16_lvd", "simdinov2_vitl16", "satmae_vitl16",
    "scalemae_vitl16", "resnet50_arctic", "vitb16_arctic", "vitb16_fulft_arctic",
]


def main() -> None:
    out = {}
    for k in MODELS:
        te = os.path.join(EMB, f"{k}_test.npy")
        lb = os.path.join(EMB, f"{k}_test_labels.npy")
        if not (os.path.exists(te) and os.path.exists(lb)):
            print(f"[SKIP] {k}: embeddings/labels test absents")
            continue
        E = np.load(te).astype(np.float32)
        L = np.load(lb)
        m = compute_all(E, L, subsample_n=20000, seed=42)
        out[k] = m
        print(f"{k:24} RankMe={m['rankme']:7.1f} SR={m['stable_rank']:6.2f} "
              f"PR={m['participation_ratio']:6.2f} kNNpur={m['knn_purity']:.3f}")

    dst = os.path.join(_ROOT, "results", "geometry_extended_12models.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[SAVED] {dst} ({len(out)} modèles)")


if __name__ == "__main__":
    main()
