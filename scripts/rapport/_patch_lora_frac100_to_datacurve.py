#!/usr/bin/env python3
"""Ajoute les 3 lignes frac=1.0 de LoRA r=8 dans geometry_datacurve.csv.

Pourquoi : le run à 100 % vit dans DINOv3_LoRA_8/embeddings/..._explora_frac100_seed*
(même dossiers que ceux servis à model_jobs()) et la campagne multi-fractions
couvre 0.5%–70% (DINOv3_LoRA/lora/embeddings/, lue par datacurve_jobs() via
registry.RUN_FAMILIES). geometry_datacurve.csv ne contient donc que 7 fractions
(0,5~%–70~%) pour ce régime tant que ce script n'a pas tourné.

Révision 2026-07-29 : réécrit pour calculer directement les métriques par seed
avec `compute()` de geometry_test_fixed_n.py (même protocole que tout le reste
du document), au lieu de copier geometry_models.csv --- ce fichier n'a plus qu'une
ligne par modèle (moyenne inter-seed, colonnes `_std`) depuis que
`run_models()` a été réécrit, et n'a plus de colonne `seed` par ligne.

Idempotent : on (model, fraction, seed) vérifie avant d'écrire.
"""
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from geometry_test_fixed_n import compute
from registry import OUT, p

DATACURVE = os.path.join(OUT, "geometry_datacurve.csv")
TARGET_KEY = "dinov3_vitb16_lvd_lora_r8"
EMB_BASE = "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100"

with open(DATACURVE) as f:
    r = csv.DictReader(f)
    existing = {(row["model"], row["fraction"], row["seed"]) for row in r}
    fieldnames_dc = r.fieldnames

new_rows = []
for seed in (0, 1, 2):
    key = (TARGET_KEY, "1.0", str(seed))
    if key in existing:
        continue
    d = p(f"{EMB_BASE}_seed{seed}")
    ep, lp = os.path.join(d, "test.npy"), os.path.join(d, "test_labels.npy")
    if not (os.path.exists(ep) and os.path.exists(lp)):
        print(f"[SKIP] seed {seed} : embeddings absents ({d})")
        continue
    m = compute(np.load(ep).astype(np.float32), np.load(lp).astype(np.int64))
    m.update(model=TARGET_KEY, fraction=1.0, seed=seed,
             emb_dir=os.path.relpath(d, p()))
    new_rows.append(m)
    print(f"  seed {seed}: rk(σ)={m['rankme_sigma_raw']:.1f} "
          f"eff(σ²)={m['eff_rank_sigma2']:.1f} sil={m['silhouette']:+.4f}")

if not new_rows:
    print("[OK] rien à faire (toutes les lignes frac=1.0 déjà présentes ou embeddings absents)")
    sys.exit(0)

with open(DATACURVE) as f:
    existing_rows = list(csv.DictReader(f))

with open(DATACURVE, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames_dc)
    w.writeheader()
    for row in existing_rows:
        w.writerow({k: row.get(k, "") for k in fieldnames_dc})
    for m in new_rows:
        w.writerow({k: m.get(k, "") for k in fieldnames_dc})

print(f"[OK] {len(new_rows)} ligne(s) frac=1.0 ajoutee(s) a geometry_datacurve.csv")
