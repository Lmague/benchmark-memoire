#!/usr/bin/env python3
"""Probe canonique pour les 13 bras de la Stage A SimDINOv2-B (2026-09) + NormTuning.

Même protocole que scripts/probe_lora3_new_runs.py (LR lbfgs multinomial, C_grid
[1e-4..10] avec sous-échantillon 20k pour la sélection de C, refit 100 %,
best_C sur val, seed=42, max_iter=2000, BLAS mono-thread §4.8) : les F1 obtenus
sont directement comparables aux autres entrées LoRA de
results/all_models_canonical_merged.json.

Entrées :
  results/lora_simb_ablation/embeddings/<tag>_seed{N}/{train,val,test}[_labels].npy
  results/lora_simb_ablation_qkv/embeddings/<tag>_seed{N}/...   (bras QKV seul)

Écrit results/simb_stageA_probe_CANONICAL.json puis fusionne (append idempotent,
jamais d'écrasement) dans results/all_models_canonical_merged.json — même
convention que probe_lora3_new_runs.py. La géométrie reste à None (elle vient de
scripts/rapport/geometry_test_fixed_n.py, protocole à n fixe).

    python3 scripts/rapport/probe_simb_stageA_canonical.py [--workers N]

Mémoire : ~1 Go par worker (train 49k×768). 3 workers par défaut (machine 13 Go).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[2]  # scripts/rapport/ -> racine du dépôt
sys.path.insert(0, str(PROJ / "scripts"))

SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
MAX_ITER = 2000
CGRID_SUBSAMPLE = 20000

# (clé registre, display, tag_stem_embeddings, out_dir_suffix)
ARMS = [
    ("simdinov2_vitb16_lora_r8_b611", "SimDINOv2-B LoRA r8 (blocs 6-11)",
     "simdinov2_vitb16_lora_r8a8_b67891011_frac100", ""),
    ("simdinov2_vitb16_lora_r8_b05", "SimDINOv2-B LoRA r8 (blocs 0-5)",
     "simdinov2_vitb16_lora_r8a8_b012345_frac100", ""),
    ("simdinov2_vitb16_lora_r8_b911", "SimDINOv2-B LoRA r8 (blocs 9-11)",
     "simdinov2_vitb16_lora_r8a8_b91011_frac100", ""),
    ("simdinov2_vitb16_lora_r2", "SimDINOv2-B LoRA r2",
     "simdinov2_vitb16_lora_r2a2_frac100", ""),
    ("simdinov2_vitb16_lora_r4", "SimDINOv2-B LoRA r4",
     "simdinov2_vitb16_lora_r4a4_frac100", ""),
    ("simdinov2_vitb16_lora_r16", "SimDINOv2-B LoRA r16",
     "simdinov2_vitb16_lora_r16a16_frac100", ""),
    ("simdinov2_vitb16_lora_r32", "SimDINOv2-B LoRA r32",
     "simdinov2_vitb16_lora_r32a32_frac100", ""),
    ("simdinov2_vitb16_lora_r8_s2", "SimDINOv2-B LoRA r8 (scaling 2)",
     "simdinov2_vitb16_lora_r8a16_frac100", ""),
    ("simdinov2_vitb16_lora_r16_s2", "SimDINOv2-B LoRA r16 (scaling 2)",
     "simdinov2_vitb16_lora_r16a32_frac100", ""),
    ("simdinov2_vitb16_lora_r8_rslora", "SimDINOv2-B LoRA r8 (rsLoRA, scaling √8)",
     "simdinov2_vitb16_lora_r8a22_frac100", ""),
    ("simdinov2_vitb16_lora_r16_rslora", "SimDINOv2-B LoRA r16 (rsLoRA, scaling √16)",
     "simdinov2_vitb16_lora_r16a64_frac100", ""),
    ("simdinov2_vitb16_lora_r8_qkv", "SimDINOv2-B LoRA r8 (Q+K+V)",
     "simdinov2_vitb16_lora_r8a8_frac100", "_qkv"),
    ("simdinov2_vitb16_norm_tuning", "SimDINOv2-B NormTuning",
     "simdinov2_vitb16_norm_tuning_frac100", ""),
]


def _make_lr(c):
    try:
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                  random_state=SEED, multi_class="multinomial")
    except (TypeError, ValueError):
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                  random_state=SEED)


def probe_one(stem, out_suffix, seed_i):
    base = PROJ / f"results/lora_simb_ablation{out_suffix}/embeddings/{stem}_seed{seed_i}"
    data = {}
    for s in ("train", "val", "test"):
        e = np.load(base / f"{s}.npy").astype(np.float32)
        l = np.load(base / f"{s}_labels.npy").astype(np.int64).ravel()
        data[s] = (e, l)
    etr, ltr = data["train"]
    eva, lva = data["val"]
    ete, lte = data["test"]
    n_classes = len(np.unique(np.concatenate([data[s][1] for s in data])))
    sc = StandardScaler()
    xtr_full = sc.fit_transform(np.asarray(etr, dtype=np.float32))
    xva = sc.transform(np.asarray(eva, dtype=np.float32))
    xte = sc.transform(np.asarray(ete, dtype=np.float32))
    rng = np.random.RandomState(SEED)
    idx = rng.choice(xtr_full.shape[0], CGRID_SUBSAMPLE, replace=False)
    xtr_cg, ltr_cg = xtr_full[idx], ltr[idx]
    labels_range = list(range(n_classes))
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        clf = _make_lr(c)
        clf.fit(xtr_cg, ltr_cg)
        f1v = f1_score(lva, clf.predict(xva), average="macro",
                       zero_division=0, labels=labels_range)
        if f1v > best_f1:
            best_f1, best_c = f1v, c
    clf = _make_lr(best_c)
    clf.fit(xtr_full, ltr)
    f1 = float(f1_score(lte, clf.predict(xte), average="macro",
                        zero_division=0, labels=labels_range))
    return f1, best_c


def run_arm(entry):
    key, display, stem, suffix = entry
    seeds_f1, best_cs = [], []
    dim = n_train = n_test = n_classes = None
    for seed_i in (0, 1, 2):
        base = PROJ / f"results/lora_simb_ablation{suffix}/embeddings/{stem}_seed{seed_i}"
        probe_test = np.load(base / "test.npy", mmap_mode="r")
        dim, n_test = probe_test.shape[1], probe_test.shape[0]
        n_train = np.load(base / "train.npy", mmap_mode="r").shape[0]
        n_classes = 11
        t = time.time()
        f1, best_c = probe_one(stem, suffix, seed_i)
        seeds_f1.append(f1)
        best_cs.append(best_c)
        print(f"[{key} seed{seed_i}] F1={f1:.4f} C={best_c} ({time.time()-t:.0f}s)",
              flush=True)
    return key, {
        "model": key, "type": "ft_fresh", "display": display,
        "f1_linear_probe": float(np.mean(seeds_f1)),
        "f1_std": float(np.std(seeds_f1, ddof=1)),
        "f1_seeds": seeds_f1,
        "best_C": best_cs,
        "silhouette_score": None, "silhouette_std": None,
        "nc1": None, "nc2_deviation_etf": None,
        "dim": int(dim),
        "n_train": int(n_train), "n_test": int(n_test),
        "n_classes": n_classes, "n_seeds": 3,
        "f1_macro_all": float(np.mean(seeds_f1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=None,
                    help="restreindre à certaines clés (debug/calibration)")
    args = ap.parse_args()
    t0 = time.time()
    arms = [a for a in ARMS if args.arms is None or a[0] in args.arms]
    print(f"== {len(arms)} bras × 3 seeds, {args.workers} workers ==", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for key, rec in ex.map(run_arm, arms):
            results[key] = rec
    out = PROJ / "results/simb_stageA_probe_CANONICAL.json"
    with open(out, "w") as f:
        json.dump({"pipeline": "LR lbfgs multinomial, C_grid [1e-4..10] (sélection "
                               "sur sous-échantillon 20k, refit 100 %), seed=42, "
                               "max_iter=2000, BLAS mono-thread (AGENTS.md §4.8)",
                   "schema": "11cls (RHOL exclue)", "date": time.strftime("%Y-%m-%d"),
                   "models": list(results.values())}, f, indent=2)
    print(f"\n-> {out}")

    merged_path = PROJ / "results/all_models_canonical_merged.json"
    with open(merged_path) as f:
        merged = json.load(f)
    existing = {m["model"] for m in merged["models"]}
    added = 0
    for k, r in results.items():
        if k in existing:
            print(f"[SKIP] {k} déjà présent")
            continue
        merged["models"].append(r)
        added += 1
    merged["models"].sort(key=lambda m: -m["f1_linear_probe"])
    merged["n_models"] = len(merged["models"])
    merged["date"] = time.strftime("%Y-%m-%d")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[OK] {added} modèles ajoutés (total={merged['n_models']})")
    print(f"\n{'='*60}\nRÉSUMÉ canonique\n{'='*60}")
    for k, r in sorted(results.items(), key=lambda kv: -kv[1]["f1_linear_probe"]):
        print(f"  {k:34s} {r['f1_linear_probe']:.4f} ± {r['f1_std']:.4f}  "
              f"{[f'{x:.4f}' for x in r['f1_seeds']]}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
