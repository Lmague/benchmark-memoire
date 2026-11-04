#!/usr/bin/env python3
"""Probe canonique de l'ablation de rang LoRA DINOv3-B (r=2,4,8,16,32 × 3 seeds).

Même protocole que scripts/probe_lora3_new_runs.py : LogisticRegression lbfgs
multinomial, C_grid [1e-4..10], best_C sur validation (sous-échantillon du train),
fit final sur tout le train, seed=42, max_iter=2000, BLAS mono-thread forcé AVANT
import numpy/sklearn (AGENTS.md §4.8 — obligatoire pour tout chiffre canonique
reproductible).

Les embeddings sont ceux de LoRA_ablation/embeddings/
dinov3_vitb16_lvd_lora_r{a}b{b}_frac100_seed{s}/ (dim=768, 11 classes sans RHOL).
Le « best_epoch » reporté vient du metrics.json du run (probe interne), car le
probe canonique (linéaire, sans entraînement du backbone) n'a pas d'époque.

Écrit results/lora_rank_ablation_CANONICAL.json.

    python3 scripts/rapport/lora_rank_ablation_probe.py
    python3 scripts/rapport/lora_rank_ablation_probe.py --only 8 0   # 1 seed de test
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
MAX_ITER = 2000
CGRID_SUBSAMPLE = 20000

RANKS = [2, 4, 8, 16, 32]          # r ; alpha = 2*r (b = 2a dans le nommage)
ALPHA = {r: 2 * r for r in RANKS}
SEEDS = [0, 1, 2]

EMB_ROOT = PROJ / "LoRA_ablation" / "embeddings"
RUN_ROOT = PROJ / "LoRA_ablation" / "runs"


def _make_lr(c):
    try:
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                   random_state=SEED, multi_class="multinomial")
    except (TypeError, ValueError):
        # sklearn >= 1.8 : multi_class supprimé, lbfgs est déjà multinomial.
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                   random_state=SEED)


def linear_probe_f1(data, n_classes):
    etr, ltr = data["train"]; eva, lva = data["val"]; ete, lte = data["test"]
    sc = StandardScaler()
    xtr_full = sc.fit_transform(np.asarray(etr, dtype=np.float32))
    xva = sc.transform(np.asarray(eva, dtype=np.float32))
    xte = sc.transform(np.asarray(ete, dtype=np.float32))
    if xtr_full.shape[0] > CGRID_SUBSAMPLE:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(xtr_full.shape[0], CGRID_SUBSAMPLE, replace=False)
        xtr_cg, ltr_cg = xtr_full[idx], ltr[idx]
    else:
        xtr_cg, ltr_cg = xtr_full, ltr
    labels_range = list(range(n_classes))
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        clf = _make_lr(c); clf.fit(xtr_cg, ltr_cg)
        f1v = f1_score(lva, clf.predict(xva), average="macro", zero_division=0,
                       labels=labels_range)
        if f1v > best_f1:
            best_f1, best_c = f1v, c
    clf = _make_lr(best_c); clf.fit(xtr_full, ltr)
    yp = clf.predict(xte)
    f1 = float(f1_score(lte, yp, average="macro", zero_division=0,
                        labels=labels_range))
    return f1, best_c


def emb_dir_of(r, s):
    return EMB_ROOT / f"dinov3_vitb16_lvd_lora_r{r}a{ALPHA[r]}_frac100_seed{s}"


def run_dir_of(r, s):
    return RUN_ROOT / f"dinov3_vitb16_lvd_lora_r{r}a{ALPHA[r]}_frac100_seed{s}"


def load_best_epoch(r, s):
    m = json.load(open(run_dir_of(r, s) / "metrics.json"))
    return int(m["best_epoch"]), m.get("f1_macro_pres_test"), m.get("f1_macro_pres_val")


def probe_one(r, s):
    emb_dir = emb_dir_of(r, s)
    data = {}
    for split in ("train", "val", "test"):
        e = np.load(emb_dir / f"{split}.npy").astype(np.float32)
        l = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64).ravel()
        data[split] = (e, l)
    n_classes = len(np.unique(np.concatenate([data[x][1] for x in data])))
    f1, best_c = linear_probe_f1(data, n_classes)
    best_epoch, f1_internal, f1_val_internal = load_best_epoch(r, s)
    return {"f1": f1, "best_C": best_c, "best_epoch": best_epoch,
            "f1_internal_test": f1_internal, "f1_internal_val": f1_val_internal,
            "n_classes": n_classes,
            "n_train": int(data["train"][0].shape[0]),
            "n_val": int(data["val"][0].shape[0]),
            "n_test": int(data["test"][0].shape[0]),
            "dim": int(data["test"][0].shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=2, type=int, default=None,
                    help="probe un seul (r, seed), ex. --only 8 0")
    args = ap.parse_args()

    t0 = time.time()
    results = {"ranks": {}, "baselines": {}, "protocol": (
        "LR lbfgs multinomial, C_grid [1e-4..10], best_C sur val "
        "(sous-échantillon 20000 du train), fit final sur tout le train, "
        "seed=42, max_iter=2000, BLAS mono-thread forcé avant import "
        "numpy/sklearn (AGENTS.md §4.8)")}

    if args.only:
        r, s = args.only
        out = probe_one(r, s)
        print(f"[r={r} seed={s}] F1={out['f1']:.4f} C={out['best_C']} "
              f"best_epoch={out['best_epoch']} (interne {out['f1_internal_test']:.4f})")
        return

    for r in RANKS:
        seeds_f1, best_Cs, best_epochs = [], [], []
        for s in SEEDS:
            out = probe_one(r, s)
            seeds_f1.append(out["f1"])
            best_Cs.append(out["best_C"])
            best_epochs.append(out["best_epoch"])
            print(f"[r={r:2d} seed={s}] F1={out['f1']:.4f} C={out['best_C']} "
                  f"epoch={out['best_epoch']} interne={out['f1_internal_test']:.4f} "
                  f"dim={out['dim']} n_train={out['n_train']} n_test={out['n_test']}")
        results["ranks"][r] = {
            "rank": r, "alpha": ALPHA[r],
            "params_lora_trainable": 2 * 768 * r * 2 * 12,  # A+B × dim × r × (q,v) × 12 blocs
            "f1_mean": float(np.mean(seeds_f1)),
            "f1_std": float(np.std(seeds_f1, ddof=1)),
            "seeds": seeds_f1,
            "best_Cs": best_Cs,
            "best_epochs": best_epochs,
            "best_epoch_mean": float(np.mean(best_epochs)),
        }

    # Baselines canoniques du registry (scripts/rapport/registry.py CANONICAL_F1)
    results["baselines"] = {
        "dinov3_vitb16_lvd": {"f1": 0.4712, "label": "DINOv3-B gelé"},
        "dinov3_vitb16_lvd_full": {"f1": 0.4786, "std": 0.0027, "label": "DINOv3-B Full FT"},
        "dinov3_vitb16_lvd_mhsa": {"f1": 0.4800, "std": 0.0020, "label": "DINOv3-B MHSA"},
        "dinov3_vitb16_lvd_lora_r8_published": {"f1": 0.4835, "std": 0.0011,
                                                 "label": "DINOv3-B LoRA r=8 (publié)"},
        "dinov3_vith16plus_lvd": {"f1": 0.4805, "label": "DINOv3 ViT-H+/16 gelé"},
    }

    out_path = PROJ / "results" / "lora_rank_ablation_CANONICAL.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {out_path}")

    print(f"\n{'='*64}\nRÉSUMÉ canonique ablation de rang (F1 probe, 3 seeds)\n{'='*64}")
    for r in RANKS:
        d = results["ranks"][r]
        print(f"  r={r:2d} (α={ALPHA[r]:2d})  {d['f1_mean']:.4f} ± {d['f1_std']:.4f}  "
              f"{[f'{x:.4f}' for x in d['seeds']]}  best_C={d['best_Cs']}  "
              f"epochs={d['best_epochs']}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
