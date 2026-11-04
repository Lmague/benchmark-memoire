#!/usr/bin/env python3
"""Probe canonique de l'ablation de bloc LoRA DINOv3-B (blocs 0-5 vs 6-11, r=8 × 3 seeds).

Même protocole que scripts/rapport/lora_rank_ablation_probe.py (lui-même calqué sur
scripts/probe_lora3_new_runs.py) : LogisticRegression lbfgs multinomial, C_grid
[1e-4..10], best_C sur validation (sous-échantillon du train), fit final sur tout
le train, seed=42, max_iter=2000, BLAS mono-thread forcé AVANT import numpy/sklearn
(AGENTS.md §4.8 — obligatoire pour tout chiffre canonique reproductible).

Les embeddings sont ceux de lora_block_ablation/embeddings/
dinov3_vitb16_lvd_lora_r8a16_{b012345,b67891011}_frac100_seed{s}/
(dim=768, 11 classes sans RHOL). Les deux configs bloquent LoRA sur une moitié des
12 blocs d'attention DINOv3-B : b012345 = blocs 0-5 (bas du réseau), b67891011 =
blocs 6-11 (haut du réseau) — à comparer au r=8 publié (Q/V sur les 12 blocs,
F1=0,4835±0,0011) pour voir si la profondeur d'adaptation compte autant que le rang.

Le « best_epoch » reporté vient du metrics.json du run (probe interne), car le
probe canonique (linéaire, sans entraînement du backbone) n'a pas d'époque.

Écrit results/lora_block_ablation_CANONICAL.json.

    python3 scripts/rapport/lora_block_ablation_probe.py
    python3 scripts/rapport/lora_block_ablation_probe.py --only b012345 0   # 1 seed de test
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

BLOCKS = {
    "b012345":   {"blocs": list(range(0, 6)), "label": "Blocs 0-5 (bas du réseau)"},
    "b67891011": {"blocs": list(range(6, 12)), "label": "Blocs 6-11 (haut du réseau)"},
}
RANK, ALPHA = 8, 16
SEEDS = [0, 1, 2]

EMB_ROOT = PROJ / "lora_block_ablation" / "embeddings"
RUN_ROOT = PROJ / "lora_block_ablation" / "runs"


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


def emb_dir_of(tag, s):
    return EMB_ROOT / f"dinov3_vitb16_lvd_lora_r{RANK}a{ALPHA}_{tag}_frac100_seed{s}"


def run_dir_of(tag, s):
    return RUN_ROOT / f"dinov3_vitb16_lvd_lora_r{RANK}a{ALPHA}_{tag}_frac100_seed{s}"


def load_best_epoch(tag, s):
    m = json.load(open(run_dir_of(tag, s) / "metrics.json"))
    return int(m["best_epoch"]), m.get("f1_macro_pres_test"), m.get("f1_macro_pres_val")


def probe_one(tag, s):
    emb_dir = emb_dir_of(tag, s)
    data = {}
    for split in ("train", "val", "test"):
        e = np.load(emb_dir / f"{split}.npy").astype(np.float32)
        l = np.load(emb_dir / f"{split}_labels.npy").astype(np.int64).ravel()
        data[split] = (e, l)
    n_classes = len(np.unique(np.concatenate([data[x][1] for x in data])))
    f1, best_c = linear_probe_f1(data, n_classes)
    best_epoch, f1_internal, f1_val_internal = load_best_epoch(tag, s)
    return {"f1": f1, "best_C": best_c, "best_epoch": best_epoch,
            "f1_internal_test": f1_internal, "f1_internal_val": f1_val_internal,
            "n_classes": n_classes,
            "n_train": int(data["train"][0].shape[0]),
            "n_val": int(data["val"][0].shape[0]),
            "n_test": int(data["test"][0].shape[0]),
            "dim": int(data["test"][0].shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs=2, metavar=("TAG", "SEED"), default=None,
                    help="probe un seul (tag, seed), ex. --only b012345 0")
    args = ap.parse_args()

    t0 = time.time()
    results = {"blocks": {}, "baselines": {}, "protocol": (
        "LR lbfgs multinomial, C_grid [1e-4..10], best_C sur val "
        "(sous-échantillon 20000 du train), fit final sur tout le train, "
        "seed=42, max_iter=2000, BLAS mono-thread forcé avant import "
        "numpy/sklearn (AGENTS.md §4.8)")}

    if args.only:
        tag, s = args.only[0], int(args.only[1])
        out = probe_one(tag, s)
        print(f"[{tag} seed={s}] F1={out['f1']:.4f} C={out['best_C']} "
              f"best_epoch={out['best_epoch']} (interne {out['f1_internal_test']:.4f})")
        return

    for tag, info in BLOCKS.items():
        seeds_f1, best_Cs, best_epochs = [], [], []
        for s in SEEDS:
            out = probe_one(tag, s)
            seeds_f1.append(out["f1"])
            best_Cs.append(out["best_C"])
            best_epochs.append(out["best_epoch"])
            print(f"[{tag} seed={s}] F1={out['f1']:.4f} C={out['best_C']} "
                  f"epoch={out['best_epoch']} interne={out['f1_internal_test']:.4f} "
                  f"dim={out['dim']} n_train={out['n_train']} n_test={out['n_test']}")
        results["blocks"][tag] = {
            "tag": tag, "blocs": info["blocs"], "label": info["label"],
            "rank": RANK, "alpha": ALPHA,
            "params_lora_trainable": 2 * 768 * RANK * 2 * len(info["blocs"]),  # A+B × dim × r × (q,v) × #blocs
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
                                                 "label": "DINOv3-B LoRA r=8, Q/V sur 12 blocs (publié)"},
    }

    out_path = PROJ / "results" / "lora_block_ablation_CANONICAL.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n-> {out_path}")

    print(f"\n{'='*64}\nRÉSUMÉ canonique ablation de bloc (F1 probe, 3 seeds)\n{'='*64}")
    for tag, info in BLOCKS.items():
        d = results["blocks"][tag]
        print(f"  {tag:11s} ({info['label']:26s})  {d['f1_mean']:.4f} ± {d['f1_std']:.4f}  "
              f"{[f'{x:.4f}' for x in d['seeds']]}  best_C={d['best_Cs']}  "
              f"epochs={d['best_epochs']}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
