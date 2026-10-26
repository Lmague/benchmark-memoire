#!/usr/bin/env python3
"""Phase 0.c — k-NN vs probe linéaire, et matrices de confusion.

1. `knn_vs_probe.csv` : pour les 12 modèles de `results/without_rhol/probe_knn_cgrid.json`,
   met en regard le probe linéaire canonique et le k-NN à k = 5/10/20/50 (test).
2. `confusion_matrices.npz` : matrices 11×11 (normalisées par ligne) pour 6 modèles clés
   à 100 %, obtenues en réajustant le probe canonique (lbfgs multinomial, `best_C` connu)
   sur les embeddings train du modèle puis en prédisant le test.

Le `best_C` est repris de la source canonique du modèle (probe_knn_cgrid.json pour les
gelés, metrics.json du run pour les fine-tunés) — aucun re-tuning ici, donc aucun risque
de diverger du chiffre canonique.

    python3 scripts/rapport/knn_and_confusion.py
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

# MONO-THREAD OBLIGATOIRE — pas un réglage de performance : le nombre de threads
# BLAS change l'ordre de réduction et donc la solution vers laquelle lbfgs
# converge (voir significance_tier.py, écart mesuré 0,0015 sur DINOv3-B MHSA
# seed0 entre 1 et 3 threads). Doit précéder l'import de numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from registry import CLASSES_11, OUT, ensure_out, p
from src.utils import make_canonical_lr

# Modèles pour lesquels on trace une matrice de confusion.
# (clé, libellé, train.npy, train_labels.npy, test.npy, test_labels.npy, best_C)
CONFUSION_MODELS = [
    ("dinov3_vitb16_lvd", "DINOv3 ViT-B/16 LVD (gelé)",
     "embeddings/dinov3_vitb16_lvd_train.npy", "embeddings/dinov3_vitb16_lvd_train_labels.npy",
     "embeddings/dinov3_vitb16_lvd_test.npy", "embeddings/dinov3_vitb16_lvd_test_labels.npy", None),
    ("dinov3_vitl16_lvd", "DINOv3 ViT-L/16 LVD (gelé)",
     "embeddings/dinov3_vitl16_lvd_train.npy", "embeddings/dinov3_vitl16_lvd_train_labels.npy",
     "embeddings/dinov3_vitl16_lvd_test.npy", "embeddings/dinov3_vitl16_lvd_test_labels.npy", None),
    ("dinov3_vitb16_lvd_lora_r8", "DINOv3 ViT-B/16 LoRA r=8",
     "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed0/train.npy",
     "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed0/train_labels.npy",
     "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed0/test.npy",
     "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed0/test_labels.npy", 0.001),
    ("vitb16_full_old", "ViT-B/16 IN-Full",
     "sota_screening/full/embeddings/vitb16_full_frac100_seed0/train.npy",
     "sota_screening/full/embeddings/vitb16_full_frac100_seed0/train_labels.npy",
     "sota_screening/full/embeddings/vitb16_full_frac100_seed0/test.npy",
     "sota_screening/full/embeddings/vitb16_full_frac100_seed0/test_labels.npy", None),
    ("vitb16_scratch_old", "ViT-B/16 IN-Scratch",
     "sota_screening/scratch/embeddings/vitb16_scratch_frac100_seed0/train.npy",
     "sota_screening/scratch/embeddings/vitb16_scratch_frac100_seed0/train_labels.npy",
     "sota_screening/scratch/embeddings/vitb16_scratch_frac100_seed0/test.npy",
     "sota_screening/scratch/embeddings/vitb16_scratch_frac100_seed0/test_labels.npy", None),
    ("scalemae_vitl16", "ScaleMAE ViT-L/16 (gelé)",
     "embeddings/scalemae_vitl16_train.npy", "embeddings/scalemae_vitl16_train_labels.npy",
     "embeddings/scalemae_vitl16_test.npy", "embeddings/scalemae_vitl16_test_labels.npy", None),
]

# Correspondance clé → run metrics.json (pour récupérer best_C des fine-tunés)
RUN_BEST_C = {
    "vitb16_full_old": "sota_screening/full/runs/frac100_seed0/metrics.json",
    "vitb16_scratch_old": "sota_screening/scratch/runs/frac100_seed0/metrics.json",
    "dinov3_vitb16_lvd_lora_r8": "results/datacurve_lora/runs/frac100_seed0/metrics.json",
}


def write_knn_vs_probe():
    d = json.load(open(p("results", "without_rhol", "probe_knn_cgrid.json")))
    probe, knn = d["probe"], d["knn"]
    rows = []
    for m in probe:
        rec = {"model": m, "probe_f1_pres": probe[m]["test"]["f1_macro_pres"],
               "probe_accuracy": probe[m]["test"]["accuracy"],
               "probe_f1_weighted": probe[m]["test"]["f1_weighted"],
               "probe_best_C": probe[m]["best_C"]}
        best_k, best_f1 = None, -1.0
        for kk, v in knn.get(m, {}).items():
            k = int(kk.split("=")[1])
            f1 = v["test"]["f1_macro_pres"]
            rec[f"knn_k{k}_f1"] = f1
            rec[f"knn_k{k}_acc"] = v["test"]["accuracy"]
            if f1 > best_f1:
                best_f1, best_k = f1, k
        rec["knn_best_k"] = best_k
        rec["knn_best_f1"] = best_f1
        rec["delta_probe_knn"] = rec["probe_f1_pres"] - best_f1
        rows.append(rec)
    rows.sort(key=lambda r: -r["probe_f1_pres"])
    cols = (["model", "probe_f1_pres", "probe_accuracy", "probe_f1_weighted", "probe_best_C"]
            + [f"knn_k{k}_{s}" for k in (5, 10, 20, 50) for s in ("f1", "acc")]
            + ["knn_best_k", "knn_best_f1", "delta_probe_knn"])
    with open(os.path.join(OUT, "knn_vs_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] knn_vs_probe.csv ({len(rows)} modèles)")


def resolve_best_c(key, default_from_probe):
    if key in RUN_BEST_C:
        return float(json.load(open(p(RUN_BEST_C[key])))["best_C"])
    d = json.load(open(p("results", "without_rhol", "probe_knn_cgrid.json")))["probe"]
    if key in d:
        return float(d[key]["best_C"])
    return default_from_probe


def write_confusions():
    mats, meta = {}, {}
    t0 = time.time()
    for key, label, trp, trl, tep, tel, forced_c in CONFUSION_MODELS:
        if not os.path.exists(p(trp)):
            print(f"  [SKIP] {key}: {trp} absent")
            continue
        C = forced_c if forced_c is not None else resolve_best_c(key, 0.001)
        Xtr = np.load(p(trp)).astype(np.float32)
        ytr = np.load(p(trl)).astype(np.int64)
        Xte = np.load(p(tep)).astype(np.float32)
        yte = np.load(p(tel)).astype(np.int64)
        # Le probe canonique standardise les features (src/probe.py::_standardize) :
        # sans cette étape le F1 refitté décroche de ~0,01 du chiffre canonique.
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = make_canonical_lr(C).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        present = np.unique(yte)
        f1 = float(f1_score(yte, pred, labels=present, average="macro", zero_division=0))
        cm = confusion_matrix(yte, pred, labels=present).astype(np.float64)
        cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
        mats[key] = cm_norm
        mats[key + "__counts"] = cm
        meta[key] = {"label": label, "best_C": C, "f1_macro_pres": f1,
                     "classes": [CLASSES_11[i] if len(present) == 11 else str(i)
                                 for i in range(len(present))],
                     "n_test": int(len(yte)), "dim": int(Xtr.shape[1]),
                     "n_train": int(len(Xtr))}
        print(f"  {key:28s} C={C:<7g} F1={f1:.4f}  ({time.time() - t0:.0f}s)")
    np.savez_compressed(os.path.join(OUT, "confusion_matrices.npz"), **mats)
    json.dump(meta, open(os.path.join(OUT, "confusion_meta.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"[OK] confusion_matrices.npz ({len(meta)} modèles)")


def main():
    ensure_out()
    print("== k-NN vs probe ==")
    write_knn_vs_probe()
    print("\n== Matrices de confusion (probe canonique refitté) ==")
    write_confusions()


if __name__ == "__main__":
    main()
