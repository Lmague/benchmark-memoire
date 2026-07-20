#!/usr/bin/env python3
"""Ensembling par moyenne d'embeddings sur les 3 seeds ExPLoRA (DINOv3-B), probe canonique
sur l'embedding moyenné, comparé aux 3 seeds individuels.

Idée testée : pour chaque tuile, on a 3 embeddings indépendants (un par seed d'entraînement
ExPLoRA — même config, même données, seed de départ différent). On moyenne ces 3 embeddings
tuile par tuile (AVANT standardisation) pour voir si le signal partagé entre seeds ressort
mieux que le bruit spécifique à chaque run — c'est un ensembling au niveau représentation,
pas au niveau prédiction (pas de vote/moyenne de logits).

Prérequis : les 3 seeds d'embeddings finaux ExPLoRA existent déjà sur le disque
(sota_screening/dinov3_vitb16_lvd_explora/embeddings/dinov3_vitb16_lvd_explora_frac100_seed{0,1,2}/).
AUCUNE réextraction nécessaire — ce script consomme ce qui est déjà là.

IMPORTANT — alignement : la moyenne n'a de sens QUE si les 3 fichiers train/val/test.npy
sont dans le MÊME ORDRE de tuiles pour les 3 seeds (mêmes splits fixes, même pipeline
d'extraction). C'est vérifié explicitement (shapes + labels identiques) avant de moyenner ;
le script s'arrête si les labels divergent entre seeds.

Usage :
  python scripts/probe_ensemble_explora.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
MAX_ITER = 2000

BASE = "sota_screening/dinov3_vitb16_lvd_explora/embeddings/dinov3_vitb16_lvd_explora_frac100"
SEEDS = [0, 1, 2]


def _make_lr(c):
    try:
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                                  random_state=SEED, multi_class="multinomial")
    except (TypeError, ValueError):
        return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, random_state=SEED)


def load_seed(seed_i: int) -> dict:
    emb_dir = PROJ / f"{BASE}_seed{seed_i}"
    data = {}
    for s in ["train", "val", "test"]:
        e = np.load(str(emb_dir / f"{s}.npy")).astype(np.float32)
        l = np.load(str(emb_dir / f"{s}_labels.npy")).astype(np.int64).ravel()
        data[s] = (e, l)
    return data


def check_alignment(per_seed: list[dict]) -> None:
    """Les 3 seeds doivent porter EXACTEMENT les mêmes tuiles, dans le même ordre — sinon
    moyenner les embeddings tuile-à-tuile n'a aucun sens (on mélangerait des tuiles
    différentes). On vérifie via les labels (proxy simple : mêmes shapes + mêmes labels
    dans le même ordre pour les 3 splits)."""
    ref = per_seed[0]
    for s in ["train", "val", "test"]:
        ref_e, ref_l = ref[s]
        for i, d in enumerate(per_seed[1:], start=1):
            e, l = d[s]
            if e.shape != ref_e.shape:
                raise RuntimeError(
                    f"[{s}] seed0 shape={ref_e.shape} != seed{i} shape={e.shape} "
                    "— les splits ne sont pas alignés, impossible de moyenner.")
            if not np.array_equal(l, ref_l):
                raise RuntimeError(
                    f"[{s}] seed0 labels != seed{i} labels — ordre des tuiles différent "
                    "entre seeds, impossible de moyenner tuile-à-tuile.")
    print("[check] alignement OK : mêmes shapes + mêmes labels (même ordre) sur les 3 seeds.")


def linear_probe_f1(data: dict, n_classes: int):
    etr, ltr = data["train"]
    eva, lva = data["val"]
    ete, lte = data["test"]
    sc = StandardScaler()
    xtr = sc.fit_transform(etr)
    xva = sc.transform(eva)
    xte = sc.transform(ete)
    labels_range = list(range(n_classes))
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        clf = _make_lr(c)
        clf.fit(xtr, ltr)
        f1v = f1_score(lva, clf.predict(xva), average="macro", zero_division=0, labels=labels_range)
        if f1v > best_f1:
            best_f1, best_c = f1v, c
    clf = _make_lr(best_c)
    clf.fit(xtr, ltr)
    yp = clf.predict(xte)
    f1 = float(f1_score(lte, yp, average="macro", zero_division=0, labels=labels_range))
    return f1, best_c


def main():
    t0 = time.time()
    print("[load] chargement des 3 seeds...")
    per_seed = [load_seed(s) for s in SEEDS]
    check_alignment(per_seed)

    n_classes = len(np.unique(per_seed[0]["train"][1]))
    print(f"[info] n_classes={n_classes}")

    results = {}

    # --- (a) probe individuel par seed, POUR COMPARAISON directe avec la moyenne ---
    for i, seed_i in enumerate(SEEDS):
        f1, best_c = linear_probe_f1(per_seed[i], n_classes)
        results[f"seed{seed_i}"] = {"f1": f1, "best_C": best_c}
        print(f"[seed{seed_i}] F1={f1:.4f} best_C={best_c}")

    # --- (b) ensembling : moyenne des embeddings BRUTS (avant standardisation), tuile à tuile ---
    ensembled = {}
    for s in ["train", "val", "test"]:
        E_mean = np.mean([per_seed[i][s][0] for i in range(len(SEEDS))], axis=0)
        L = per_seed[0][s][1]  # identique sur les 3 (vérifié par check_alignment)
        ensembled[s] = (E_mean.astype(np.float32), L)

    f1_ens, best_c_ens = linear_probe_f1(ensembled, n_classes)
    results["ensemble_mean_3seeds"] = {"f1": f1_ens, "best_C": best_c_ens}
    print(f"[ensemble] F1={f1_ens:.4f} best_C={best_c_ens}")

    mean_individual = float(np.mean([results[f"seed{s}"]["f1"] for s in SEEDS]))
    std_individual = float(np.std([results[f"seed{s}"]["f1"] for s in SEEDS]))
    delta = f1_ens - mean_individual
    print(f"\n[résumé] moyenne des 3 F1 individuels = {mean_individual:.4f} ± {std_individual:.4f}")
    print(f"[résumé] F1 de l'embedding moyenné    = {f1_ens:.4f}  (Δ = {delta:+.4f} vs moyenne individuelle)")

    out = {
        "pipeline": "LR lbfgs, C_grid [1e-4..10], seed=42, max_iter=2000 — ensembling embeddings",
        "schema": f"{n_classes}cls",
        "individual_seeds": {f"seed{s}": results[f"seed{s}"] for s in SEEDS},
        "ensemble_mean_3seeds": results["ensemble_mean_3seeds"],
        "mean_individual_f1": mean_individual,
        "std_individual_f1": std_individual,
        "delta_ensemble_vs_mean_individual": delta,
    }
    out_path = PROJ / "results/probe_ensemble_explora_dinov3b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
