#!/usr/bin/env python3
"""Compare deux stratégies d'ensembling sur les 3 seeds ExPLoRA (DINOv3-B, run CORRIGÉ
20/07/2026, csv_dir fix — F1 individuels attendus proches de 0.48, PAS 0.52) :

  (a) Moyenne des EMBEDDINGS (features brutes moyennées tuile-à-tuile AVANT standardisation,
      un seul probe entraîné sur la moyenne) — un seul modèle final.
  (b) Moyenne des PROBABILITÉS (un probe complet, indépendant, par seed ; les 3 distributions
      de proba de test sont moyennées, puis argmax) — ensembling de décisions, 3 modèles.

(b) est généralement plus robuste en pratique : chaque probe capture indépendamment le bruit
d'init de son seed, la moyenne de décisions lisse ce bruit mieux qu'une moyenne de features
qui peut juste produire un point flou entre les 3 optima. (a) est moins cher (1 seul probe)
mais plus fragile si les 3 embeddings ne sont pas bien alignés géométriquement entre seeds.

Prérequis : les 3 seeds d'embeddings finaux ExPLoRA existent déjà sur le disque, run corrigé.
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


def linear_probe_f1(data: dict, n_classes: int, return_proba: bool = False):
    """Probe canonique standard. Si return_proba=True, retourne aussi les probabilités
    de test (n_test, n_classes) alignées sur `labels_range`, pour permettre l'ensembling
    par moyenne de probabilités en aval (voir ensemble_proba_f1)."""
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
    if return_proba:
        # clf.classes_ peut être un sous-ensemble/ordre différent de labels_range si une
        # classe est absente du train — on réaligne explicitement sur labels_range pour que
        # la moyenne inter-seeds soit colonne-à-colonne comparable (même ordre de classes).
        proba_raw = clf.predict_proba(xte)
        proba = np.zeros((xte.shape[0], n_classes), dtype=np.float64)
        for j, c in enumerate(clf.classes_):
            proba[:, int(c)] = proba_raw[:, j]
        return f1, best_c, proba
    return f1, best_c


def ensemble_proba_f1(per_seed: list[dict], n_classes: int):
    """Ensembling par MOYENNE DES PROBABILITÉS : un probe complet (train+selection C sur
    val) par seed, puis moyenne des proba de test sur les 3 seeds, puis argmax.
    Différent de l'ensembling par moyenne des embeddings (un seul probe sur la moyenne des
    features) : ici chaque seed garde son propre modèle, seule la décision finale est agrégée.
    """
    labels_range = list(range(n_classes))
    probas, f1s_individual, best_cs = [], [], []
    lte_ref = None
    for i, data in enumerate(per_seed):
        f1, best_c, proba = linear_probe_f1(data, n_classes, return_proba=True)
        probas.append(proba)
        f1s_individual.append(f1)
        best_cs.append(best_c)
        _, lte = data["test"]
        if lte_ref is None:
            lte_ref = lte
        elif not np.array_equal(lte, lte_ref):
            raise RuntimeError("labels test divergents entre seeds — alignement invalide.")

    proba_mean = np.mean(probas, axis=0)
    y_pred = proba_mean.argmax(axis=1)
    f1_ens = float(f1_score(lte_ref, y_pred, average="macro", zero_division=0, labels=labels_range))
    return f1_ens, f1s_individual, best_cs


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

    f1_ens_emb, best_c_ens = linear_probe_f1(ensembled, n_classes)
    results["ensemble_mean_embeddings"] = {"f1": f1_ens_emb, "best_C": best_c_ens}
    print(f"[ensemble embeddings] F1={f1_ens_emb:.4f} best_C={best_c_ens}")

    # --- (c) ensembling : moyenne des PROBABILITÉS (probe séparé par seed, proba moyennée) ---
    f1_ens_proba, f1s_check, best_cs_check = ensemble_proba_f1(per_seed, n_classes)
    results["ensemble_mean_proba"] = {"f1": f1_ens_proba, "best_C_per_seed": best_cs_check}
    print(f"[ensemble proba]      F1={f1_ens_proba:.4f}  (best_C par seed: {best_cs_check})")

    mean_individual = float(np.mean([results[f"seed{s}"]["f1"] for s in SEEDS]))
    std_individual = float(np.std([results[f"seed{s}"]["f1"] for s in SEEDS]))
    delta_emb = f1_ens_emb - mean_individual
    delta_proba = f1_ens_proba - mean_individual
    print(f"\n[résumé] moyenne des {len(SEEDS)} F1 individuels     = {mean_individual:.4f} ± {std_individual:.4f}")
    print(f"[résumé] F1 ensemble EMBEDDINGS (features moyennées) = {f1_ens_emb:.4f}  (Δ = {delta_emb:+.4f})")
    print(f"[résumé] F1 ensemble PROBA (décisions moyennées)     = {f1_ens_proba:.4f}  (Δ = {delta_proba:+.4f})")
    if f1_ens_proba > f1_ens_emb:
        print(f"[résumé] → PROBA gagne (+{f1_ens_proba - f1_ens_emb:.4f} vs embeddings)")
    elif f1_ens_emb > f1_ens_proba:
        print(f"[résumé] → EMBEDDINGS gagne (+{f1_ens_emb - f1_ens_proba:.4f} vs proba)")
    else:
        print("[résumé] → égalité exacte")

    out = {
        "pipeline": "LR lbfgs, C_grid [1e-4..10], seed=42, max_iter=2000 — comparaison ensembling",
        "schema": f"{n_classes}cls",
        "individual_seeds": {f"seed{s}": results[f"seed{s}"] for s in SEEDS},
        "ensemble_mean_embeddings": results["ensemble_mean_embeddings"],
        "ensemble_mean_proba": results["ensemble_mean_proba"],
        "mean_individual_f1": mean_individual,
        "std_individual_f1": std_individual,
        "delta_embeddings_vs_mean_individual": delta_emb,
        "delta_proba_vs_mean_individual": delta_proba,
    }
    out_path = PROJ / "results/probe_ensemble_explora_dinov3b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {out_path}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
