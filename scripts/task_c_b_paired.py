#!/usr/bin/env python3
"""TÂCHE C (b) — Tests appariés par tuile pour les 13 paires « palier A ».

Re-fit rapide du probe pour récupérer y_pred (17 598 tuiles), puis 3 tests
deepsig sur les scores de correction binaire (0/1) :
  - aso()  → eps_min
  - bootstrap_test()  → p-valeur appariée
  - permutation_test() → p-valeur de permutation

NB : pour les tests par tuile on utilise ``num_bootstrap_iterations=200``
au lieu des 1000 par défaut de deepsig — sinon chaque aso() prend >5 min
(200 × 200 = 40 000 quantiles par appel).  On perd un peu de précision sur
les p-valeurs mais le verdict (A>B ou non) reste stable.

Sorties : data/headline_pairs_paired_tests.json + data/headline_pairs_paired_tests.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

OUT = PROJ / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)

EMB = PROJ / "embeddings"
SEED = 42
N_BOOT_TILE = 200           # bootstrap interne de deepsig (par défaut : 1000 — trop lent)
N_PAIRED = 13               # nb de paires palier A (utilisé pour Bonferroni)

# best_C LU depuis la source canonique ``results/with_rhol/probe_knn_cgrid.json``
# (sélection par grille sur val, JAMAIS de C forcé en dur — réf. pass nocturne 1).
MODELS = [
    "resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
    "dinov3_vitl16_sat", "dinov3_vitl16_lvd", "simdinov2_vitb16",
    "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16",
    "resnet50_arctic", "vitb16_arctic", "vitb16_fulft_arctic",
]


def _load_best_canonical() -> dict[str, float]:
    """Best_C par modèle depuis la source canonique ``probe_knn_cgrid.json``.

    Ancienne version bugguée : ``BEST_C = 0.01`` en dur pour les 12 modèles.
    """
    pk = PROJ / "_anciennes_experiences" / "with_rhol" / "probe_knn_cgrid.json"
    with open(pk) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["best_C"]) for m, d in probe.items() if "best_C" in d}


BEST_C = _load_best_canonical()

HEADLINE_PAIRS = [
    ("dinov3_vitb16_lvd",      "simdinov2_vitb16"),
    ("dinov3_vitb16_lvd",      "vitb16_arctic"),
    ("dinov3_vitb16_lvd",      "simdinov2_vitl16"),
    ("dinov3_vitb16_lvd",      "vitb16_fulft_arctic"),
    ("dinov3_vitl16_lvd",      "simdinov2_vitb16"),
    ("dinov3_vitl16_lvd",      "vitb16_fulft_arctic"),     # ← HEADLINE
    ("dinov3_vitl16_lvd",      "simdinov2_vitl16"),
    ("dinov3_vitl16_lvd",      "vitb16_arctic"),
    ("dinov3_vitl16_sat",      "resnet50_arctic"),
    ("simdinov2_vitb16",       "vitb16_arctic"),
    ("simdinov2_vitl16",       "vitb16_arctic"),
    ("simdinov2_vitl16",       "vitb16_fulft_arctic"),
    ("vitb16_arctic",          "vitb16_fulft_arctic"),
]


def refit_predict(model_key: str):
    Etr = np.load(EMB / f"{model_key}_train.npy").astype(np.float32)
    Ltr = np.load(EMB / f"{model_key}_train_labels.npy").astype(np.int64)
    Ete = np.load(EMB / f"{model_key}_test.npy").astype(np.float32)
    Lte = np.load(EMB / f"{model_key}_test_labels.npy").astype(np.int64)
    sc = StandardScaler()
    xtr = sc.fit_transform(Etr); xte = sc.transform(Ete)
    clf = LogisticRegression(C=BEST_C[model_key], max_iter=2000, solver="lbfgs", random_state=SEED)
    clf.fit(xtr, Ltr)
    return Lte, clf.predict(xte)


def main() -> None:
    print("=" * 78)
    print("TÂCHE C(b) — tests appariés par tuile (paires palier A)")
    print(f"           num_bootstrap_iterations={N_BOOT_TILE} (réduit vs défaut 1000)")
    print("=" * 78)

    # Cache y_pred
    print("\nRe-fit des probes (best_C depuis probe_knn_cgrid.json, 12 modèles)…")
    y_pred_cache: dict[str, np.ndarray] = {}
    y_true_ref: np.ndarray | None = None
    for k in MODELS:
        t0 = time.time()
        y_true, y_pred = refit_predict(k)
        if y_true_ref is None:
            y_true_ref = y_true
        y_pred_cache[k] = y_pred
        print(f"  {k:24s}  refit={time.time()-t0:5.1f}s")

    # Scores de correction par tuile
    per_tile = {k: (y_pred_cache[k] == y_true_ref).astype(np.int64) for k in MODELS}
    acc = {k: float(per_tile[k].mean()) for k in MODELS}
    print(f"\nAccuracy par modèle (sur {len(y_true_ref)} tuiles) :")
    for k in MODELS:
        print(f"  {k:24s}  {acc[k]:.4f}")

    from deepsig import aso, bootstrap_test, permutation_test
    print(f"\n3 tests appariés (aso + bootstrap_test + permutation_test), "
          f"{len(HEADLINE_PAIRS)} paires :")
    rows = []
    for a, b in HEADLINE_PAIRS:
        sa, sb = per_tile[a], per_tile[b]
        t0 = time.time()
        e_aso = aso(sa, sb, seed=SEED, confidence_level=0.95,
                    num_bootstrap_iterations=N_BOOT_TILE, num_jobs=-1,
                    show_progress=False)
        e_aso_bonf = aso(sa, sb, seed=SEED, confidence_level=0.95,
                         num_comparisons=N_PAIRED,
                         num_bootstrap_iterations=N_BOOT_TILE, num_jobs=-1,
                         show_progress=False)
        p_boot = float(bootstrap_test(sa, sb, num_samples=N_BOOT_TILE, seed=SEED))
        p_perm = float(permutation_test(sa, sb, num_samples=N_BOOT_TILE, seed=SEED))
        dt = time.time() - t0
        rows.append({
            "model_a": a, "model_b": b,
            "acc_a": acc[a], "acc_b": acc[b],
            "delta_acc_a_minus_b": acc[a] - acc[b],
            "aso_eps_min": float(e_aso),
            "aso_eps_min_bonferroni": float(e_aso_bonf),
            "bootstrap_p_value": p_boot,
            "permutation_p_value": p_perm,
            "n_test_tiles": int(len(sa)),
            "aso_strict": "A>B (confiance haute)" if e_aso < 0.2 else (
                "A>B (modéré)" if e_aso < 0.5 else "indistinguables"),
            "asymp_significant_at_0.05": (p_boot < 0.05) or (p_perm < 0.05),
        })
        print(f"  {a:24s} vs {b:24s}  Δacc={acc[a]-acc[b]:+.4f}  "
              f"ASO ε={e_aso:.3f}  p_boot={p_boot:.3f}  p_perm={p_perm:.3f}  "
              f"({dt:.1f}s)")

    # CSV
    csv_path = OUT / "headline_pairs_paired_tests.csv"
    with open(csv_path, "w") as f:
        f.write("model_a,model_b,acc_a,acc_b,delta_acc,aso_eps_min,aso_eps_min_bonferroni,"
                "bootstrap_p_value,permutation_p_value,aso_strict\n")
        for r in rows:
            f.write(f"{r['model_a']},{r['model_b']},{r['acc_a']:.6f},{r['acc_b']:.6f},"
                    f"{r['delta_acc_a_minus_b']:+.6f},{r['aso_eps_min']:.4f},"
                    f"{r['aso_eps_min_bonferroni']:.4f},"
                    f"{r['bootstrap_p_value']:.4f},{r['permutation_p_value']:.4f},"
                    f"{r['aso_strict']}\n")
    print(f"\n[csv] -> {csv_path}")

    # JSON
    json_path = OUT / "headline_pairs_paired_tests.json"
    with open(json_path, "w") as f:
        json.dump({
            "n_test_tiles": int(len(y_true_ref)),
            "n_bootstrap_internal": N_BOOT_TILE,
            "seed": SEED,
            "score_type": "per-tile binary correctness (0/1) sur 17 598 tuiles test",
            "n_pairs": len(HEADLINE_PAIRS),
            "source_code": ("deepsig 1.2.5 — "
                            "from deepsig import aso, bootstrap_test, permutation_test"),
            "source_paper": "Dror et al. 2019 (ASO) ; Efron-Tibshirani 1994 (bootstrap) ; Noreen 1989 (permutation)",
            "pairs": rows,
            "interpretation": {
                "aso_eps_min_lt_0.2": "A > B avec confiance haute (Dror et al. 2019)",
                "aso_eps_min_lt_0.5": "A > B (modéré)",
                "aso_eps_min_ge_0.5": "indistinguables (non-rejet H0)",
                "bootstrap_p_value": "p-valeur single-tailed (H1: A > B) du paired bootstrap test",
                "permutation_p_value": "p-valeur single-tailed du permutation-randomization test",
                "n_bootstrap_internal_200_note": ("200 itérations au lieu des 1000 par défaut de "
                                                  "deepsig — compromis vitesse/precision pour ce "
                                                  "test à 17 598 observations par paire."),
            },
        }, f, indent=2)
    print(f"[json] -> {json_path}")

    # Petit récap
    print("\n" + "=" * 78)
    print("RÉCAP — Tests appariés par tuile (n=17 598)")
    print("=" * 78)
    n_strict = sum(1 for r in rows if r["aso_eps_min"] < 0.2)
    n_mod = sum(1 for r in rows if r["aso_eps_min"] < 0.5)
    n_boot_sig = sum(1 for r in rows if r["bootstrap_p_value"] < 0.05)
    n_perm_sig = sum(1 for r in rows if r["permutation_p_value"] < 0.05)
    print(f"  Paires  A>B (eps_min<0.5)  : {n_mod}/{len(rows)}")
    print(f"  Paires  A>B strict (eps<0.2): {n_strict}/{len(rows)}")
    print(f"  P-valeur bootstrap < 0.05  : {n_boot_sig}/{len(rows)}")
    print(f"  P-valeur permutation < 0.05 : {n_perm_sig}/{len(rows)}")
    print("=" * 78)
    print("TÂCHE C(b) terminée.")
    print("=" * 78)


if __name__ == "__main__":
    main()
