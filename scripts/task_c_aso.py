#!/usr/bin/env python3
"""TÂCHE C — deep-significance (ASO, bootstrap test, permutation test).

Stratégie
---------
Trois passes complémentaires, toutes sur le test set, déterministe (seed=42) :

(a) ``multi_aso`` sur les **distributions bootstrap f1_macro_pres** (n=1000) des
    12 modèles.  Renvoie une matrice 12×12 d'eps_min avec correction de
    Bonferroni sur les C(12,2) = 66 paires.  C'est le test "échantillon
    d'échantillon" (chaque modèle = 1000 scores f1).

(b) Pour les **paires palier A** (statistiquement indiscernables d'après la
    matrice bootstrap existante) on applique **3 tests appariés par tuile** sur
    les prédictions 0/1 (correct/incorrect) :
      - ASO ``aso(a, b)`` (Dror et al. 2019, deepsig)
      - paired bootstrap test ``bootstrap_test(a, b)`` (Efron-Tibshirani)
      - permutation-randomization test ``permutation_test(a, b)`` (Noreen)
    On rapporte ``eps_min`` (ASO) et la p-value (bootstrap, permutation).

(c) Pour le **mapping few-shot** (3 seeds par modèle) : on note qu'on n'a
    PAS de 3 seeds par modèle (le probe est déterministe avec lbfgs + seed=42).
    On le mentionne explicitement dans le log.

Toutes les distributions bootstrap f1 sont **recalculées** (re-fit LR + 1000
tirages) à partir des embeddings cachés, pas lues d'un fichier — pour donner
à deepsignificance les arrays de scores dont il a besoin.

Sorties
-------
  - data/bootstrap_distributions.json   (n=1000 f1_macro_pres par modèle)
  - data/aso_matrix_eps_min.csv         (matrice eps_min 12×12, Bonferroni)
  - data/aso_matrix_eps_min.json        (version structurée + p-value bootstrap)
  - data/headline_pairs_paired_tests.json  (résultats des 3 tests palier A)
  - figures/heatmap_eps_min.png|pdf

PIÈGE ÉVITÉ : on n'utilise pas le package PyPI ``logme`` (sans rapport) ; on
importe bien ``from deepsig import aso, multi_aso, bootstrap_test, permutation_test``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

OUT = PROJ / "results" / "transfer"
FIG = PROJ / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

EMB = PROJ / "embeddings"
N_BOOT = 1000
SEED = 42

# Modèles benchmarkés, dans l'ordre canonique des configs.
# best_C est LU depuis la source canonique ``results/with_rhol/probe_knn_cgrid.json``
# (sélection par grille sur val, JAMAIS de C forcé en dur — ancienne version bugguée
# avec C=0.01 pour les 12 modèles, réf. pass nocturne 1).
MODELS = [
    "resnet50_imagenet", "vitb16_imagenet", "dinov3_vitb16_lvd",
    "dinov3_vitl16_sat", "dinov3_vitl16_lvd", "simdinov2_vitb16",
    "simdinov2_vitl16", "satmae_vitl16", "scalemae_vitl16",
    "resnet50_arctic", "vitb16_arctic", "vitb16_fulft_arctic",
]


def _load_best_canonical() -> dict[str, float]:
    """Best_C par modèle depuis la source canonique ``probe_knn_cgrid.json``.

    La grille étendue ``C∈{1e-4..10}`` est sélectionnée sur val par f1_macro_all.
    Tout script qui force ``C=0.01`` est un BUG — voir :func:`_load_best_canonical`.
    """
    pk = PROJ / "results" / "with_rhol" / "probe_knn_cgrid.json"
    with open(pk) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["best_C"]) for m, d in probe.items() if "best_C" in d}


BEST_C = _load_best_canonical()

# Paires « palier A » à tester en mode apparié (cf. AGENT_LOG.md §4)
HEADLINE_PAIRS = [
    ("dinov3_vitb16_lvd",      "simdinov2_vitb16"),     # p_a_gt_b=0.458
    ("dinov3_vitb16_lvd",      "vitb16_arctic"),        # p_a_gt_b=0.056
    ("dinov3_vitb16_lvd",      "simdinov2_vitl16"),     # p_a_gt_b=0.045
    ("dinov3_vitb16_lvd",      "vitb16_fulft_arctic"),  # p_a_gt_b=0.003
    ("dinov3_vitl16_lvd",      "simdinov2_vitb16"),     # p_a_gt_b=0.993
    ("dinov3_vitl16_lvd",      "vitb16_fulft_arctic"),  # p_a_gt_b=0.483 (HEADLINE)
    ("dinov3_vitl16_lvd",      "simdinov2_vitl16"),     # p_a_gt_b=0.823
    ("dinov3_vitl16_lvd",      "vitb16_arctic"),        # p_a_gt_b=0.856
    ("dinov3_vitl16_sat",      "resnet50_arctic"),      # p_a_gt_b=0.480
    ("simdinov2_vitb16",       "vitb16_arctic"),        # p_a_gt_b=0.180
    ("simdinov2_vitl16",       "vitb16_arctic"),        # p_a_gt_b=0.567
    ("simdinov2_vitl16",       "vitb16_fulft_arctic"),  # p_a_gt_b=0.797
    ("vitb16_arctic",          "vitb16_fulft_arctic"),  # p_a_gt_b=0.863
]


def refit_predict(model_key: str):
    """Re-fit LogisticRegression avec best_C + StandardScaler, return (y_true, y_pred)."""
    Etr = np.load(EMB / f"{model_key}_train.npy").astype(np.float32)
    Ltr = np.load(EMB / f"{model_key}_train_labels.npy").astype(np.int64)
    Ete = np.load(EMB / f"{model_key}_test.npy").astype(np.float32)
    Lte = np.load(EMB / f"{model_key}_test_labels.npy").astype(np.int64)
    sc = StandardScaler()
    xtr = sc.fit_transform(Etr)
    xte = sc.transform(Ete)
    clf = LogisticRegression(C=BEST_C[model_key], max_iter=2000, solver="lbfgs",
                             random_state=SEED, n_jobs=-1)
    clf.fit(xtr, Ltr)
    return Lte, clf.predict(xte)


def bootstrap_f1_pres(y_true, y_pred, n_boot: int = N_BOOT, seed: int = SEED):
    """Renvoie (f1_pres_distribution[N], observed_f1_pres)."""
    rng = np.random.RandomState(seed)
    N = len(y_true)
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, N, N)
        yt, yp = y_true[idx], y_pred[idx]
        present = sorted({int(v) for v in yt})
        out[b] = f1_score(yt, yp, average="macro", zero_division=0, labels=present)
    obs = f1_score(y_true, y_pred, average="macro", zero_division=0,
                   labels=sorted({int(v) for v in y_true}))
    return out, float(obs)


def main() -> None:
    print("=" * 78)
    print("TÂCHE C — deep-significance : ASO (66 paires) + 3 tests appariés (palier A)")
    print("=" * 78)

    # 1. Re-fit + bootstrap pour chaque modèle
    boot_dist: dict[str, np.ndarray] = {}     # {model: f1_pres[N_BOOT]}
    y_pred_cache: dict[str, np.ndarray] = {}  # {model: y_pred[17598]} pour tests appariés
    y_true_ref = None
    obs_f1: dict[str, float] = {}

    for k in MODELS:
        t0 = time.time()
        y_true, y_pred = refit_predict(k)
        if y_true_ref is None:
            y_true_ref = y_true
        else:
            assert (y_true == y_true_ref).all(), "désalignement test labels"
        f1_dist, obs = bootstrap_f1_pres(y_true, y_pred, N_BOOT, SEED)
        boot_dist[k] = f1_dist
        y_pred_cache[k] = y_pred
        obs_f1[k] = obs
        print(f"[{k:24s}] refit+boot={time.time()-t0:5.1f}s  obs={obs:.4f}  "
              f"mean={f1_dist.mean():.4f}  std={f1_dist.std():.4f}")

    # Sauver les distributions bootstrap pour réutilisation
    boot_path = OUT / "bootstrap_distributions.json"
    with open(boot_path, "w") as f:
        json.dump({"n_bootstrap": N_BOOT, "seed": SEED,
                   "per_model": {k: v.tolist() for k, v in boot_dist.items()},
                   "observed_f1_macro_pres": obs_f1}, f, indent=2)
    print(f"\n[json] distributions bootstrap -> {boot_path}")

    # 2. multi_aso sur les 12×1000 f1 distributions
    from deepsig import aso, multi_aso
    print("\n" + "-" * 78)
    print("(a) multi_aso sur bootstrap f1 distributions (66 paires, Bonferroni)")
    print("-" * 78)
    scores = np.stack([boot_dist[k] for k in MODELS])  # (12, 1000)
    M = scores.shape[0]
    eps_mat = multi_aso(scores, confidence_level=0.95, seed=SEED, use_bonferroni=True,
                        num_jobs=-1, show_progress=False)
    eps_mat = np.asarray(eps_mat)
    assert eps_mat.shape == (M, M), f"forme inattendue : {eps_mat.shape}"

    # CSV matrice eps_min
    csv_path = OUT / "aso_matrix_eps_min.csv"
    with open(csv_path, "w") as f:
        f.write("," + ",".join(MODELS) + "\n")
        for i, m in enumerate(MODELS):
            f.write(m + "," + ",".join(f"{eps_mat[i, j]:.4f}" for j in range(M)) + "\n")
    print(f"[csv] -> {csv_path}")

    # Heatmap PNG/PDF
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(eps_mat, cmap="RdYlGn_r", vmin=0, vmax=0.5, aspect="auto")
    ax.set_xticks(range(M))
    ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M))
    ax.set_yticklabels(MODELS, fontsize=8)
    for i in range(M):
        for j in range(M):
            if i != j:
                ax.text(j, i, f"{eps_mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black")
    ax.set_title(r"ASO $\varepsilon_{min}$ matrix — 12 modèles, bootstrap f1 distributions"
                 "\n(plus bas = A plus stochastiquement dominant sur B ; "
                 r"Bonferroni sur 66 paires, $\alpha=0.05$)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\varepsilon_{min}$")
    fig.tight_layout()
    fig.savefig(FIG / "heatmap_eps_min.png", dpi=150)
    fig.savefig(FIG / "heatmap_eps_min.pdf")
    plt.close(fig)
    print(f"[fig] heatmap -> {FIG/'heatmap_eps_min.png'}")

    # JSON matrice eps_min
    mat_json = OUT / "aso_matrix_eps_min.json"
    with open(mat_json, "w") as f:
        json.dump({
            "models": MODELS,
            "n_bootstrap": N_BOOT, "seed": SEED,
            "bonferroni_corrected": True,
            "n_comparisons": M * (M - 1) // 2,
            "confidence_level": 0.95,
            "eps_min": eps_mat.tolist(),
            "interpretation": ("eps_min < 0.5 ⇒ A > B (Dror et al. 2019) ; "
                               "eps_min < 0.2 ⇒ confiance élevée ; "
                               "matrice antisymétrique (eps_min[i,j] = 1 - eps_min[j,i])")
        }, f, indent=2)
    print(f"[json] -> {mat_json}")

    # 3. Tests appariés par tuile pour les paires palier A
    print("\n" + "-" * 78)
    print("(b) Tests appariés par tuile — paires palier A (0/1 correctness)")
    print("-" * 78)
    from deepsig import bootstrap_test, permutation_test
    per_tile = {k: (y_pred_cache[k] == y_true_ref).astype(np.int64) for k in MODELS}

    headline_results = []
    for a, b in HEADLINE_PAIRS:
        sa, sb = per_tile[a], per_tile[b]
        # ASO
        e_aso = aso(sa, sb, seed=SEED, confidence_level=0.95, show_progress=False)
        # Paired bootstrap test (single-tailed : A > B)
        p_boot = float(bootstrap_test(sa, sb, num_samples=N_BOOT, seed=SEED))
        # Permutation test
        p_perm = float(permutation_test(sa, sb, num_samples=N_BOOT, seed=SEED))
        delta_acc = float(sa.mean() - sb.mean())
        headline_results.append({
            "model_a": a, "model_b": b,
            "delta_accuracy_a_minus_b": delta_acc,
            "acc_a": float(sa.mean()), "acc_b": float(sb.mean()),
            "aso_eps_min": float(e_aso),
            "aso_eps_min_bonferroni": float(aso(sa, sb, seed=SEED, confidence_level=0.95,
                                                num_comparisons=len(HEADLINE_PAIRS),
                                                show_progress=False)),
            "bootstrap_p_value": p_boot,
            "permutation_p_value": p_perm,
            "n_test_tiles": int(len(sa)),
            "verdict": "A > B" if e_aso < 0.5 else "B ≥ A (A non strictement supérieur)",
            "aso_strict": "A > B (confiance haute)" if e_aso < 0.2 else (
                "A > B (modéré)" if e_aso < 0.5 else "indistinguables"),
        })
        print(f"  {a:24s} vs {b:24s}  Δacc={delta_acc:+.4f}  "
              f"ASO ε={e_aso:.3f}  p_boot={p_boot:.3f}  p_perm={p_perm:.3f}  "
              f"verdict={headline_results[-1]['aso_strict']}")

    headline_path = OUT / "headline_pairs_paired_tests.json"
    with open(headline_path, "w") as f:
        json.dump({
            "n_bootstrap_per_tiles": N_BOOT, "seed": SEED,
            "score_type": "per-tile binary correctness (0/1) over 17 598 test tiles",
            "pairs": headline_results,
            "note_few_shot": ("Le mapping few-shot (3 seeds par modèle) n'est "
                              "PAS applicable ici : le probe est déterministe "
                              "(lbfgs + seed=42 fixé). ASO est donc appliqué sur "
                              "les distributions bootstrap f1 (1000 tirages) et "
                              "sur la correction par tuile 0/1, pas sur 3 seeds.")
        }, f, indent=2)
    print(f"\n[json] -> {headline_path}")

    # 4. Petit récap console
    print("\n" + "=" * 78)
    print("RÉCAP — ASO sur bootstrap distributions (n=1000, Bonferroni)")
    print("=" * 78)
    print(f"  {len(MODELS)} modèles × {N_BOOT} tirages bootstrap × C(12,2)={M*(M-1)//2} paires")
    n_dist = int(np.sum(eps_mat < 0.5)) // 2
    n_strict = int(np.sum(eps_mat < 0.2)) // 2
    print(f"  Paires distinguables (eps_min < 0.5, sur une moitié) : {n_dist}")
    print(f"  Paires très significatives (eps_min < 0.2)            : {n_strict}")
    print(f"  Heatmap : {FIG/'heatmap_eps_min.png'}")
    print("=" * 78)
    print("TÂCHE C terminée.")
    print("=" * 78)


if __name__ == "__main__":
    main()
