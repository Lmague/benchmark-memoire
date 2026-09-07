#!/usr/bin/env python3
"""Phase 0.b — Géométrie latente sur les embeddings TEST, à n fixe (17 598 tuiles).

Pourquoi : la géométrie de `results/datacurve_lora/geometry_all_metrics.csv` est
calculée sur les embeddings *train*, dont la taille varie de 247 à 49 281 lignes.
Or RankMe ≤ min(n, D) : la « montée de RankMe avec les données » y est partiellement
un artefact de taille d'échantillon. Ici tout est calculé sur le **même** jeu test
(n = 17 598 pour tous les points), ce qui rend les courbes comparables entre elles.

Trois conventions de rang effectif coexistent dans le dépôt et sont TOUTES calculées
ici, sous des noms distincts (elles diffèrent d'un facteur ~7) :

  rankme_sigma_raw       exp H(σ/Σσ)   sur embeddings bruts    → `geometry_extended_12models.json`
  rankme_sigma_centered  exp H(σ/Σσ)   sur embeddings centrés  → `results/datacurve_lora/`
  eff_rank_sigma2        exp H(σ²/Σσ²) sur embeddings centrés  → `geometry_extended.json`,
                                                                 `final_2026-07/geometry_11cls.json`
                                                                 (colonne « RankMe » des rapports)

Idem pour l'anisotropie (cosinus moyen vs λ₁·D/Σλ) et NC1 (tr Σ_W/tr Σ_B vs l'inverse).

Sorties :
  results/rapport_data/geometry_datacurve.csv  — régime × fraction × seed (runs locaux)
  results/rapport_data/geometry_models.csv     — 20 modèles à 100 % (3 seeds concaténés)
  results/rapport_data/geometry_meta.json      — protocole + contrôles de non-régression

    python3 scripts/rapport/geometry_test_fixed_n.py [--quick]
"""
from __future__ import annotations

import argparse
import csv
import glob
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

import numpy as np
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from registry import FROZEN_MODELS, OUT, RUN_FAMILIES, ensure_out, p, emb_prefix
# Fonctions canoniques du dépôt — ne pas réimplémenter (src/latent.py,
# scripts/geometry_extended.py, scripts/compute_all_clustering_logme.py)
from src.latent import anisotropy as aniso_cosine
from scripts.geometry_extended import (alpha_spectral, fisher_ratio,
                                       intrinsic_dim_twonn, knn_purity,
                                       participation_ratio, stable_rank)
from scripts.compute_all_clustering_logme import logme, nc1_metrics, nc2_dev_etf

SEED = 42
SUBSAMPLE = 20000


def _entropy_rank(w, eps=1e-12):
    w = np.asarray(w, dtype=np.float64)
    q = w / (w.sum() + eps)
    return float(np.exp(-(q * np.log(q + eps)).sum()))


def aniso_spectral(sig_centered, dim):
    """λ₁·D / Σλ — convention de `results/datacurve_lora/geometry_full.py` (1..D)."""
    eig = np.asarray(sig_centered, dtype=np.float64) ** 2
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return 0.0
    return float(eig.max() / eig.sum() * eig.size)


def compute(E: np.ndarray, L: np.ndarray, seed: int = SEED) -> dict:
    """Toutes les métriques, dans les deux/trois conventions."""
    rng = np.random.RandomState(seed)
    n, dim = E.shape
    if n > SUBSAMPLE:
        idx = rng.choice(n, SUBSAMPLE, replace=False)
        Es, Ls = E[idx], L[idx]
    else:
        Es, Ls = E, L
    Es64 = Es.astype(np.float64)

    sig_raw = np.linalg.svd(Es64, compute_uv=False)
    sig_cen = np.linalg.svd(Es64 - Es64.mean(0, keepdims=True), compute_uv=False)
    alpha = alpha_spectral(sig_cen)

    out = {
        "n": int(Es.shape[0]), "dim": int(dim),
        # --- rang effectif, 3 conventions
        "rankme_sigma_raw": _entropy_rank(sig_raw),
        "rankme_sigma_centered": _entropy_rank(sig_cen),
        "eff_rank_sigma2": _entropy_rank(sig_cen ** 2),
        "rankme_sigma_raw_norm": _entropy_rank(sig_raw) / dim,
        "eff_rank_sigma2_norm": _entropy_rank(sig_cen ** 2) / dim,
        # --- spectre
        "stable_rank": stable_rank(sig_cen),
        "participation_ratio": participation_ratio(sig_cen),
        "alpha_spectral": alpha["alpha_spectral"],
        "alpha_r2": alpha["alpha_r2"],
        # --- anisotropie, 2 conventions
        "aniso_cosine": aniso_cosine(Es, seed=seed),
        "aniso_spectral": aniso_spectral(sig_cen, dim),
        # --- séparabilité
        "fisher_ratio": fisher_ratio(Es64, Ls),
        "knn_purity": knn_purity(Es, Ls, seed=seed),
        "intrinsic_dim_twonn": intrinsic_dim_twonn(Es, seed=seed),
        # --- neural collapse, 2 conventions
        "nc1_canonical": nc1_metrics(Es64, Ls),
        # --- clustering (float64 : reproduit exactement compute_all_clustering_logme.py)
        "silhouette": float(silhouette_score(Es64, Ls, random_state=seed)),
        "dbi": float(davies_bouldin_score(Es64, Ls)),
        "chi": float(calinski_harabasz_score(Es64, Ls)),
    }
    out["nc1_inverse"] = 1.0 / (out["nc1_canonical"] + 1e-12)
    c_mean, c_std = nc2_dev_etf(Es64, Ls)
    out["nc2_mean_cos"], out["nc2_std_cos"] = c_mean, c_std
    return out


# ── Sources d'embeddings ─────────────────────────────────────────────────────
def datacurve_jobs() -> list[tuple]:
    """(model, fraction, seed, dir) pour tous les runs à embeddings test locaux."""
    jobs = []
    for key, fam in RUN_FAMILIES.items():
        emb_dir = fam[2]
        if not emb_dir:
            continue
        for d in sorted(glob.glob(p(emb_dir, f"{emb_prefix(key)}*"))):
            if not os.path.isfile(os.path.join(d, "test.npy")):
                continue
            name = os.path.basename(d)
            frac_tok = [t for t in name.split("_") if t.startswith("frac")]
            seed_tok = [t for t in name.split("_") if t.startswith("seed")]
            if not frac_tok or not seed_tok:
                continue
            fs = frac_tok[0][4:]
            frac = 0.005 if fs == "000" else int(fs) / 100.0
            jobs.append((key, frac, int(seed_tok[0][4:]), d))
    return sorted(jobs)


def model_jobs() -> list[tuple]:
    """(model, kind, paires_test, paires_train) — une paire par seed.

    Les paires train ne servent qu'à LogME : c'est le seul score de sélection sans
    étiquettes qui se calcule, par définition, sur les données d'entraînement de la
    tâche cible (You et al., ICML 2021). Le reste de la géométrie reste sur le test.
    """
    jobs = []
    for key in FROZEN_MODELS:
        jobs.append((key, "frozen",
                     [(p("embeddings", f"{key}_test.npy"),
                       p("embeddings", f"{key}_test_labels.npy"))],
                     [(p("embeddings", f"{key}_train.npy"),
                       p("embeddings", f"{key}_train_labels.npy"))]))
    # FT : dossiers de run (test.npy / test_labels.npy), 3 seeds
    ft_bases = {
        "dinov3_vitb16_lvd_full": "ft_ssl_results/dinov3_vitb16_lvd_full_embeddings/dinov3_vitb16_lvd_full_frac100",
        "dinov3_vitb16_lvd_mhsa": "ft_ssl_results/dinov3_vitb16_lvd_mhsa_embeddings/dinov3_vitb16_lvd_mhsa_frac100",
        "simdinov2_vitb16_full": "ft_ssl_results/simdinov2_vitb16_full_embeddings/simdinov2_vitb16_full_frac100",
        "simdinov2_vitb16_mhsa": "ft_ssl_results/simdinov2_vitb16_mhsa_embeddings/simdinov2_vitb16_mhsa_frac100",
        # NB nommage trompeur, mapping fixé par scripts/compute_all_clustering_logme.py :
        # ce dossier `..._explora_frac100` contient bien LoRA r=8.
        "dinov3_vitb16_lvd_lora_r8": "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100",
        "resnet50_fulft_sota": "embeddings/vitb16_full_frac100",     # 2048-dim = ResNet-50
        "vitb16_full_old": "sota_screening/full/embeddings/vitb16_full_frac100",
        "vitb16_mhsa_old": "sota_screening/mhsa/embeddings/vitb16_mhsa_frac100",
        "vitb16_scratch_old": "sota_screening/scratch/embeddings/vitb16_scratch_frac100",
        # ── ajout 2026-07-28 : LoRA r=8 DINOv3-L, SimDINOv2-B, SimDINOv2-L ──
        "dinov3_vitl16_lvd_lora": "ViTB_L_LoRA/lora_3models/embeddings/dinov3_vitl16_lvd_lora_frac100",
        "simdinov2_vitl16_lora": "ViTB_L_LoRA/lora_3models/embeddings/simdinov2_vitl16_lora_frac100",
        "simdinov2_vitb16_lora": "ViTB_L_LoRA/lora_3models/simdinov2_vitb16/embeddings/frac100",
    }
    for key, base in ft_bases.items():
        pairs = [(p(f"{base}_seed{s}", "test.npy"), p(f"{base}_seed{s}", "test_labels.npy"))
                 for s in (0, 1, 2)]
        tr = [(p(f"{base}_seed{s}", "train.npy"), p(f"{base}_seed{s}", "train_labels.npy"))
              for s in (0, 1, 2)]
        keep = [i for i, (a, _) in enumerate(pairs) if os.path.exists(a)]
        if keep:
            jobs.append((key, "ft", [pairs[i] for i in keep],
                         [tr[i] for i in keep if os.path.exists(tr[i][0])]))
        else:
            print(f"  [SKIP] {key}: embeddings absents ({base})")
    # ── ajout 2026-08-31 : self-distillation contexte→tuile (results/context_distill) ──
    # Embeddings sig_embeddings/<tag> déjà en schéma 11 classes (pas de remap).
    for ckey, cbase in {
        "ctxdistill_dB_tL":   "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100",
        "ctxdistill_dA_tL":   "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100",
        "ctxdistill_dA_tEMA": "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dA_tEMA_ctx1024_r2a4_frac100",
    }.items():
        pairs = [(p(f"{cbase}_seed{s}", "test.npy"), p(f"{cbase}_seed{s}", "test_labels.npy"))
                 for s in (0, 1, 2)]
        tr = [(p(f"{cbase}_seed{s}", "train.npy"), p(f"{cbase}_seed{s}", "train_labels.npy"))
              for s in (0, 1, 2)]
        keep = [i for i, (a, _) in enumerate(pairs) if os.path.exists(a)]
        if keep:
            jobs.append((ckey, "ft", [pairs[i] for i in keep],
                         [tr[i] for i in keep if os.path.exists(tr[i][0])]))
        else:
            print(f"  [SKIP] {ckey}: embeddings absents ({cbase})")
    # ── DINOv3 ViT-S/16 LoRA r=8 (runs_vits16, 3 seeds, déjà en schéma 11 classes) ──
    cbase = "runs_vits16/embeddings/seed"
    pairs = [(p(f"{cbase}{s}", "test.npy"), p(f"{cbase}{s}", "test_labels.npy"))
             for s in (0, 1, 2)]
    tr = [(p(f"{cbase}{s}", "train.npy"), p(f"{cbase}{s}", "train_labels.npy"))
          for s in (0, 1, 2)]
    if all(os.path.exists(a) for a, _ in pairs):
        jobs.append(("dinov3_vits16_lvd_lora_r8", "ft", pairs, tr))
    else:
        print(f"  [SKIP] dinov3_vits16_lvd_lora_r8: embeddings absents ({cbase})")
    # ── ajout 2026-09 : ablation LoRA/PEFT SimDINOv2-B, Stage A (13 bras) ───
    # Embeddings results/lora_simb_ablation[_qkv]/embeddings/<tag>_seed{s}/ déjà
    # en schéma 11 classes. Les 2 runs SimDINOv2-B entraînés Design B (fusion 512)
    # n'ont PAS d'embeddings rapatriés (checkpoints sur $SCRATCH uniquement) :
    # entrées aspirantes → SKIP attendu, levé quand les sig_embeddings arrivent.
    for ckey, cbase in {
        "simdinov2_vitb16_lora_r8_b611":   "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b67891011_frac100",
        "simdinov2_vitb16_lora_r8_b05":    "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b012345_frac100",
        "simdinov2_vitb16_lora_r8_b911":   "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b91011_frac100",
        "simdinov2_vitb16_lora_r2":        "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r2a2_frac100",
        "simdinov2_vitb16_lora_r4":        "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r4a4_frac100",
        "simdinov2_vitb16_lora_r16":       "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a16_frac100",
        "simdinov2_vitb16_lora_r32":       "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r32a32_frac100",
        "simdinov2_vitb16_lora_r8_s2":     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a16_frac100",
        "simdinov2_vitb16_lora_r16_s2":    "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a32_frac100",
        "simdinov2_vitb16_lora_r8_rslora": "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a22_frac100",
        "simdinov2_vitb16_lora_r16_rslora":"results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a64_frac100",
        "simdinov2_vitb16_lora_r8_qkv":    "results/lora_simb_ablation_qkv/embeddings/simdinov2_vitb16_lora_r8a8_frac100",
        "simdinov2_vitb16_norm_tuning":    "results/lora_simb_ablation/embeddings/simdinov2_vitb16_norm_tuning_frac100",
        "ctxdistill_dB_tSL_r2a4":  "results/context_distill/sig_embeddings/simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r2a4_frac100",
        "ctxdistill_dB_tSL_r8a16": "results/context_distill/sig_embeddings/simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_r8a16_frac100",
    }.items():
        pairs = [(p(f"{cbase}_seed{s}", "test.npy"), p(f"{cbase}_seed{s}", "test_labels.npy"))
                 for s in (0, 1, 2)]
        tr = [(p(f"{cbase}_seed{s}", "train.npy"), p(f"{cbase}_seed{s}", "train_labels.npy"))
              for s in (0, 1, 2)]
        keep = [i for i, (a, _) in enumerate(pairs) if os.path.exists(a)]
        if keep:
            jobs.append((ckey, "ft", [pairs[i] for i in keep],
                         [tr[i] for i in keep if os.path.exists(tr[i][0])]))
        else:
            print(f"  [SKIP] {ckey}: embeddings absents ({cbase})")
    return jobs


def logme_of(pairs, seed=SEED):
    """LogME moyen sur les paires fournies (une par seed), sous-échantillon 20 000.

    Réutilise `logme()` de scripts/compute_all_clustering_logme.py — même variante
    (cible scalaire centrée) que le chiffre canonique de results/logme_train_all.json,
    sinon les valeurs ne seraient pas comparables à l'existant.
    """
    if not pairs:
        return None
    vals = []
    for ep, lp in pairs:
        if not (os.path.exists(ep) and os.path.exists(lp)):
            continue
        E = np.load(ep).astype(np.float64)
        L = np.load(lp).astype(np.int64)
        rng = np.random.RandomState(seed)
        if len(E) > SUBSAMPLE:
            idx = rng.choice(len(E), SUBSAMPLE, replace=False)
            E, L = E[idx], L[idx]
        vals.append(logme(E, L))
    if not vals:
        return None
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def run_datacurve(quick=False):
    jobs = datacurve_jobs()
    if quick:
        jobs = jobs[:4]
    print(f"== Géométrie datacurve : {len(jobs)} runs ==")
    rows, t0 = [], time.time()
    for i, (model, frac, seed, d) in enumerate(jobs, 1):
        E = np.load(os.path.join(d, "test.npy")).astype(np.float32)
        L = np.load(os.path.join(d, "test_labels.npy")).astype(np.int64)
        m = compute(E, L)
        m.update(model=model, fraction=frac, seed=seed,
                 emb_dir=os.path.relpath(d, p()))
        rows.append(m)
        print(f"  [{i:3d}/{len(jobs)}] {model:26s} f={frac:<5.3f} s{seed}  "
              f"rk(σ)={m['rankme_sigma_raw']:6.1f} eff(σ²)={m['eff_rank_sigma2']:6.1f} "
              f"sil={m['silhouette']:+.4f} kNN={m['knn_purity']:.3f} "
              f"({time.time() - t0:.0f}s)")
    cols = ["model", "fraction", "seed", "n", "dim", "rankme_sigma_raw",
            "rankme_sigma_centered", "eff_rank_sigma2", "rankme_sigma_raw_norm",
            "eff_rank_sigma2_norm", "stable_rank", "participation_ratio",
            "alpha_spectral", "alpha_r2", "aniso_cosine", "aniso_spectral",
            "fisher_ratio", "knn_purity", "intrinsic_dim_twonn", "nc1_canonical",
            "nc1_inverse", "silhouette", "dbi", "chi", "nc2_mean_cos", "nc2_std_cos",
            "emb_dir"]
    with open(os.path.join(OUT, "geometry_datacurve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] geometry_datacurve.csv ({len(rows)} runs)")
    return rows


def run_models(quick=False):
    """Protocole canonique : métriques calculées PAR SEED puis moyennées.

    Ne pas concaténer les seeds avant le calcul : l'union de trois modèles affinés
    différents occupe plus de directions qu'un seul, ce qui gonfle mécaniquement le
    rang effectif (LoRA r=8 : 43,7 par-seed → 71,6 en concaténé). La moyenne
    par-seed reproduit exactement les valeurs de `geometry_extended.json`.
    """
    jobs = model_jobs()
    if quick:
        jobs = jobs[:3]
    print(f"\n== Géométrie modèles (100 %, 3 seeds concaténés) : {len(jobs)} modèles ==")
    rows, t0 = [], time.time()
    metric_keys = None
    for i, (key, kind, pairs, train_pairs) in enumerate(jobs, 1):
        per_seed = []
        for ep, lp in pairs:
            per_seed.append(compute(np.load(ep).astype(np.float32),
                                    np.load(lp).astype(np.int64)))
        if metric_keys is None:
            metric_keys = [k for k, v in per_seed[0].items()
                           if isinstance(v, (int, float)) and v is not None]
        m = {"model": key, "kind": kind, "n_seeds": len(per_seed)}
        for k in metric_keys:
            v = [s[k] for s in per_seed if s.get(k) is not None]
            if not v:
                continue
            m[k] = float(np.mean(v))
            if k not in ("n", "dim"):
                m[k + "_std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        m["n"] = int(per_seed[0]["n"])
        m["dim"] = int(per_seed[0]["dim"])
        for tag, pr in (("logme_train", train_pairs), ("logme_test", pairs)):
            r = logme_of(pr)
            if r is not None:
                m[tag], m[tag + "_std"] = r
        rows.append(m)
        print(f"  [{i:2d}/{len(jobs)}] {key:28s} rk(σ)={m['rankme_sigma_raw']:6.1f} "
              f"eff(σ²)={m['eff_rank_sigma2']:6.1f} sil={m['silhouette']:+.5f} "
              f"aniso={m['aniso_cosine']:.3f} ({time.time() - t0:.0f}s)")
    base = ["rankme_sigma_raw", "rankme_sigma_centered", "eff_rank_sigma2",
            "rankme_sigma_raw_norm", "eff_rank_sigma2_norm", "stable_rank",
            "participation_ratio", "alpha_spectral", "alpha_r2", "aniso_cosine",
            "aniso_spectral", "fisher_ratio", "knn_purity", "intrinsic_dim_twonn",
            "nc1_canonical", "nc1_inverse", "silhouette", "dbi", "chi",
            "nc2_mean_cos", "nc2_std_cos", "logme_train", "logme_test"]
    cols = ["model", "kind", "n_seeds", "n", "dim"]
    for b in base:
        cols += [b, b + "_std"]
    with open(os.path.join(OUT, "geometry_models.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] geometry_models.csv ({len(rows)} modèles)")
    return rows


def regression_checks(model_rows):
    """Valeurs canoniques attendues (tolérance : sous-échantillonnage identique)."""
    # Références : geometry_extended_12models.json (rankme σ bruts),
    # geometry_extended.json (eff. rank σ², stable rank),
    # all_models_full_table.json (silhouette). Attention : la colonne silhouette de
    # `results/literature_experiments.csv` est ANTÉRIEURE et diverge (protocole
    # différent) — ne pas l'utiliser comme référence.
    ref = {
        # modèle : (rankme_sigma_raw, eff_rank_sigma2, stable_rank, silhouette)
        "dinov3_vitb16_lvd": (357.57, 61.39, 5.848, 0.006557),
        "dinov3_vitl16_lvd": (449.21, 58.50, 5.612, 0.011071),
        "resnet50_imagenet": (420.35, 21.90, 2.746, -0.026281),
        # Modèles affinés : silhouette volontairement non contrôlée — voir la note
        # ci-dessous, le protocole publié n'est pas le même.
        "dinov3_vitb16_lvd_lora_r8": (None, 43.69, 4.898, None),
        "dinov3_vitb16_lvd_full": (None, 48.60, 5.830, None),
        "scalemae_vitl16": (None, 10.90, 2.540, -0.030820),
    }
    # NOTE (divergence de protocole documentée) : pour les modèles affinés,
    # `scripts/compute_all_clustering_logme.py` calcule silhouette, NC1, DBI, CHI et
    # LogME sur les 3 seeds CONCATÉNÉS (52 794 lignes) puis sous-échantillonne à
    # 20 000, alors que les colonnes spectrales du même tableau proviennent de
    # `geometry_extended.json`, calculé PAR SEED puis moyenné. Le tableau maître
    # publié mélange donc deux protocoles selon la colonne. De plus le tirage
    # concaténé dépend de la position du modèle dans la boucle (un unique
    # RandomState est consommé pour tous). Ici tout est par-seed puis moyenné :
    # déterministe et homogène, mais les silhouettes des modèles affinés diffèrent
    # des valeurs publiées (LoRA r=8 : 0,0444 ici vs 0,0257 publié).
    res = []
    idx = {r["model"]: r for r in model_rows}
    for k, (rk, ef, sr, sil) in ref.items():
        if k not in idx:
            continue
        r = idx[k]
        for name, expected, got in (("rankme_sigma_raw", rk, r["rankme_sigma_raw"]),
                                    ("eff_rank_sigma2", ef, r["eff_rank_sigma2"]),
                                    ("stable_rank", sr, r["stable_rank"]),
                                    ("silhouette", sil, r["silhouette"])):
            if expected is None:
                continue
            rel = abs(got - expected) / (abs(expected) + 1e-12)
            res.append({"model": k, "metric": name, "expected": expected,
                        "got": got, "rel_err": rel, "ok": bool(rel < 0.02)})
    # LogME train : doit retomber sur results/logme_train_all.json
    fp = p("results", "logme_train_all.json")
    if os.path.exists(fp):
        canon = json.load(open(fp))["logme_train"]
        for r in model_rows:
            if r["model"] in canon and r.get("logme_train") is not None:
                exp, got = canon[r["model"]], r["logme_train"]
                res.append({"model": r["model"], "metric": "logme_train",
                            "expected": exp, "got": got,
                            "rel_err": abs(got - exp) / abs(exp),
                            # tolérance absolue : LogME vaut ~5e4, le tirage du sous-échantillon
                            # de 20 000 suffit à déplacer la valeur de quelques unités
                            "ok": bool(abs(got - exp) < 10.0)})
    print("\n== Contrôles de non-régression (vs geometry_extended*.json) ==")
    for r in res:
        print(f"  [{'OK  ' if r['ok'] else 'FAIL'}] {r['model']:20s} {r['metric']:20s} "
              f"attendu {r['expected']:>10.5f}  obtenu {r['got']:>10.5f}  "
              f"({100 * r['rel_err']:.2f} %)")
    return res


def identity_check(model_rows):
    """Le dossier d'embeddings de LoRA r=8 s'appelle `..._explora_frac100` : on vérifie
    par le rang effectif qu'on a bien chargé LoRA et pas un autre modèle."""
    idx = {r["model"]: r for r in model_rows}
    out = {}
    for k, expected in (("dinov3_vitb16_lvd_lora_r8", 43.7),):
        if k in idx:
            got = idx[k]["eff_rank_sigma2"]
            out[k] = {"eff_rank_sigma2_attendu": expected, "obtenu": got,
                      "coherent": bool(abs(got - expected) / expected < 0.10)}
    print("\n== Contrôle d'identité LoRA r=8 (eff_rank σ²) ==")
    for k, v in out.items():
        print(f"  [{'OK  ' if v['coherent'] else 'FAIL'}] {k:28s} "
              f"attendu {v['eff_rank_sigma2_attendu']:.1f}  obtenu {v['obtenu']:.1f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="échantillon réduit (test)")
    ap.add_argument("--only", choices=["datacurve", "models"], default=None)
    a = ap.parse_args()
    ensure_out()
    t0 = time.time()
    dc = run_datacurve(a.quick) if a.only != "models" else []
    mo = run_models(a.quick) if a.only != "datacurve" else []
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "protocole": ("embeddings TEST, sous-échantillon 20 000 (seed 42) ; "
                      "n = 17 598 pour un run seul, 52 794→20 000 pour 3 seeds concaténés"),
        "conventions": {
            "rankme_sigma_raw": "exp H(σ/Σσ), embeddings bruts — geometry_extended_12models.json",
            "rankme_sigma_centered": "exp H(σ/Σσ), embeddings centrés — results/datacurve_lora/",
            "eff_rank_sigma2": "exp H(σ²/Σσ²), embeddings centrés — colonne « RankMe » des rapports",
            "aniso_cosine": "cosinus moyen sur 10 000 paires (Ethayarajh 2019)",
            "aniso_spectral": "λ₁·D/Σλ — convention geometry_full.py (1..D)",
            "nc1_canonical": "tr(Σ_W)/tr(Σ_B) — bas = bon (convention des rapports)",
            "nc1_inverse": "tr(Σ_B)/tr(Σ_W) — haut = bon (convention geometry_full.py)",
        },
        "n_runs_datacurve": len(dc), "n_modeles": len(mo),
        "duree_s": round(time.time() - t0, 1),
    }
    if mo:
        meta["controles"] = regression_checks(mo)
        meta["identite_lora"] = identity_check(mo)
    json.dump(meta, open(os.path.join(OUT, "geometry_meta.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n[FIN] {time.time() - t0:.0f}s — {OUT}")


if __name__ == "__main__":
    main()
