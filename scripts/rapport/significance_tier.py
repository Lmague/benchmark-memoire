#!/usr/bin/env python3
"""Bootstrap hiérarchique apparié sur TOUT le palier compétitif (registry.TIER_K).

Pourquoi ce script existe
-------------------------
`results/significance_matrix_8group_fresh.json` ne couvre que 8 modèles. Or trois
membres du palier compétitif en sont absents — dont **le modèle n°1 du benchmark**
(DINOv3-B LoRA r=8), plus ViT-B/16 IN-Full et IN-MHSA. Le tableau maître le signalait
d'ailleurs : « Non testé formellement ». La phrase centrale de `performances.pdf`
(« tie statistique … les IC95 se recouvrent ») citait donc une table qui ne contenait
pas le modèle dont elle parlait, et son ±0,0011 était un écart-type inter-seed — pas
un IC95 bootstrap, une quantité ~4 fois plus étroite.

Ce script refait le test sur tous les modèles du palier (registry.TIER_K — 11 au
lancement initial, 14 depuis l'ajout 2026-07-28 des 3 régimes LoRA r=8 DINOv3-L/
SimDINOv2-B/SimDINOv2-L, qui entrent dans le palier sans en déplacer aucun membre),
avec le protocole canonique et les MÊMES indices de rééchantillonnage que la
campagne à 8 groupes (seed 42, 10 000 tirages) : les CI des 8 groupes déjà publiés
doivent se reproduire à l'identique, ce qui sert de contrôle de non-régression.

Protocole (identique à `probe.py` / `src.probe.linear_probe`)
-------------------------------------------------------------
  - StandardScaler sur le train, appliqué à val et test ;
  - LogisticRegression lbfgs multinomial, max_iter=2000, seed 42 ;
  - best_C choisi sur la VALIDATION (grille 1e-4 … 10), jamais sur le test ;
  - métrique : F1-macro sur les classes présentes dans le jeu test (11 classes).

Le bootstrap est apparié : les mêmes indices de tuiles ET de seeds sont partagés par
tous les groupes, donc les Δ entre modèles sont comparés sur les mêmes rééchantillons.

Sortie : results/significance_matrix_tier.json
Cache  : results/rapport_data/tier_preds_cache.npz (prédictions par seed — un
         redémarrage ne refait pas les probes déjà calculées).

    python3 scripts/rapport/significance_tier.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# MONO-THREAD OBLIGATOIRE — ce n'est pas un réglage de performance.
# Le nombre de threads BLAS change l'ordre de réduction, donc la solution vers
# laquelle lbfgs converge : DINOv3-B MHSA seed0 vaut 0,4821 à 1 thread (la valeur
# publiée dans significance_matrix_8group_fresh.json) et 0,4836 à 3 threads, alors
# que le solveur converge dans les deux cas (260 itérations sur 2000 autorisées).
# L'écart, 0,0015, est du même ordre que les effets discutés dans les rapports.
# Le parallélisme vient de joblib sur la grille C (processus séparés), pas de BLAS.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from registry import CANONICAL_F1, OUT, TIER_K, ensure_out, p

SEED, N_BOOT, MAX_ITER = 42, 10_000, 2000
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]   # configs/frozen_eval.yaml
N_CLASSES = 11
CACHE = os.path.join(OUT, "tier_preds_cache.npz")

# ── Les 20 groupes du palier (TIER_K=20 : 0,5080 → 0,4689, décrochage +0,0069) ────
# kind: "ft" = dossier de run (train/val/test.npy, déjà en schéma 11 classes)
#       "frozen" = embeddings/{key}_{split}.npy, schéma 12 classes → RHOL (idx 7) retiré
# best_C connu = celui de la campagne canonique ; None = à sélectionner sur val ici.
GROUPS = [
    ("DINOv3-B LoRA r=8", "ft", "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed{s}", [0, 1, 2], None),
    ("DINOv3 ViT-H+16", "frozen", "dinov3_vith16plus_lvd", [None], None),
    ("DINOv3-B MHSA", "ft", "ft_ssl_results/dinov3_vitb16_lvd_mhsa_embeddings/dinov3_vitb16_lvd_mhsa_frac100_seed{s}", [0, 1, 2], 0.001),
    ("DINOv3 ViT-L16", "frozen", "dinov3_vitl16_lvd", [None], None),
    ("DINOv3-B Full", "ft", "ft_ssl_results/dinov3_vitb16_lvd_full_embeddings/dinov3_vitb16_lvd_full_frac100_seed{s}", [0, 1, 2], 0.001),
    ("SimDINOv2 ViT-L16", "frozen", "simdinov2_vitl16", [None], None),
    ("SimDINOv2 ViT-B16", "frozen", "simdinov2_vitb16", [None], None),
    ("DINOv3 ViT-B16", "frozen", "dinov3_vitb16_lvd", [None], None),
    ("ViT-B/16 IN-Full", "ft", "sota_screening/full/embeddings/vitb16_full_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B MHSA", "ft", "ft_ssl_results/simdinov2_vitb16_mhsa_embeddings/simdinov2_vitb16_mhsa_frac100_seed{s}", [0, 1, 2], 0.001),
    ("ViT-B/16 IN-MHSA", "ft", "sota_screening/mhsa/embeddings/vitb16_mhsa_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B Full", "ft", "ft_ssl_results/simdinov2_vitb16_full_embeddings/simdinov2_vitb16_full_frac100_seed{s}", [0, 1, 2], None),
    # ── ajout 2026-07-28 : TIER_K passe de 11 à 14, ces 3 modèles entrent dans le
    # palier sans en déplacer aucun (registry.tier_break() recalculé) ──
    ("DINOv3-L LoRA r=8", "ft", "ViTB_L_LoRA/lora_3models/embeddings/dinov3_vitl16_lvd_lora_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-L LoRA r=8", "ft", "ViTB_L_LoRA/lora_3models/embeddings/simdinov2_vitl16_lora_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r=8", "ft", "ViTB_L_LoRA/lora_3models/simdinov2_vitb16/embeddings/frac100_seed{s}", [0, 1, 2], None),
    # ── ajout 2026-07-30 : TIER_K passe de 14 à 15 (registry.tier_break() recalculé) ;
    # DINOv3 ViT-H+/16 s'insère au rang 3 sans déplacer aucun membre existant, mais
    # repousse vitb16_mhsa_old du rang 14 au rang 15 — reste dans le palier grâce au
    # décalage de TIER_K, cf. registry.py ──
    # ── ajout 2026-08-31 : self-distillation contexte→tuile (kind="ft", déjà 11cls) ──
    ("Contexte R2 (B, fusion)", "ft",
     "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed{s}", [0, 1, 2], None),
    ("Contexte R1 (A, tuile)", "ft",
     "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dA_tL_ctx1024_r2a4_frac100_seed{s}", [0, 1, 2], None),
    ("Contexte R3 (A, EMA)", "ft",
     "results/context_distill/sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dA_tEMA_ctx1024_r2a4_frac100_seed{s}", [0, 1, 2], None),
    # ── ajout 2026-08-31 : TIER_K=20 — DINOv3 ViT-S/16 entre au rang 13 (LoRA) / 19 (gelé) ──
    ("DINOv3 ViT-S/16 LoRA r=8", "ft", "runs_vits16/embeddings/seed{s}", [0, 1, 2], None),
    ("DINOv3 ViT-S/16", "frozen", "dinov3_vits16_lvd", [None], None),
    # ── ajout 2026-09 : ablation LoRA/PEFT SimDINOv2-B, Stage A (13 bras) ────
    # Même test (17 598 tuiles, ordre vérifié identique au screening), schéma 11
    # classes déjà appliqué, best_C=None → sélection sur validation ici.
    ("SimDINOv2-B LoRA r8 (blocs 6-11)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b67891011_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r8 (blocs 0-5)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b012345_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r8 (blocs 9-11)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a8_b91011_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r2", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r2a2_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r4", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r4a4_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r16", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a16_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r32", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r32a32_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r8 (scaling 2)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a16_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r16 (scaling 2)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a32_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r8 (rsLoRA)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r8a22_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r16 (rsLoRA)", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_lora_r16a64_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B LoRA r8 (Q+K+V)", "ft",
     "results/lora_simb_ablation_qkv/embeddings/simdinov2_vitb16_lora_r8a8_frac100_seed{s}", [0, 1, 2], None),
    ("SimDINOv2-B NormTuning", "ft",
     "results/lora_simb_ablation/embeddings/simdinov2_vitb16_norm_tuning_frac100_seed{s}", [0, 1, 2], None),
]
# ⚠ `DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed*` EST LoRA r=8
#   malgré son nom de dossier (AGENT_MEMORY.md, 2026-07-22).

# Plus de raccourci "best_C canonique déjà publié" ici (retiré 2026-07-29) : ce
# raccourci hardcodait pour SimDINOv2-B Full un best_C (0,001) qui n'était PAS
# celui sélectionné par une grille sur validation dans ce script — il reproduisait
# une erreur de results/ft_ssl_probe_CANONICAL.json (sous-échantillon de grille C,
# corrigé depuis dans src/probe.py). Tous les modèles à best_c=None passent
# maintenant par la même sélection sur validation, sans exception.


def f1_pres(y_true, y_pred, k=N_CLASSES):
    """F1-macro sur les classes présentes dans y_true — équivalent exact de
    sklearn f1_score(average='macro', zero_division=0, labels=present), mais via
    bincount : le bootstrap l'appelle 10 000 fois par seed."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(np.float64)
    denom = 2.0 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1[cm.sum(1) > 0].mean())


def _lr(c):
    return LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER,
                              random_state=SEED)


def load_split(kind, path, seed):
    """Renvoie {split: (X, y)} en schéma 11 classes."""
    out = {}
    if kind == "ft":
        base = p(path.format(s=seed))
        for s in ("train", "val", "test"):
            X = np.load(os.path.join(base, f"{s}.npy")).astype(np.float32)
            y = np.load(os.path.join(base, f"{s}_labels.npy")).astype(np.int64).ravel()
            out[s] = (X, y)
    else:
        for s in ("train", "val", "test"):
            X = np.load(p("embeddings", f"{path}_{s}.npy")).astype(np.float32)
            y = np.load(p("embeddings", f"{path}_{s}_labels.npy")).astype(np.int64).ravel()
            keep = y != 7                      # RHOL, absente de val et test
            X, y = X[keep], y[keep].copy()
            y[y > 7] -= 1                      # recompacte les indices
            out[s] = (X, y)
    return out


def fit_seed(kind, path, seed, best_c):
    """Probe canonique → (y_test, y_pred, best_C). Sélectionne C sur val si besoin."""
    d = load_split(kind, path, seed)
    sc = StandardScaler()
    xtr = sc.fit_transform(d["train"][0])
    xva, xte = sc.transform(d["val"][0]), sc.transform(d["test"][0])
    ytr, yva, yte = d["train"][1], d["val"][1], d["test"][1]

    if best_c is None:
        from joblib import Parallel, delayed

        def one(c):
            clf = _lr(c).fit(xtr, ytr)
            # sélection sur les 11 classes (f1_macro_all), comme src.probe
            from sklearn.metrics import f1_score
            return c, f1_score(yva, clf.predict(xva), average="macro",
                               zero_division=0, labels=list(range(N_CLASSES)))

        res = Parallel(n_jobs=len(C_GRID), backend="loky")(
            delayed(one)(c) for c in C_GRID)
        best_c = max(res, key=lambda t: t[1])[0]
        print(f"      grille C → best_C={best_c}  "
              + "  ".join(f"{c:g}:{f:.4f}" for c, f in res), flush=True)

    clf = _lr(best_c).fit(xtr, ytr)
    return yte, clf.predict(xte), best_c


def collect_predictions():
    """Fit de chaque seed, avec cache disque : relancer ne refait rien d'acquis."""
    cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    for name, kind, path, seeds, best_c in GROUPS:
        for s in seeds:
            key = f"{name}|seed{s}" if s is not None else name
            if f"{key}|pred" in cache:
                print(f"  [cache] {key}", flush=True)
                continue
            t = time.time()
            yt, yp, used_c = fit_seed(kind, path, s, best_c)
            cache[f"{key}|true"], cache[f"{key}|pred"] = yt, yp
            cache[f"{key}|C"] = np.array([used_c])
            np.savez_compressed(CACHE, **cache)
            print(f"  [fit] {key:32s} C={used_c:<8g} F1={f1_pres(yt, yp):.4f} "
                  f"({time.time() - t:.0f}s)", flush=True)
    return cache


def main():
    ensure_out()
    t0 = time.time()
    print(f"== Palier compétitif : {len(GROUPS)} groupes, "
          f"{len(GROUPS) * (len(GROUPS) - 1) // 2} paires ==\n")
    print("=== PROBES ===", flush=True)
    cache = collect_predictions()

    n_tiles = len(cache[f"{GROUPS[0][0]}|seed0|true"])
    # Mêmes indices que la campagne à 8 groupes : ordre de tirage identique, donc
    # les distributions bootstrap des groupes déjà publiés se reproduisent.
    rng = np.random.RandomState(SEED)
    tile_idx = rng.randint(0, n_tiles, size=(N_BOOT, n_tiles), dtype=np.int32)
    seed_idx = rng.randint(0, 2**30, size=N_BOOT, dtype=np.int32)

    print(f"\n=== BOOTSTRAP (n={N_BOOT}, {n_tiles} tuiles) ===", flush=True)
    seed_dist, seed_obs = {}, {}
    for name, _k, _p, seeds, _c in GROUPS:
        for s in seeds:
            key = f"{name}|seed{s}" if s is not None else name
            yt, yp = cache[f"{key}|true"], cache[f"{key}|pred"]
            seed_obs[key] = f1_pres(yt, yp)
            d = np.empty(N_BOOT)
            for b in range(N_BOOT):
                i = tile_idx[b]
                d[b] = f1_pres(yt[i], yp[i])
            seed_dist[key] = d
            print(f"  {key:34s} obs={seed_obs[key]:.4f}", flush=True)

    stats, dists, obs = {}, {}, {}
    for name, _k, _p, seeds, _c in GROUPS:
        keys = [f"{name}|seed{s}" if s is not None else name for s in seeds]
        if len(keys) == 1:
            d, o = seed_dist[keys[0]], seed_obs[keys[0]]
        else:
            d = np.array([seed_dist[keys[seed_idx[b] % len(keys)]][b]
                          for b in range(N_BOOT)])
            o = float(np.mean([seed_obs[k] for k in keys]))
        dists[name], obs[name] = d, o
        stats[name] = {"observed": o, "mean": float(d.mean()),
                       "std": float(d.std(ddof=1)),
                       "ci95_low": float(np.percentile(d, 2.5)),
                       "ci95_high": float(np.percentile(d, 97.5))}

    print("\n=== GROUPES ===")
    for name in sorted(stats, key=lambda n: -stats[n]["observed"]):
        s = stats[name]
        print(f"  {name:22s} F1={s['observed']:.4f}  "
              f"IC95=[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]")

    pairs = {}
    for a, b in itertools.combinations([g[0] for g in GROUPS], 2):
        delta = obs[a] - obs[b]
        db = dists[a] - dists[b]
        pairs[f"{a} : {b}"] = {
            "model_a": a, "model_b": b,
            "delta_observed_a_minus_b": float(delta),
            "p_a_gt_b": float(np.mean(dists[a] > dists[b])),
            "p_two_sided": float(np.mean(np.abs(db - delta) >= np.abs(delta))),
            "ci95_a": [stats[a]["ci95_low"], stats[a]["ci95_high"]],
            "ci95_b": [stats[b]["ci95_low"], stats[b]["ci95_high"]],
            "ci95_disjoint": bool(stats[a]["ci95_low"] > stats[b]["ci95_high"]
                                  or stats[b]["ci95_low"] > stats[a]["ci95_high"]),
            "method": "paired bootstrap (shared tile + seed indices)",
        }

    # Benjamini-Hochberg sur TOUTES les paires du palier
    order = sorted(pairs.items(), key=lambda kv: kv[1]["p_two_sided"])
    m, max_rank = len(order), 0
    for rank, (_k, v) in enumerate(order, 1):
        v["bh_rank"], v["bh_threshold"] = rank, 0.05 * rank / m
        if v["p_two_sided"] <= v["bh_threshold"]:
            max_rank = rank
    for _k, v in order:
        v["bh_reject"] = v["bh_rank"] <= max_rank
    n_rej = sum(v["bh_reject"] for v in pairs.values())

    print(f"\n=== PAIRES : {n_rej}/{m} rejetées par Benjamini-Hochberg ===")
    for k, v in order:
        if v["bh_reject"]:
            print(f"  {k:46s} Δ={v['delta_observed_a_minus_b']:+.4f} "
                  f"p={v['p_two_sided']:.4f}")

    out = {
        "n_bootstrap": N_BOOT, "seed": SEED, "metric": "f1_macro_pres",
        "schema": "11cls without_rhol", "n_tiles": n_tiles,
        "method": "paired hierarchical bootstrap (shared tile + seed indices across all groups)",
        "population": f"palier compétitif — les {TIER_K} premiers modèles par F1 "
                      "(registry.TIER_K, rupture mesurée du classement)",
        "bh_alpha": 0.05, "bh_n_rejected": n_rej,
        "groups": [{"name": g[0],
                    "seed_keys": [f"{g[0]}|seed{s}" if s is not None else g[0]
                                  for s in g[3]],
                    "n_seeds": len(g[3]),
                    "best_C": [float(cache[f"{g[0]}|seed{s}|C" if s is not None
                                           else f"{g[0]}|C"][0]) for s in g[3]],
                    "observed_f1_per_seed": [
                        seed_obs[f"{g[0]}|seed{s}" if s is not None else g[0]]
                        for s in g[3]],
                    "stats": stats[g[0]]} for g in GROUPS],
        "pairs": pairs,
    }
    fp = p("results", "significance_matrix_tier.json")
    json.dump(out, open(fp, "w"), indent=1, ensure_ascii=False)
    print(f"\n[OK] {fp} — {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
