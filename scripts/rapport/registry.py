#!/usr/bin/env python3
"""Registre central des modèles et des familles de runs — dossier `rapport_bouguessa/`.

Un seul endroit où sont déclarés :
  - le nom d'affichage canonique de chaque modèle,
  - son type (frozen / ft_fresh / ft_old),
  - l'initialisation du backbone,
  - le pipeline d'entraînement dont il provient (AGENTS.md §4.2 : ne jamais conflater
    `results/datacurve/`, `sota_screening/` et `dinov3b_lora8`),
  - où trouver ses runs (`metrics.json`) et ses embeddings test.

Importé par tous les scripts de `scripts/rapport/`.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "rapport_data")

CLASSES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
              "RUBC", "SEDG", "TUSS", "WILL"]
# Classes jamais apprenables (support test trop faible) — F1 ≡ 0 partout
CLASSES_DEAD = ["ARCA", "DRYI", "RUBC"]
CLASSES_8 = ["ALDE", "BIRC", "LICH", "MOSS", "PETF", "SEDG", "TUSS", "WILL"]

# ── Familles de runs fine-tunés (un metrics.json par run) ─────────────────────
# key: (display, runs_glob_dir, embeddings_dir, type, init, pipeline)
RUN_FAMILIES = {
    "vitb16_full_old": (
        "ViT-B/16 IN-Full", "sota_screening/full/runs",
        "sota_screening/full/embeddings", "ft_old", "ImageNet", "sota_screening"),
    "vitb16_mhsa_old": (
        "ViT-B/16 IN-MHSA", "sota_screening/mhsa/runs",
        "sota_screening/mhsa/embeddings", "ft_old", "ImageNet", "sota_screening"),
    "vitb16_scratch_old": (
        "ViT-B/16 IN-Scratch", "sota_screening/scratch/runs",
        "sota_screening/scratch/embeddings", "ft_old", "aléatoire", "sota_screening"),
    "dinov3_vitb16_lvd_lora_r8": (
        # Révision 2026-07-29 : la courbe de données (multi-fractions) vit dans
        # DINOv3_LoRA/lora/embeddings/dinov3_vitb16_lvd_lora_frac{000..070}_seed*
        # (7 fractions × 3 seeds). Le 100 % canonique reste dans
        # DINOv3_LoRA_8/embeddings/..._explora_frac100_seed* et est alimenté
        # séparément via model_jobs() (geometry_test_fixed_n.py:171).
        "DINOv3 ViT-B/16 LoRA r=8", "results/datacurve_lora/runs",
        "DINOv3_LoRA/lora/embeddings", "ft_fresh", "DINOv3-B", "dinov3b_lora8",
        "frac", "dinov3_vitb16_lvd_lora"),
    "dinov3_vitb16_lvd_full": (
        "DINOv3 ViT-B/16 Full", "ft_ssl_results/dinov3_vitb16_lvd_full_runs",
        "ft_ssl_results/dinov3_vitb16_lvd_full_embeddings", "ft_fresh", "DINOv3-B", "ft_ssl"),
    "dinov3_vitb16_lvd_mhsa": (
        "DINOv3 ViT-B/16 MHSA", "ft_ssl_results/dinov3_vitb16_lvd_mhsa_runs",
        "ft_ssl_results/dinov3_vitb16_lvd_mhsa_embeddings", "ft_fresh", "DINOv3-B", "ft_ssl"),
    "simdinov2_vitb16_full": (
        "SimDINOv2 ViT-B/16 Full", "ft_ssl_results/simdinov2_vitb16_full_runs",
        "ft_ssl_results/simdinov2_vitb16_full_embeddings", "ft_fresh", "SimDINOv2-B", "ft_ssl"),
    "simdinov2_vitb16_mhsa": (
        "SimDINOv2 ViT-B/16 MHSA", "ft_ssl_results/simdinov2_vitb16_mhsa_runs",
        "ft_ssl_results/simdinov2_vitb16_mhsa_embeddings", "ft_fresh", "SimDINOv2-B", "ft_ssl"),
    "resnet50_fulft_sota": (
        "ResNet-50 IN-FT", "runs",
        None, "ft_old", "ImageNet", "sota_screening"),

    # ── LoRA r=8, ajout 2026-07-28 : DINOv3-L, SimDINOv2-B, SimDINOv2-L ──────
    # Les 3 runs partagent `ViTB_L_LoRA/lora_3models/runs/` (glob nu = collision) ;
    # les embeddings de dinov3_vitl16_lvd_lora et simdinov2_vitl16_lora partagent
    # `.../embeddings/` (même piège), mais ceux de simdinov2_vitb16_lora vivent
    # dans un sous-dossier DÉDIÉ à nommage différent (`simdinov2_vitb16/embeddings/
    # frac100_seed{N}/`, SANS préfixe de modèle) — vérifié sur disque 2026-07-28.
    # `runs_prefix`/`emb_prefix` (7e/8e éléments, optionnels) filtrent le glob
    # nu ("*") des consommateurs quand le dossier est partagé ; laisser vide
    # quand le dossier est déjà dédié (comportement historique inchangé).
    "dinov3_vitl16_lvd_lora": (
        "DINOv3 ViT-L/16 LoRA r=8", "ViTB_L_LoRA/lora_3models/runs",
        "ViTB_L_LoRA/lora_3models/embeddings", "ft_fresh", "DINOv3-L", "lora_3models",
        "dinov3_vitl16_lvd_lora", "dinov3_vitl16_lvd_lora"),
    "simdinov2_vitb16_lora": (
        "SimDINOv2 ViT-B/16 LoRA r=8", "ViTB_L_LoRA/lora_3models/runs",
        "ViTB_L_LoRA/lora_3models/simdinov2_vitb16/embeddings", "ft_fresh",
        "SimDINOv2-B", "lora_3models", "simdinov2_vitb16_lora", ""),
    "simdinov2_vitl16_lora": (
        "SimDINOv2 ViT-L/16 LoRA r=8", "ViTB_L_LoRA/lora_3models/runs",
        "ViTB_L_LoRA/lora_3models/embeddings", "ft_fresh", "SimDINOv2-L", "lora_3models",
        "simdinov2_vitl16_lora", "simdinov2_vitl16_lora"),
    # ── ajout 2026-08-31 : self-distillation contexte→tuile (résultats locaux) ──
    "ctxdistill_dB_tL": (
        "Contexte R2 (distil. B, fusion 1536)", None, None, "ft_fresh", "Contexte",
        "context_distill"),
    "ctxdistill_dA_tL": (
        "Contexte R1 (distil. A, tuile 768)", None, None, "ft_fresh", "Contexte",
        "context_distill"),
    "ctxdistill_dA_tEMA": (
        "Contexte R3 (distil. A, EMA)", None, None, "ft_fresh", "Contexte",
        "context_distill"),
    # ── ajout 2026-09 : ablation LoRA/PEFT SimDINOv2-B, Stage A (42 runs) ─────
    # Tous les bras partagent results/lora_simb_ablation/{runs,embeddings} (sauf
    # QKV : results/lora_simb_ablation_qkv/*) ; `runs_prefix`/`emb_prefix` filtrent
    # le glob par stem de tag. Pipeline dédié "simb_stageA" (AGENTS.md §4.2).
    "simdinov2_vitb16_lora_r8_b611": (
        "SimDINOv2-B LoRA r8 (blocs 6-11)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a8_b67891011_frac100",
        "simdinov2_vitb16_lora_r8a8_b67891011_frac100"),
    "simdinov2_vitb16_lora_r8_b05": (
        "SimDINOv2-B LoRA r8 (blocs 0-5)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a8_b012345_frac100",
        "simdinov2_vitb16_lora_r8a8_b012345_frac100"),
    "simdinov2_vitb16_lora_r8_b911": (
        "SimDINOv2-B LoRA r8 (blocs 9-11)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a8_b91011_frac100",
        "simdinov2_vitb16_lora_r8a8_b91011_frac100"),
    "simdinov2_vitb16_lora_r2": (
        "SimDINOv2-B LoRA r2", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r2a2_frac100",
        "simdinov2_vitb16_lora_r2a2_frac100"),
    "simdinov2_vitb16_lora_r4": (
        "SimDINOv2-B LoRA r4", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r4a4_frac100",
        "simdinov2_vitb16_lora_r4a4_frac100"),
    "simdinov2_vitb16_lora_r16": (
        "SimDINOv2-B LoRA r16", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r16a16_frac100",
        "simdinov2_vitb16_lora_r16a16_frac100"),
    "simdinov2_vitb16_lora_r32": (
        "SimDINOv2-B LoRA r32", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r32a32_frac100",
        "simdinov2_vitb16_lora_r32a32_frac100"),
    "simdinov2_vitb16_lora_r8_s2": (
        "SimDINOv2-B LoRA r8 (scaling 2)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a16_frac100",
        "simdinov2_vitb16_lora_r8a16_frac100"),
    "simdinov2_vitb16_lora_r16_s2": (
        "SimDINOv2-B LoRA r16 (scaling 2)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r16a32_frac100",
        "simdinov2_vitb16_lora_r16a32_frac100"),
    "simdinov2_vitb16_lora_r8_rslora": (
        "SimDINOv2-B LoRA r8 (rsLoRA)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a22_frac100",
        "simdinov2_vitb16_lora_r8a22_frac100"),
    "simdinov2_vitb16_lora_r16_rslora": (
        "SimDINOv2-B LoRA r16 (rsLoRA)", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r16a64_frac100",
        "simdinov2_vitb16_lora_r16a64_frac100"),
    "simdinov2_vitb16_lora_r8_qkv": (
        "SimDINOv2-B LoRA r8 (Q+K+V)", "results/lora_simb_ablation_qkv/runs",
        "results/lora_simb_ablation_qkv/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_lora_r8a8_frac100",
        "simdinov2_vitb16_lora_r8a8_frac100"),
    "simdinov2_vitb16_norm_tuning": (
        "SimDINOv2-B NormTuning", "results/lora_simb_ablation/runs",
        "results/lora_simb_ablation/embeddings", "ft_fresh", "SimDINOv2-B",
        "simb_stageA", "simdinov2_vitb16_norm_tuning_frac100",
        "simdinov2_vitb16_norm_tuning_frac100"),
}


def runs_prefix(key: str) -> str:
    """Préfixe de glob pour `runs_glob_dir` — vide = dossier dédié (comportement
    historique, glob nu "*")."""
    t = RUN_FAMILIES[key]
    return t[6] if len(t) > 6 else ""


def emb_prefix(key: str) -> str:
    """Idem pour `embeddings_dir`."""
    t = RUN_FAMILIES[key]
    return t[7] if len(t) > 7 else ""

# Régimes de la courbe de données (4 régimes comparables)
DATACURVE_REGIMES = ["vitb16_full_old", "vitb16_mhsa_old",
                     "vitb16_scratch_old", "dinov3_vitb16_lvd_lora_r8"]

# ── Modèles exclus des rapports (2026-07-26) ─────────────────────────────────
# Les deux régimes « ExPLoRA » reposent sur une implémentation invalide : ils sont
# retirés de tous les documents de `rapport_bouguessa/`. Leurs runs et embeddings
# restent sur disque (`sota_screening/explora/`, `DINOv3_ExPLoRA_Like/`,
# `embeddings/dinov3_vitb16_lvd_explora_frac100_seed*`) et les JSON canoniques ne
# sont pas modifiés : cet ensemble sert à filtrer les sources brutes que les scripts
# de rapport lisent sans passer par CANONICAL_F1 / RUN_FAMILIES.
#
# ATTENTION : `DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100_seed*`
# porte le même nom de dossier mais EST LoRA r=8 (AGENT_MEMORY.md, 2026-07-22).
EXCLUDED_MODELS = {"vitb16_explora_old", "dinov3_vitb16_lvd_explora"}

# ── Modèles gelés (embeddings sous embeddings/{key}_test.npy) ─────────────────
FROZEN_MODELS = {
    "dinov3_vitl16_lvd":      ("DINOv3 ViT-L/16 LVD", 1024),
    "dinov3_vith16plus_lvd":  ("DINOv3 ViT-H+/16 LVD", 1280),
    "simdinov2_vitl16":       ("SimDINOv2 ViT-L/16", 1024),
    "simdinov2_vitb16":       ("SimDINOv2 ViT-B/16", 768),
    "dinov3_vitb16_lvd":      ("DINOv3 ViT-B/16 LVD", 768),
    "dinov3_vits16_lvd":      ("DINOv3 ViT-S/16 LVD", 384),
    "dinov3_vitl16_sat":      ("DINOv3 ViT-L/16 SAT", 1024),
    "vitb16_imagenet":        ("ViT-B/16 IN-1k", 768),
    "scalemae_vitl16":        ("ScaleMAE ViT-L/16", 1024),
    "satmae_vitl16":          ("SatMAE ViT-L/16", 1024),
    "resnet50_imagenet":      ("ResNet-50 IN-1k", 2048),
}

# Modèles dépréciés (seed unique) — AGENT_MEMORY.md §RÉSULTATS DÉPRÉCIÉS
DEPRECATED = {"vitb16_fulft_arctic", "vitb16_arctic", "resnet50_arctic"}

# ── Valeurs canoniques de contrôle (AGENTS.md §3, AGENT_MEMORY.md) ───────────
CANONICAL_F1 = {
    "dinov3_vitb16_lvd_lora_r8": (0.4835, 0.0011),
    "dinov3_vitb16_lvd_mhsa":    (0.4800, 0.0020),
    "dinov3_vitl16_lvd":         (0.4792, None),
    "dinov3_vitb16_lvd_full":    (0.4786, 0.0027),
    "simdinov2_vitl16":          (0.4760, None),
    "simdinov2_vitb16":          (0.4723, None),
    "dinov3_vitb16_lvd":         (0.4712, None),
    "vitb16_full_old":           (0.4708, 0.0079),
    "simdinov2_vitb16_mhsa":     (0.4691, 0.0022),
    "vitb16_mhsa_old":           (0.4689, 0.0050),
    "simdinov2_vitb16_full":     (0.4782, 0.0074),
    "dinov3_vitl16_sat":         (0.4620, None),
    "resnet50_fulft_sota":       (0.4571, 0.0033),
    "vitb16_imagenet":           (0.4500, None),
    "scalemae_vitl16":           (0.4480, None),
    "satmae_vitl16":             (0.4091, None),
    "resnet50_imagenet":         (0.4076, None),
    "vitb16_scratch_old":        (0.3561, 0.0133),
    # ── ajout 2026-07-28, valeurs calculées par scripts/probe_lora3_new_runs.py ──
    "simdinov2_vitl16_lora":     (0.4835, 0.0068),
    "dinov3_vitl16_lvd_lora":    (0.4799, 0.0030),
    "simdinov2_vitb16_lora":     (0.4781, 0.0028),
    # ── ajout 2026-07-30, valeur calculée par probe.py (cf. scripts/add_dinov3_hplus_canonical.py) ──
    "dinov3_vith16plus_lvd":     (0.4805, None),
    # ── ajout 2026-08-27 DINOv3 ViT-S/16 (LVD) : LoRA r=8 3 seeds, JOB Narval 1872074 (log) ──
    "dinov3_vits16_lvd_lora_r8": (0.4774, 0.0022),
    "dinov3_vits16_lvd":         (0.4689, None),
    # ── ajout 2026-08-31 : self-distillation contexte→tuile (results/context_distill) ──
    "ctxdistill_dB_tL":          (0.5080, 0.0013),   # Design B fusionné 1536 (non déployable)
    "ctxdistill_dA_tL":          (0.4870, 0.0011),   # Design A tuile 768
    "ctxdistill_dA_tEMA":        (0.4836, 0.0014),   # Design A EMA-self
    # ── ajout 2026-09 : Stage A SimDINOv2-B (13 bras, reprobe canonique
    # results/simb_stageA_probe_CANONICAL.json) + 2 SimDINOv2-B entraînés
    # Design B (metrics.json, même provenance que ctxdistill R1/R2/R3) ──
    'simdinov2_vitb16_lora_r8_b611': (0.4802, 0.0027),
    'simdinov2_vitb16_lora_r8_b05': (0.4770, 0.0024),
    'simdinov2_vitb16_lora_r8_b911': (0.4804, 0.0002),
    'simdinov2_vitb16_lora_r2': (0.4793, 0.0024),
    'simdinov2_vitb16_lora_r4': (0.4785, 0.0049),
    'simdinov2_vitb16_lora_r16': (0.4769, 0.0016),
    'simdinov2_vitb16_lora_r32': (0.4750, 0.0017),
    'simdinov2_vitb16_lora_r8_s2': (0.4760, 0.0018),
    'simdinov2_vitb16_lora_r16_s2': (0.4749, 0.0026),
    'simdinov2_vitb16_lora_r8_rslora': (0.4753, 0.0023),
    'simdinov2_vitb16_lora_r16_rslora': (0.4737, 0.0035),
    'simdinov2_vitb16_lora_r8_qkv': (0.4788, 0.0015),
    'simdinov2_vitb16_norm_tuning': (0.4783, 0.0004),
    'ctxdistill_dB_tSL_r2a4': (0.5030, 0.0012),  # metrics.json, pas de bootstrap (NO_BOOTSTRAP)
    'ctxdistill_dB_tSL_r8a16': (0.5057, 0.0072),  # metrics.json, pas de bootstrap (NO_BOOTSTRAP)
}

# Valeurs jamais réintroduites (AGENTS.md §4.1, AGENT_MEMORY.md)
# NB : les F1 des régimes ExPLoRA exclus (0,4816 / 0,4668) ne sont PAS listés ici —
# ces valeurs apparaissent légitimement ailleurs (IC95 de SimDINOv2 ViT-L, run LoRA à
# 10 %), un contrôle par valeur ferait des faux positifs. La garde contre leur retour
# est le contrôle de chaîne `c_no_explora()` de check_consistency.py, qui couvre aussi
# `tables/`.
FORBIDDEN_VALUES = ["0.5256", "0.4675", "0.4680"]

# ── Palier compétitif ────────────────────────────────────────────────────────
# Le « top-8 » des campagnes antérieures était une coupe de RANG héritée de l'époque
# où le benchmark comptait 12 modèles (`scripts/silhouette_top8_test.py:140`,
# `scripts/compute_correlations.py:420` : `models_sorted[:8]`), pas une rupture
# mesurée. Elle tombait entre deux modèles séparés de 0,0004 — l'écart le plus petit
# du classement.
#
# K = 14 est la rupture empirique du classement à 21 modèles (recalculée 2026-07-28
# après l'ajout des 3 régimes LoRA r=8 DINOv3-L/SimDINOv2-B/SimDINOv2-L, qui entrent
# tous les trois dans le palier sans en déplacer aucun membre) : les rangs 1 à 14
# forment un bloc continu (0,4835 → 0,4689, étendue 0,0146) et l'écart au rang 15
# vaut +0,0069, soit ≈4,3 fois l'écart médian entre rangs consécutifs (0,0016).
# Vérifiable par `tier_break()` ci-dessous.
#
# Recalculé 2026-07-29 après correction de SimDINOv2-B Full (0,4677→0,4782,
# cf. src/probe.py : la valeur publiée utilisait un best_C non optimal sur
# validation pour 2 des 3 seeds). SimDINOv2-B Full passe du rang 14 (borne basse
# du palier) au rang 7 ; vitb16_mhsa_old (déjà dans le palier, rang 13→14) devient
# la nouvelle borne basse. TIER_K reste 14 : la composition du palier ne change
# pas, seul l'ordre interne et les bornes exactes changent.
#
# Recalculé 2026-07-30 après l'ajout de DINOv3 ViT-H+/16 (F1=0,4805, rang 3).
# Contrairement aux recalculs précédents, ce nouveau membre déplace la composition :
# il s'insère au-dessus de vitb16_mhsa_old (rang 14→15), qui sortirait du palier si
# TIER_K restait à 14. `tier_break()` confirme que la rupture empirique suit le même
# écart qu'avant (rang 15→16, 0,4689→0,4620, +0,0069) mais un rang plus bas :
# TIER_K passe donc à 15 pour garder la même composition + le nouveau membre.
#
# Recalculé 2026-08-31 après l'ajout des 3 modèles de contexte spatial (R1/R2/R3) et
# de DINOv3 ViT-S/16 (gelé + LoRA r=8). Le bloc continu va désormais du rang 1
# (Contexte R2, 0,5080) au rang 20 (ViT-B/16 IN-MHSA, 0,4689) ; le décrochage vaut
# toujours +0,0069 (0,4689 → 0,4620, DINOv3 ViT-L/16 SAT). Le très grand écart
# R2→R1 (+0,021) est un décrochage de tête de liste — R2 est un design non déployable
# (borne supérieure théorique, contexte requis à l'inférence), pas la fin du bloc
# compétitif : `tier_break()` le saute. TIER_K passe donc à 20.
#
# Recalculé 2026-09 après l'ajout de l'ablation LoRA/PEFT SimDINOv2-B (13 bras,
# 0,4737 → 0,4804) et des 2 SimDINOv2-B entraînés Design B (0,5030 / 0,5057).
# Tous tombent DANS le bloc (min 0,4737 > borne basse 0,4689 ; l'écart
# SimB-r8a16→SimB-r2a4→R1, 0,0160, est sauté comme décrochage de tête par
# `tier_break(outlier_gap=0.01)`, même statut que R2). Le décrochage structurel
# reste 0,4689 → 0,4620 (+0,0069) : TIER_K passe à 35. Les 2 SimDINOv2-B entraînés
# n'ont pas d'embeddings rapatriés → exclus du bootstrap (NO_BOOTSTRAP_EMBEDDINGS) ;
# le bootstrap couvre donc 33 groupes.
TIER_K = 35
TIER_POP = f"top{TIER_K}"

# Nom de groupe du bootstrap apparié (`significance_tier.GROUPS`) → clé du registre.
# Deux conventions de probe coexistent et ne doivent JAMAIS être mélangées dans un
# même Δ : le probe linéaire canonique (`all_models_canonical_merged.json`,
# CANONICAL_F1) et le probe interne des runs, qui est celui du bootstrap
# (`significance_matrix_tier.json`). Cette table permet aux générateurs de mesurer
# l'écart entre les deux et de l'annoncer en note de tableau, plutôt que de le
# laisser deviner. Voir `make_tables._conv_gaps()`.
TIER_GROUP_KEY = {
    "DINOv3-B LoRA r=8":    "dinov3_vitb16_lvd_lora_r8",
    "DINOv3-B MHSA":        "dinov3_vitb16_lvd_mhsa",
    "DINOv3-B Full":        "dinov3_vitb16_lvd_full",
    "DINOv3-L LoRA r=8":    "dinov3_vitl16_lvd_lora",
    "SimDINOv2-B LoRA r=8": "simdinov2_vitb16_lora",
    "SimDINOv2-L LoRA r=8": "simdinov2_vitl16_lora",
    "SimDINOv2-B MHSA":     "simdinov2_vitb16_mhsa",
    "SimDINOv2-B Full":     "simdinov2_vitb16_full",
    "ViT-B/16 IN-Full":     "vitb16_full_old",
    "ViT-B/16 IN-MHSA":     "vitb16_mhsa_old",
    "DINOv3 ViT-L16":       "dinov3_vitl16_lvd",
    "DINOv3 ViT-H+16":      "dinov3_vith16plus_lvd",
    "DINOv3 ViT-B16":       "dinov3_vitb16_lvd",
    "SimDINOv2 ViT-L16":    "simdinov2_vitl16",
    "SimDINOv2 ViT-B16":    "simdinov2_vitb16",
    "Contexte R2 (B, fusion)": "ctxdistill_dB_tL",
    "Contexte R1 (A, tuile)":  "ctxdistill_dA_tL",
    "Contexte R3 (A, EMA)":    "ctxdistill_dA_tEMA",
    "DINOv3 ViT-S/16 LoRA r=8": "dinov3_vits16_lvd_lora_r8",
    "DINOv3 ViT-S/16":          "dinov3_vits16_lvd",
    # ── ajout 2026-09 : ablation LoRA/PEFT SimDINOv2-B, Stage A (13 bras) ──
    "SimDINOv2-B LoRA r8 (blocs 6-11)": "simdinov2_vitb16_lora_r8_b611",
    "SimDINOv2-B LoRA r8 (blocs 0-5)":  "simdinov2_vitb16_lora_r8_b05",
    "SimDINOv2-B LoRA r8 (blocs 9-11)": "simdinov2_vitb16_lora_r8_b911",
    "SimDINOv2-B LoRA r2":              "simdinov2_vitb16_lora_r2",
    "SimDINOv2-B LoRA r4":              "simdinov2_vitb16_lora_r4",
    "SimDINOv2-B LoRA r16":             "simdinov2_vitb16_lora_r16",
    "SimDINOv2-B LoRA r32":             "simdinov2_vitb16_lora_r32",
    "SimDINOv2-B LoRA r8 (scaling 2)":  "simdinov2_vitb16_lora_r8_s2",
    "SimDINOv2-B LoRA r16 (scaling 2)": "simdinov2_vitb16_lora_r16_s2",
    "SimDINOv2-B LoRA r8 (rsLoRA)":     "simdinov2_vitb16_lora_r8_rslora",
    "SimDINOv2-B LoRA r16 (rsLoRA)":    "simdinov2_vitb16_lora_r16_rslora",
    "SimDINOv2-B LoRA r8 (Q+K+V)":      "simdinov2_vitb16_lora_r8_qkv",
    "SimDINOv2-B NormTuning":           "simdinov2_vitb16_norm_tuning",
}

# ── Noms d'affichage des groupes du bootstrap ───────────────────────────────
# Les noms bruts du JSON (`significance_matrix_tier.json`) sont des identifiants de
# runs ("Contexte R2 (B, fusion)", "DINOv3 ViT-B16", "ViT-B/16 IN-MHSA") : lisibles
# dans le code, pas dans un tableau — ils ne disent ni le backbone (DINOv3-B ?
# SimDINOv2-B ?), ni l'état (gelé ou affiné). Cette table fournit POUR CHAQUE
# GROUPE : (nom long — backbone + régime + état, pour les tableaux t_ci,
# t_signif_bh ; nom court — compact avec marqueur (gelé), pour les axes des
# matrices de significativité). Les générateurs doivent passer par là ; les
# LOOKUPS internes (TIER_GROUP_KEY, paires du JSON) restent sur le nom brut.
# Révision 2026-09-08 (retour lecteur : « on sait pas quel modèle c'est »).
TIER_GROUP_DISPLAY = {
    # -- contexte spatial (backbone DINOv3-B ; R2 = borne non déployable) --
    "Contexte R2 (B, fusion)": ("DINOv3-B Contexte R2 — affiné, fusion 1536",
                                "D3-B ctx R2 (aff.)"),
    "Contexte R1 (A, tuile)":  ("DINOv3-B Contexte R1 — affiné, tuile seule",
                                "D3-B ctx R1 (aff.)"),
    "Contexte R3 (A, EMA)":    ("DINOv3-B Contexte R3 — affiné, EMA self-distill.",
                                "D3-B ctx R3 (aff.)"),
    # -- famille DINOv3 --
    "DINOv3 ViT-H+16":         ("DINOv3 ViT-H+/16 LVD — gelé",
                                "D3-H+ (gelé)"),
    "DINOv3 ViT-L16":          ("DINOv3 ViT-L/16 LVD — gelé",
                                "D3-L (gelé)"),
    "DINOv3 ViT-B16":          ("DINOv3 ViT-B/16 LVD — gelé",
                                "D3-B (gelé)"),
    "DINOv3 ViT-S/16":         ("DINOv3 ViT-S/16 LVD — gelé",
                                "D3-S (gelé)"),
    "DINOv3-B LoRA r=8":       ("DINOv3 ViT-B/16 — affiné LoRA r=8",
                                "D3-B LoRA r8"),
    "DINOv3-B MHSA":           ("DINOv3 ViT-B/16 — affiné MHSA-only",
                                "D3-B MHSA"),
    "DINOv3-B Full":           ("DINOv3 ViT-B/16 — affiné complet (Full)",
                                "D3-B Full"),
    "DINOv3-L LoRA r=8":       ("DINOv3 ViT-L/16 — affiné LoRA r=8",
                                "D3-L LoRA r8"),
    "DINOv3 ViT-S/16 LoRA r=8": ("DINOv3 ViT-S/16 — affiné LoRA r=8",
                                "D3-S LoRA r8"),
    # -- famille SimDINOv2 (iNat-Plantae) --
    "SimDINOv2 ViT-B16":       ("SimDINOv2 ViT-B/16 — gelé",
                                "SimB (gelé)"),
    "SimDINOv2 ViT-L16":       ("SimDINOv2 ViT-L/16 — gelé",
                                "SimL (gelé)"),
    "SimDINOv2-B LoRA r=8":    ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, tous blocs (ancre)",
                                "SimB r8 tous blocs"),
    "SimDINOv2-L LoRA r=8":    ("SimDINOv2 ViT-L/16 — affiné LoRA r=8",
                                "SimL LoRA r8"),
    "SimDINOv2-B MHSA":        ("SimDINOv2 ViT-B/16 — affiné MHSA-only",
                                "SimB MHSA"),
    "SimDINOv2-B Full":        ("SimDINOv2 ViT-B/16 — affiné complet (Full)",
                                "SimB Full"),
    "SimDINOv2-B NormTuning":  ("SimDINOv2 ViT-B/16 — affiné NormTuning (normes+tête)",
                                "SimB NormTuning"),
    "SimDINOv2-B LoRA r2":     ("SimDINOv2 ViT-B/16 — affiné LoRA r=2, tous blocs",
                                "SimB LoRA r2"),
    "SimDINOv2-B LoRA r4":     ("SimDINOv2 ViT-B/16 — affiné LoRA r=4, tous blocs",
                                "SimB LoRA r4"),
    "SimDINOv2-B LoRA r16":    ("SimDINOv2 ViT-B/16 — affiné LoRA r=16, tous blocs",
                                "SimB LoRA r16"),
    "SimDINOv2-B LoRA r32":    ("SimDINOv2 ViT-B/16 — affiné LoRA r=32, tous blocs",
                                "SimB LoRA r32"),
    "SimDINOv2-B LoRA r8 (blocs 0-5)":   ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, blocs 0-5",
                                "SimB r8 b0-5"),
    "SimDINOv2-B LoRA r8 (blocs 6-11)":  ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, blocs 6-11",
                                "SimB r8 b6-11"),
    "SimDINOv2-B LoRA r8 (blocs 9-11)":  ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, blocs 9-11",
                                "SimB r8 b9-11"),
    "SimDINOv2-B LoRA r8 (Q+K+V)":       ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, cibles Q+K+V",
                                "SimB r8 QKV"),
    "SimDINOv2-B LoRA r8 (scaling 2)":   ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, scaling α=2r",
                                "SimB r8 sc2"),
    "SimDINOv2-B LoRA r16 (scaling 2)":  ("SimDINOv2 ViT-B/16 — affiné LoRA r=16, scaling α=2r",
                                "SimB r16 sc2"),
    "SimDINOv2-B LoRA r8 (rsLoRA)":      ("SimDINOv2 ViT-B/16 — affiné LoRA r=8, rsLoRA",
                                "SimB r8 rsLoRA"),
    "SimDINOv2-B LoRA r16 (rsLoRA)":     ("SimDINOv2 ViT-B/16 — affiné LoRA r=16, rsLoRA",
                                "SimB r16 rsLoRA"),
    # -- famille ImageNet (initialisation IN-1k, affinée) --
    "ViT-B/16 IN-Full":        ("ViT-B/16 IN — affiné complet (Full)",
                                "IN-B Full"),
    "ViT-B/16 IN-MHSA":        ("ViT-B/16 IN — affiné MHSA-only",
                                "IN-B MHSA"),
}

# ── Modèles du palier sans embeddings rapatriés : point-estimates documentés ─
# Leurs F1 (metrics.json) figurent dans CANONICAL_F1 et le tableau maître, mais ils
# ne peuvent pas entrer dans le bootstrap apparié (pas d'embeddings test locaux —
# checkpoints sur $SCRATCH uniquement). check_consistency.py les exclut du contrôle
# de couverture ; toute comparaison les impliquant reste un point-estimate (§4.4).
NO_BOOTSTRAP_EMBEDDINGS = {
    "ctxdistill_dB_tSL_r2a4",    # SimDINOv2-B entraîné Design B, fusion 512
    "ctxdistill_dB_tSL_r8a16",   # idem r8a16
}


def tier_break(min_gap: float = 0.004, outlier_gap: float = 0.01) -> int:
    """Rang du premier décrochage du classement canonique — justifie TIER_K.

    Un écart > `outlier_gap` (ex. Contexte R2 → R1, +0,021) est un décrochage de
    tête de liste (modèle hors-bloc, de type borne théorique), pas la fin du bloc
    compétitif : on le saute et on prend le premier écart >= `min_gap` en dessous.
    """
    f1 = sorted((v[0] for v in CANONICAL_F1.values()), reverse=True)
    for i in range(len(f1) - 1):
        gap = f1[i] - f1[i + 1]
        if gap > outlier_gap:
            continue
        if gap > min_gap:
            return i + 1
    return len(f1)


FRACTION_LABEL = {0.005: "0,5\\,%", 0.01: "1\\,%", 0.05: "5\\,%", 0.1: "10\\,%",
                  0.25: "25\\,%", 0.5: "50\\,%", 0.7: "70\\,%", 1.0: "100\\,%"}


def p(*parts) -> str:
    """Chemin absolu depuis la racine du dépôt."""
    return os.path.join(ROOT, *parts)


def ensure_out() -> str:
    os.makedirs(OUT, exist_ok=True)
    return OUT

# Vues pratiques sur TIER_GROUP_DISPLAY (les générateurs ne manipulent que celles-là).
TIER_DISPLAY_LONG = {k: v[0] for k, v in TIER_GROUP_DISPLAY.items()}
TIER_DISPLAY_SHORT = {k: v[1] for k, v in TIER_GROUP_DISPLAY.items()}
