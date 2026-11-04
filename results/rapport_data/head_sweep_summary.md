# Synthèse des campagnes « têtes » non linéaires

Les campagnes de 2026-09-08/12 sont complètes : 33 groupes tuile et 24 tags fusion. Le juge de paix est le `delta_vs_lbfgs` (sonde canonique), pas le `delta_vs_lin_adamw` (baseline de campagne plus faible).

## Campagne tile (33 groupes)

- Meilleure tête non linéaire > **lbfgs canonique** : 0/33
- Meilleure tête non linéaire > baseline AdamW : 16/33
- Δ vs lbfgs : moyenne -0.0103, max -0.0010 (SimDINOv2-B MHSA)
- Δ vs AdamW : moyenne -0.0009, max +0.0102 (DINOv3 ViT-B16)

## Campagne fused (24 groupes)

- Meilleure tête non linéaire > **lbfgs canonique** : 2/24
- Meilleure tête non linéaire > baseline AdamW : 16/24
- Δ vs lbfgs : moyenne -0.0082, max +0.0076 (dinov3_vits16_FROZEN_fused_ctx512_frac100_seed0)
- Δ vs AdamW : moyenne +0.0024, max +0.0127 (dinov3_vitb16_lvd_FROZEN_fused_ctx1024_frac100_seed0)

## Lecture

- **Vue tuile : 0/33.** Aucune tête non linéaire ne bat la sonde linéaire canonique ; le plateau plat n'est pas un artefact de sonde linéaire.
- **Vue fusion : 2/24**, de ≤ +0,0076 et sur des tags seed-unique sans IC : pas un résultat citable comme gain.
- Le MLP-2 bat la baseline AdamW (16/33, 9/24) mais jamais le lbfgs : l'apparente supériorité non linéaire disparaît à optimisation égale.

Généré par `scripts/rapport/head_sweep_summary.py`.
