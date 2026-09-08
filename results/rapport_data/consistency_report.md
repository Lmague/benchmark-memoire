# Contrôle de cohérence — `results/rapport_data/`

15/15 contrôles passent.

| Contrôle | Statut | Détail |
|---|---|---|
| F1 canoniques du registre == all_models_canonical_merged.json | ✅ | — |
| écart probe interne / probe canonique < 0,02 à 100 % | ✅ | max |Δ| = 0.0043 |
| eff_rank_sigma2 == colonne RankMe de geometrie.tex (±3 %) | ✅ | — |
| LoRA r=8 identifié (eff_rank σ² ≈ 43,7) | ✅ | obtenu 43.7 |
| total tuiles train (12 classes) == 49 433 | ✅ | obtenu 49433 |
| total tuiles train (11 classes, sans RHOL) == 49 281 | ✅ | obtenu 49281 |
| RHOL absente de val et test | ✅ | val=0 test=0 |
| total tuiles test == 17 598 | ✅ | obtenu 17598 |
| aucune valeur dépréciée dans rapport_bouguessa/*.tex | ✅ | — |
| le bootstrap apparié couvre le palier (35 modèles, sauf 2 sans embeddings) | ✅ | 33 groupes, 528 paires |
| les 8 groupes communs reproduisent la campagne publiée | ✅ | écart max 0.00027 |
| aucune occurrence d'ExPLoRA dans les sources LaTeX | ✅ | — |
| aucun modèle déprécié (seed unique) dans rapport_bouguessa/*.tex | ✅ | — |
| fragments de tableaux équilibrés (begin/end par environnement) | ✅ | — |
| toutes les figures référencées existent | ✅ | — |

## Écart probe interne (run) vs probe canonique, à 100 %

| Modèle | Δ (interne − canonique) |
|---|---|
| `dinov3_vitl16_lvd_lora` | +0.0043 |
| `vitb16_full_old` | +0.0039 |
| `simdinov2_vitb16_lora_r8_qkv` | +0.0024 |
| `vitb16_scratch_old` | -0.0012 |
| `dinov3_vitb16_lvd_mhsa` | -0.0009 |
| `dinov3_vitb16_lvd_lora_r8` | -0.0006 |
| `dinov3_vitb16_lvd_full` | +0.0003 |
| `resnet50_fulft_sota` | +0.0002 |
| `simdinov2_vitb16_norm_tuning` | -0.0002 |
| `vitb16_mhsa_old` | -0.0002 |
| `simdinov2_vitb16_lora_r8_b911` | +0.0001 |
| `simdinov2_vitb16_lora_r16_rslora` | -0.0001 |
| `simdinov2_vitl16_lora` | -0.0001 |
| `simdinov2_vitb16_lora_r32` | +0.0001 |
| `simdinov2_vitb16_lora_r16_s2` | +0.0001 |
| `simdinov2_vitb16_lora_r8_s2` | +0.0001 |
| `simdinov2_vitb16_lora_r2` | +0.0001 |
| `simdinov2_vitb16_mhsa` | -0.0001 |
| `simdinov2_vitb16_lora_r8_b05` | +0.0001 |
| `simdinov2_vitb16_lora_r8_rslora` | +0.0000 |
| `simdinov2_vitb16_full` | -0.0000 |
| `simdinov2_vitb16_lora_r8_b611` | -0.0000 |
| `simdinov2_vitb16_lora_r16` | +0.0000 |
| `simdinov2_vitb16_lora_r4` | +0.0000 |
| `simdinov2_vitb16_lora` | -0.0000 |

Les tableaux maîtres citent le **probe canonique** ; les courbes de données citent le **probe interne du run** (protocole homogène le long de la courbe).
