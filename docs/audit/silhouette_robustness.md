# TEST DE ROBUSTESSE — Silhouette sur le palier compétitif

Date : 2026-07-15
Script : `scripts/silhouette_top8_test.py`
Résultats : `results/silhouette_robustness_CANONICAL.json` (commit `7cff2fb`)

---

## ÉTAT GIT

```
HEAD   : a14e1c3 (+ 1 commit local = 7cff2fb)
origin : bc220ef (HEAD en avance de 8 commits)
Status : 30 fichiers non commités (préexistants)
Commits récents :
  7cff2fb T1+T2: test robustesse silhouette palier compétitif
  a14e1c3 audit: rapport de cohérence complet
  3fdefc6 T1+T2+T3+mécanique: corrections manuscrit
```

---

## TÂCHE 0 — VÉRIFICATION D'INTÉGRITÉ

### Champs perdus entre `f7bc3e4^` et `f7bc3e4`

Fichier comparé : `git show f7bc3e4^:results/all_scores_consolidated.json` vs `results/all_scores_consolidated.json`.

**12 champs perdus :** `f1_source`, `global_alpha`, `global_cumvar_top100`, `global_cumvar_top20`, `global_effective_rank`, `global_participation_ratio`, `global_stable_rank`, `ideal_cos_etf`, `nc2_deviation_etf`, `nc2_verdict`, `schema`, `split`.

**5 champs critiques pour le Tableau 5 :** `global_effective_rank`, `global_stable_rank`, `global_participation_ratio`, `global_alpha`, `nc2_deviation_etf`.

**2 champs gagnés :** `_provenance_11cls`, `_provenance_12cls`.

### Vérification de cohérence

Les 6 métriques cluster (silhouette, DB, CH, NC1, dim_mle_mean, dim_mle_median) sont **identiques** entre l'ancien et le nouveau fichier (Δ < 0.000001 pour tous les modèles). Les F1 diffèrent massivement — l'ancien fichier stockait des valeurs 12cls sous l'étiquette 11cls (confirmé : `dinov3_vitl16_lvd` old=0.4789 = `f1_macro_pres_12cls`, pas 11cls).

### Sources des 11 métriques pour cette mission

| # | Métrique Tableau 5 | Source |
|---|-------------------|--------|
| 1 | Silhouette | `all_scores_consolidated.json` (nouveau) |
| 2 | Davies-Bouldin | `all_scores_consolidated.json` (nouveau) |
| 3 | Calinski-Harabasz | `all_scores_consolidated.json` (nouveau) |
| 4 | NC1 | `all_scores_consolidated.json` (nouveau) |
| 5 | NC2 dév. ETF | Calculé : `abs(nc2_mean_cos - (-0.1))` pour 11cls |
| 6 | Dim. MLE (moy.) | `all_scores_consolidated.json` (nouveau) |
| 7 | Dim. MLE (méd.) | `all_scores_consolidated.json` (nouveau) |
| 8 | Rang effectif | `f7bc3e4^:results/all_scores_consolidated.json` (`global_effective_rank`) |
| 9 | Stable rank | `f7bc3e4^:results/all_scores_consolidated.json` (`global_stable_rank`) |
| 10 | Participation ratio | `f7bc3e4^:results/all_scores_consolidated.json` (`global_participation_ratio`) |
| 11 | α spectral | `f7bc3e4^:results/all_scores_consolidated.json` (`global_alpha`) |

**Note :** Les valeurs `global_*` diffèrent des champs `stable_rank`/`participation_ratio`/`alpha_spectral` de `geometry_extended_12models.json` (Δ systématique) — ce sont des calculs différents. Les valeurs de l'ancien consolidated sont certifiées comme étant celles du Tableau 5 (vérifiées via `relance2_correlations_11cls_n12.json` — le ρ=0.825 pour la silhouette correspond).

**11/11 métriques disponibles.** Aucune perte bloquante.

---

## TÂCHE 1 — LE TEST

### Populations

Définies par rang de F1 (`f1_macro_pres_11cls`, schéma without_rhol).

**n12** (12 modèles, F1 ∈ [0.4077, 0.4796]) :
```
vitb16_fulft_arctic      0.4796
dinov3_vitl16_lvd        0.4792
simdinov2_vitl16         0.4760
vitb16_arctic            0.4759
simdinov2_vitb16         0.4723
dinov3_vitb16_lvd        0.4712
resnet50_arctic          0.4620
dinov3_vitl16_sat        0.4617
vitb16_imagenet          0.4500
scalemae_vitl16          0.4483
satmae_vitl16            0.4093
resnet50_imagenet        0.4077
```

**top8** (8 meilleurs, F1 ≥ 0.4617) :
```
vitb16_fulft_arctic      0.4796
dinov3_vitl16_lvd        0.4792
simdinov2_vitl16         0.4760
vitb16_arctic            0.4759
simdinov2_vitb16         0.4723
dinov3_vitb16_lvd        0.4712
resnet50_arctic          0.4620
dinov3_vitl16_sat        0.4617
```
Exclus : vitb16_imagenet (0.4500), scalemae_vitl16 (0.4483), satmae_vitl16 (0.4093), resnet50_imagenet (0.4077).

**top6** (6 meilleurs, F1 ≥ 0.4712) :
```
vitb16_fulft_arctic      0.4796
dinov3_vitl16_lvd        0.4792
simdinov2_vitl16         0.4760
vitb16_arctic            0.4759
simdinov2_vitb16         0.4723
dinov3_vitb16_lvd        0.4712
```
Exclus : resnet50_arctic (0.4620), dinov3_vitl16_sat (0.4617), + les 4 faibles.

**Comparaison seuils rapport3 :** Rapport3 top-8 ≈ F1 ≥ 0.462, top-6 ≈ F1 ≥ 0.471. Nos seuils (0.4617, 0.4712) sont cohérents à 0.0003 près — l'écart vient de l'utilisation du schéma manuscrit (without_rhol) plutôt que du schéma rapport3 (with_rhol, f1_macro_pres).

### Résultat principal — SILHOUETTE

| Population | n | ρ | p | τ | IC95 | p_bonf |
|-----------|---|---|-----|---|------|--------|
| n12 | 12 | **0.8601** | 0.0003 | 0.7576 | [0.469, 1.000] | 0.0036 |
| top8 | 8 | **0.5238** | 0.1827 | 0.4286 | [−0.324, 1.000] | 1.0000 |
| top6 | 6 | **0.5429** | 0.2657 | 0.4667 | [−0.500, 1.000] | 1.0000 |

Δ vs n12 : top8 = **−0.3363**, top6 = **−0.3173**.

**La corrélation s'effondre.** Restreinte au palier compétitif, la silhouette n'est plus significative (p > 0.18) et ses IC95 englobent zéro.

### Toutes les métriques sur top-8

| Métrique | ρ (n12) | ρ (top8) | p (top8) | Δ vs n12 |
|----------|---------|----------|----------|----------|
| Silhouette | +0.8601 | +0.5238 | 0.1827 | −0.3363 |
| Davies-Bouldin | −0.3916 | −0.2857 | 0.4927 | +0.1059 |
| Calinski-Harabasz | +0.2378 | +0.1429 | 0.7358 | −0.0949 |
| NC1 | +0.0909 | +0.0714 | 0.8665 | −0.0195 |
| NC2 dév. ETF | −0.5594 | −0.1667 | 0.6932 | +0.3928 |
| Dim. MLE (moy.) | −0.1748 | −0.5000 | 0.2070 | −0.3252 |
| Dim. MLE (méd.) | −0.1189 | −0.4048 | 0.3199 | −0.2859 |
| Rang effectif | +0.0000 | −0.0714 | 0.8665 | −0.0714 |
| Stable rank | +0.1608 | +0.0714 | 0.8665 | −0.0894 |
| Participation ratio | +0.0839 | −0.0238 | 0.9554 | −0.1077 |
| α spectral | −0.0979 | −0.0952 | 0.8225 | +0.0027 |

**AUCUNE métrique n'est significative sur le palier compétitif.** Tous les p > 0.18. Toutes les corrélations s'effondrent vers ≈ 0 et plusieurs changent de signe.

### Leave-one-out — SILHOUETTE

| Population | ρ_full | LOO min | LOO max | Δ LOO | Modèle influent |
|-----------|---|---|------|------|----------|
| n12 | 0.8601 | 0.8182 | 0.9273 | 0.1091 | dinov3_vitb16_lvd |
| top8 | 0.5238 | 0.2857 | 0.7143 | 0.4286 | simdinov2_vitb16 |
| top6 | 0.5429 | 0.2000 | 0.9000 | 0.7000 | vitb16_fulft_arctic |

**Le manuscrit affirme que la silhouette est « stable en leave-one-out ».** C'est vrai pour n=12 (Δ = 0.11). C'est **faux** pour top8 (Δ = 0.43, un seul modèle fait passer ρ de 0.29 à 0.71) et top6 (Δ = 0.70, le plafond passe de 0.20 à 0.90 selon qu'on retire ou non le meilleur modèle).

### Comparaison avec les 6 métriques de rapport3 (repère uniquement)

| Métrique | ρ n12 (rapport3) | ρ top8 (rapport3) | ρ top6 (rapport3) |
|----------|-----------------|-------------------|-------------------|
| LogME (train) | +0.78 | +0.286 | +0.429 |
| NESum | +0.55 | −0.048 | −0.314 |
| α-ReQ | −0.65 | +0.095 | −0.029 |
| RankMe normalisé | +0.54 | +0.024 | −0.486 |
| RankMe brut | +0.32 | −0.143 | +0.086 |
| Anisotropie | −0.60 | −0.024 | +0.143 |
| **Silhouette (cette étude)** | **+0.86** | **+0.524** | **+0.543** |

**La silhouette suit exactement le même patron que les 6 autres : effondrement sur le palier compétitif.** Elle conserve le ρ le plus élevé en top8 (0.524 vs 0.286 pour LogME), mais l'écart n'est pas interprétable à n=8 (IC95 gigantesques). Aucune métrique ne départage le peloton de tête.

---

## TÂCHE 2 — COLINÉARITÉ INTER-MÉTRIQUES

### Matrice de corrélation (Spearman, n=12, 11cls)

Paires avec |ρ| > 0.9 :

| Paire | ρ |
|-------|-----|
| Dim. MLE (moy.) vs Dim. MLE (méd.) | **+0.979** |
| Rang effectif vs Stable rank | **+0.958** |
| Rang effectif vs Participation ratio | **+0.965** |
| Stable rank vs Participation ratio | **+0.986** |

### Verdict sur Bonferroni ×22

La correction ×22 traite les 11 métriques comme indépendantes. Or :
- Dim. MLE moyenne et médiane sont une seule métrique mesurée deux fois (ρ = 0.979).
- Rang effectif, stable rank et participation ratio sont trois fonctions du même spectre de valeurs propres (ρ ≥ 0.958 entre elles) — une seule famille spectrale.

**Nombre effectif de familles indépendantes ≈ 5–6, pas 11.** La correction ×22 est conservatrice pour les familles non colinéaires mais **sur-pénalise injustement** les métriques de la famille spectrale (elles auraient dû être groupées). L'avertissement du rapport3 (§1.3) — « Les voir survivre ensemble à la correction BH revient à compter une seule preuve trois fois » — s'applique intégralement au Tableau 5 du manuscrit.

---

## TÂCHE 3 — EXTRACTION DES RAPPORTS PDF

### 3.1 — rapport3, Tableau 8 (valeurs top-8/top-6)

Source : `docs/report/sections/transfer.tex`, Tableau~\\ref{tab:correl_intra} (L117–138).

Valeurs du rapport3 confirmées contre le bloc de référence encodé dans la mission : **toutes concordent** (à l'arrondi près). Le tableau est reproduit dans la section Tâche 1 ci-dessus.

### 3.2 — rapport3, Tableau 4 (pureté k-NN par classe)

Source : `docs/report/sections/latent.tex`, Tableau~\\ref{tab:knn_purity_target} (L132–161).

Cette analyse **n'est pas dans le manuscrit**. Elle documente la pureté k-NN (k=20, cosine) des 3 classes problématiques (ARCA, DRYI, RUBC) sur les 12 modèles. Pureté max = 0.36, ce qui confirme que ces classes ne sont pas séparables dans l'espace latent quel que soit le modèle — gelé comme fine-tuné.

Fichier source : `figures/pca_class_diag/knn_purity_summary.csv` — existe dans l'archive `_anciennes_experiences/figures_2026-06_racine/figures/pca_class_diag/knn_purity_summary.csv` (145 lignes, 12 modèles × 12 classes). **Existe, mais uniquement dans l'archive, pas dans `results/`.**

### 3.3 — rapport_exploration, Tableau 5 (tuiles par classe et split)

Source : `docs/rapport_exploration.tex`, Tableau~\\ref{tab:tiles_split} (L197–221).

**Arithmétique confirmée :**
- Train total = 49 433
- RHOL train = 152
- **49 433 − 152 = 49 281** ← c'est le chiffre « 49 281 » du manuscrit (§3.4, §5.4)
- Total général = 80 240 (pas ~70 000)

**Le manuscrit §5.4 écrit « 49 281 tuiles d'entraînement au maximum, soit environ 69.5 % des ~70 000 tuiles totales ».**
- 49 281 / 80 240 = 61.4 % (pas 69.5 %)
- 49 433 / 80 240 = 61.6 % (cohérent avec §3.2)
- Le « ~70 000 » est une erreur : le total réel est 80 240.
- 49 281 / 70 000 = 70.4 % (pas 69.5 %)

**→ BLOQUANT.** Le manuscrit doit corriger « ~70 000 » en « 80 240 » et « 69.5 % » en « 61.4 % », OU expliciter que 49 281 = train sans RHOL (11cls) et 49 433 = train avec RHOL (12cls), et que le total varie selon le schéma.

### 3.4 — rapport3, Tableau 1 (métriques fine-tunés)

Source : `docs/report/sections/latent.tex`, Tableau~\\ref{tab:latent} (L6–32).

Métriques des 3 modèles fine-tunés (RankMe, RankMe/dim, anisotropie, stable rank, kNN purity) :

| Modèle | RankMe | RankMe/dim | Anisotropie | Stable Rank | kNN Pur. |
|--------|--------|-----------|-------------|-------------|----------|
| ResNet-50 FT | 1015.5 | 0.496 | 0.299 | 4.17 | 0.668 |
| ViT-B/16 MHSA FT | 346.8 | 0.452 | 0.394 | 3.78 | 0.681 |
| ViT-B/16 Full FT | 348.3 | 0.454 | 0.498 | 3.86 | 0.679 |

**Ces valeurs existent-elles dans un JSON ?** Les embeddings fine-tunés ont été extraits localement d'après la légende (« embeddings extraits localement »). Aucun JSON dans `results/` ne contient ces 3 modèles avec ces métriques. Le `geometry_extended_12models.json` inclut les 12 modèles mais avec des métriques différentes (`rankme`, `stable_rank`, `alpha_spectral`, etc. — pas `knn_purity`, et les valeurs de `stable_rank` ne correspondent pas). 

**→ BLOQUANT de reproductibilité.** Les métriques géométriques des 3 modèles fine-tunés n'existent que dans le PDF du rapport3. Le JSON source (`geometry_extended_12models.json`) a des champs différents (pas de knn_purity) et des valeurs de stable_rank différentes.

### 3.5 — rapport_exploration : point de départ des modèles affinés

Source : `docs/rapport_exploration.tex`, L262–263.

Citation exacte : « Les trois modèles affinés partent du **ViT-B ImageNet ou de ResNet-50** ; la variante "MHSA FT" ne dégèle que les couches d'auto-attention. »

**Confirmé.** Les modèles affinés partent de ViT-B/16 ImageNet et ResNet-50 ImageNet, **pas de DINOv3**. Le fine-tuning est un fine-tuning supervisé classique sur Arctic-TVC, pas un pré-entraînement continu SSL.

---

## BLOCAGES

1. **Métriques fine-tunés (T3.4) :** Les valeurs géométriques des 3 modèles fine-tunés n'existent que dans le PDF du rapport3. Pas de JSON reproductible. Fichier à créer : `results/ft_latent_metrics.json` à partir des embeddings fine-tunés (sur Colab Drive).

2. **Total tuiles (T3.3) :** Le manuscrit écrit « ~70 000 » alors que le total réel est 80 240. Les pourcentages §5.4 sont faux.

3. **knn_purity_summary.csv (T3.2) :** Existe dans l'archive mais pas dans `results/`. Devrait être promu dans `results/` pour reproductibilité.

---

## ESCALADE (décisions scientifiques)

1. **La contribution « sélection de modèle sans affinage » survit-elle ?** Les chiffres disent : sur le palier compétitif (top-8, top-6), la silhouette n'est pas significative (p = 0.18–0.27), son IC95 englobe zéro, et le LOO est instable (Δ = 0.43–0.70). Le protocole de triage fonctionne pour écarter les 4 modèles faibles, pas pour départager les 8 compétitifs. → **Décision auteur.**
2. **Correction Bonferroni ×22 :** Faut-il la réduire (grouper les familles colinéaires) ou la garder conservatrice ?
3. **Publication des métriques fine-tunés :** Régénérer et commiter le JSON manquant ?

---

## INCERTAIN

- Le ρ=0.8601 (cette étude) diffère du ρ=0.825 du manuscrit. L'écart vient de F1 légèrement différentes (probe_knn_cgrid vs relance2_correlations). Les deux proviennent de sources canoniques différentes.
- Les IC95 bootstrap à n=6 ou n=8 sont gigantesques (borne supérieure = 1.000 pour presque toutes les métriques) — c'est inhérent à la petite taille d'échantillon, pas une erreur de calcul.
- `nc2_deviation_etf` a été recalculé comme `abs(nc2_mean_cos - (-0.1))` pour 11cls. La valeur originale dans l'ancien consolidated était calculée sur 12cls.

---

RHO SILHOUETTE — n12: 0.8601 | top8: 0.5238 | top6: 0.5429 — METRIQUES DISPONIBLES: 11/11 — BLOQUÉ: 3
