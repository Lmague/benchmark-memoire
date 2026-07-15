# AUDIT DE COHÉRENCE — Corrections manuscrit + audit complet

Date : 2026-07-15
Cible : `paper_arctic_fm_benchmark.tex` (soumission IEEE TGRS)
Fichiers canoniques : `results/all_scores_consolidated.json`, `results/T2c_headline_pair_p_CANONICAL.json`, `results/T2a_kmeans_control.json`, `results/without_rhol/probe_knn_cgrid.json`, `results/with_rhol/probe_knn_cgrid.json`

---

## ÉTAT GIT

```
HEAD     : a9c6c55 archive: pre-reconciliation all_scores_consolidated.json snapshot
origin   : bc220ef (HEAD est en avance de 7 commits)
Fichiers non commités : 30 (git status --short | wc -l)
```

```
a9c6c55 archive: pre-reconciliation all_scores_consolidated.json snapshot
f7bc3e4 Kill RECONCILIATION.md manual step
b0d7d3c Tâche B: restore probe_knn_cgrid.json
8da1745 T2c: commit canonical headline pair (11cls, paired bootstrap n=10000)
7d50ea4 T1.d: commit canonical results files for manuscript reproducibility
```

---

## VOLET 1 — CORRECTIONS

### T1 — Revendication « sans aucune étiquette de tâche »

**Preuve.** Quatre scripts calculent `silhouette_score()` avec les labels de vérité terrain :
- `scripts/pipeline_final_2026_07.py` L139 : `silhouette_score(emb_scaled, labels_s)` — `labels_s` = labels du test set
- `scripts/relance2_linear_probe.py` L234 : `silhouette_score(X, Lr)` — `Lr` = `np.asarray(L).ravel()`, L = labels
- `scripts/final_recompute_all.py` L313 : `silhouette_score(X, Lr, random_state=SEED)` — idem
- `scripts/literature_experiments.py` L284 : `silhouette_score(X, L)` — `L` = labels

**T2a_kmeans_control.json** : silhouette calculée sur des clusters KMeans (3 seeds : 0, 7, 42).
- 11cls : ρ ∈ [0.154, 0.301], p ∈ [0.342, 0.633] — TOUS non significatifs
- 8cls : ρ ∈ [0.280, 0.378], p ∈ [0.226, 0.379] — TOUS non significatifs
- Meilleur ρ = 0.378 (seed 7, 8cls), p = 0.226 — loin de ρ = 0.825

**Conclusion : Issue B est factuellement établie.** Le score de silhouette rapporté est calculé sur la vérité terrain (labels), pas sur des clusters non supervisés. Le contrôle KMeans montre que sans labels, la corrélation disparaît.

**Issue A (patch préparé, NON appliqué)** — si le contrôle KMeans avait tenu :
> La formulation aurait été : « le score de silhouette calculé sur des pseudo-classes KMeans… » avec distinction explicite entre ρ=0.825 (labels) et ρ=0.378 (KMeans). Aucune justification factuelle pour l'appliquer — le KMeans ne corrèle pas significativement.

**Issue B (APPLIQUÉ)** — reformulation en « sans affinage ni évaluation supervisée ».

5 emplacements corrigés :
1. Résumé (L52) : « sans affinage ni évaluation supervisée » (remplace « sans aucune étiquette de tâche »)
2. Titre §4.5 (L279) : « sans affinage ni évaluation supervisée » (remplace « sans étiquettes de tâche »)
3. Corps §4.5 (L282) : « sans affinage ni évaluation supervisée » (remplace « sans aucune étiquette de tâche aval »)
4. Légende Figure 3 (L287) : « sans affinage ni évaluation supervisée » (remplace « sans étiquette de tâche »)
5. Conclusion (L369) : « sans affinage ni évaluation supervisée » (remplace « sans aucune étiquette de tâche »)

---

### T2 — §4.6 contradiction layerwise

**Source réelle des valeurs +0.825 et −0.881 :**
- Fichier : `results/layerwise_probe.json` (commit `7d50ea4`, absent du working tree, extrait via `git show 7d50ea4:results/layerwise_probe.json`)
- Modèle : ViT-B/16 ImageNet, 12 couches, F1 par couche (linear probe)
- Spearman(RankMe normalisé, F1) = +0.825175, p = 0.000951
- Spearman(anisotropie, F1) = −0.881119, p = 0.000153
- **Les valeurs sont vérifiées et exactes.** Ce ne sont PAS des fabrications.

**RankMe normalisé vs Rang effectif — deux métriques distinctes :**
- `RankMe normalisé` = `rankme(Z) / dim` (défini dans `src/latent.py` L29-32)
- `Rang effectif` (Tableau 5) = `global_effective_rank`, métrique spectrale différente
- Code source : `rankme()` utilise exp(-entropie des valeurs singulières normalisées) ; `effective_rank` est une mesure différente (aussi dans `src/latent.py`)

**Correction appliquée.** Le paragraphe §4.6 affirme une « cohérence » entre intra-modèle et inter-modèles qui est contredite par le Tableau 5 (Rang effectif inter-modèles : ρ = −0.028, p = 0.931). La reformulation :
- Remplace « sont cohérentes avec les corrélations inter-modèles » par « **divergent** des corrélations inter-modèles »
- Documente les deux causes de cette divergence : (i) confondant de dimension (768 vs 1024 vs 2048), (ii) unité d'analyse différente (couche vs modèle)
- **Renforce** la thèse de découplage au lieu de l'affaiblir

**Signalement 1 (non corrigé) :** Le F1 layerwise repose sur ViT-B/16 ImageNet, avant-dernier du Tableau 3 (F1=0.4500). Ancrer une généralisation sur le modèle le plus faible du corpus est un choix scientifique à assumer explicitement.

**Signalement 2 (non corrigé) :** La valeur 0.825 apparaît deux fois :
- Inter-modèles : silhouette ρ = 0.825 (n = 12 modèles, relance2_correlations_11cls_n12.json)
- Intra-modèle : RankMe normalisé ρ = 0.825 (n = 12 couches, layerwise_probe.json)
- Vérification mathématique : pour n = 12, ρ = 1 − 6Σd²/(12×143). Pour ρ = 0.825175, Σd² = 50. C'est une coïncidence mathématiquement possible (Σd² = 50 est un entier plausible). Ce n'est pas forcément un copier-coller, mais les deux calculs doivent être sourcés indépendamment.

---

### T3 — Le p ≈ 0.52 (proxy RidgeClassifier) dans le résumé

**Source canonique :** `results/T2c_headline_pair_p_CANONICAL.json`
- p_two_sided = 0.8972 (LogisticRegression lbfgs, 11cls, n=10000, bootstrap apparié)
- p_frozen_gt_finetuned = 0.4486 (unilatéral)

**Correction appliquée au résumé :**
- « p ≈ 0.52 » → « p = 0.897 (test bilatéral, bootstrap apparié n = 10 000) »
- IC95 inchangés (proviennent d'une autre source, voir T4)

**Convention de signe de Δ :**
- T2c : `delta_observed = −0.000409` (frozen − finetuned)
- Résumé et §4.1 : `Δ = +0.0004` (finetuned − frozen)
- Cohérent si on inverse le signe et arrondit. Convention uniforme dans tout le manuscrit : finetuned − frozen. **OK, pas de correction nécessaire.**

**Note de cohérence §4.1 — RÉÉCRITE :**
Avant : comparaison bancale entre p=0.483 (12cls) et p≈0.52 (proxy Ridge).
Après : « La valeur canonique du delta 11cls (Δ = +0.0004, p = 0.897, test bilatéral apparié sur prédictions LogisticRegression lbfgs, n = 10 000) confirme l'indistinguabilité déjà documentée au schéma 12 classes (p_a_gt_b = 0.483, test unilatéral apparié, LogisticRegression lbfgs, n = 1 000). Les deux p unilatéraux — 0.449 (11cls) et 0.483 (12cls) — sont méthodologiquement cohérents (même classifieur, même type de test, schémas différents). »

**Vérification p=0.483 :** `results/significance_matrix_all12.json` — apparié, LogisticRegression lbfgs, 12cls, n=1000. `p_a_gt_b = 0.483` (unilatéral). **Comparable à T2c** (même classifieur, même type de test apparié).

**Occurrences du proxy Ridge supprimées :** « Note de cohérence » réécrite (voir ci-dessus). Aucune autre occurrence trouvée.

---

### T4 — Les IC du Tableau 3 ne viennent pas du même calcul que les F1

**Recherche exhaustive :**
- `results/bootstrap_ci_2026-07-09.json` : 12cls (with_rhol), n=1000, 96 modèles — PAS 11cls
- `results/significance_matrix_all12.json` : 12cls, dérivé du précédent, 12 modèles canoniques
- `results/T2c_headline_pair_p_CANONICAL.json` : 11cls, n=10000, 2 modèles SEULEMENT
- Aucun bootstrap 11cls couvrant les 12 modèles canoniques n'existe dans le dépôt ni dans l'historique git

**→ BLOQUÉ — T4.** Les IC95 du Tableau 3 (lignes ViT-B/16 FT et DINOv3 ViT-L/16) sont des valeurs 12cls accolées à des F1 11cls. Corriger 2 lignes sur 12 créerait une asymétrie pire (2 IC n=10000 11cls + 10 IC n=1000 12cls).

**Ce qu'il faudrait :**
- Un bootstrap apparié 11cls sur les 12 modèles canoniques (LogisticRegression lbfgs, n = 1 000 ou 10 000)
- Fichier à produire : `results/bootstrap_ci_11cls_12models.json`
- Machine : Narval (embeddings fine-tunés et frozen 11cls complets), ou local si tous les embeddings 11cls sont disponibles

**Divergence de légende :** Le Tableau 3 dit « n = 1 000 », le fichier `significance_matrix_all12.json` confirme n_bootstrap = 1 000. La légende T2c dit n = 10 000. Cohérent avec les sources respectives. **Pas de correction — la légende du Tableau 3 n'est pas fausse pour ce qu'elle décrit (même si le schéma n'est pas le bon).**

**Tableau 4 (8cls) :** Les IC95 du Tableau 4 n'ont pas pu être vérifiés (pas de fichier bootstrap 8cls canonique dans le working tree). `results/8cls/bootstrap_ci.json` était dans le commit `7d50ea4` mais n'existe plus à HEAD. Le fichier `results/bootstrap_ci_2026-07-09.json` contient 96 modèles mais en schéma 12cls uniquement. → **SIGNALÉ, non corrigé.**

---

## VOLET 2 — AUDIT DE COHÉRENCE

### a. Cohérence des chiffres entre sections

**Tableau comparatif F1 11cls — Tableau 3 vs sources canoniques :**

| # | Modèle | Tableau 3 | `all_scores_consolidated.json` (probe_knn_cgrid, 11cls) | `relance2_correlations_11cls_n12.json` | Écart Tableau vs consolidated |
|---|--------|-----------|--------------------------------------------------------|--------------------------------------|------|
| 1 | ViT-B/16 FT (Arctic, full) | 0.4796 | 0.4796 | 0.4796 | 0 ✓ |
| 2 | DINOv3 ViT-L/16 (LVD) | 0.4792 | 0.4792 | 0.4792 | 0 ✓ |
| 3 | SimDINOv2 ViT-L/16 | 0.4760 | 0.4760 | 0.4760 | 0 ✓ |
| 4 | ViT-B/16 FT (Arctic, MHSA) | 0.4759 | 0.4759 | 0.4759 | 0 ✓ |
| 5 | SimDINOv2 ViT-B/16 | 0.4723 | 0.4723 | 0.4723 | 0 ✓ |
| 6 | DINOv3 ViT-B/16 (LVD) | 0.4712 | 0.4712 | 0.4712 | 0 ✓ |
| 7 | DINOv3 ViT-L/16 (SAT-493M) | 0.4620 | 0.4617 | 0.4620 | **0.0003** ⚠ |
| 8 | ResNet-50 FT (Arctic, full) | 0.4619 | 0.4620 | 0.4619 | **−0.0001** ⚠ |
| 9 | ViT-B/16 ImageNet | 0.4500 | 0.4500 | 0.4500 | 0 ✓ |
| 10 | Scale-MAE ViT-L/16 | 0.4480 | 0.4483 | 0.4480 | **−0.0003** ⚠ |
| 11 | SatMAE ViT-L/16 | 0.4091 | 0.4093 | 0.4091 | **−0.0002** ⚠ |
| 12 | ResNet-50 ImageNet | 0.4076 | 0.4077 | 0.4076 | **−0.0001** ⚠ |

**Analyse :** Le Tableau 3 utilise les valeurs de `relance2_correlations_11cls_n12.json` (précision flottante non arrondie), tandis que `all_scores_consolidated.json` pointe vers `probe_knn_cgrid.json`. Les deux sources diffèrent de 0.0001–0.0003 pour 5 modèles. L'écart est minuscule (< 1‰) mais révèle une **duplication de sources** : deux fichiers JSON contiennent des F1 11cls légèrement différents pour les mêmes modèles. La source `probe_knn_cgrid` est antérieure (best_C par `f1_macro_all`), la source `relance2_correlations` est plus récente (best_C par `f1_macro_pres` ou run séparé).

**Sévérité : MAJEUR** — les valeurs diffèrent mais dans la limite de l'arrondi à 4 décimales. Cependant la **provenance est ambiguë** : `AGENTS.md` et `all_scores_consolidated.json` désignent `probe_knn_cgrid`, mais le Tableau 3 suit `relance2_correlations`.

**IC95 divergentes §4.2 :** Le texte cite `[0.4728, 0.4847]` pour ViT-B/16 FT. Le Tableau 3 donne `[0.4727, 0.4844]`. Ces deux IC viennent de deux sources bootstrap différentes :
- `significance_matrix_all12.json` : 12cls, n=1000 → `[0.4728, 0.4847]` pour `vitb16_fulft_arctic`
- Tableau 3 : source incertaine mais probablement la même avec arrondi différent → `[0.4727, 0.4844]`
**Sévérité : MAJEUR** — deux IC différentes pour le même modèle, même schéma, dans le même article.

**Δ = +0.0084 et p = 0.003 (§4.2) :** Source non trouvée comme fichier JSON canonique. Le calcul est plausible (0.4796 − 0.4712 = 0.0084, p = 0.003 suggère une différence significative entre ViT-B/16 FT et ViT-B/16 frozen) mais **non vérifiable contre un fichier JSON** → **MAJEUR**.

---

### b. Comptages et sommes

**§3.3 — « trois génuinistes » → 4 modèles :**
« trois génuinistes (ResNet-50 et ViT-B/16 ImageNet, DINOv3 ViT-B/16 et ViT-L/16 pré-entraînés LVD) »
Ceci liste **4** modèles : ResNet-50, ViT-B/16 ImageNet, DINOv3 ViT-B/16, DINOv3 ViT-L/16.
→ **CORRIGÉ :** « quatre génuinistes »

**§3.3 — « deux géospatiaux satellite » → 3 modèles :**
« deux géospatiaux satellite (SatMAE ViT-L/16, Scale-MAE ViT-L/16, DINOv3 ViT-L/16 pré-entraîné SAT-493M) »
Ceci liste **3** modèles.
→ **CORRIGÉ :** « trois géospatiaux satellite »

**§3.3 — « deux bio-spécialisés » → 2 modèles :**
SimDINOv2 ViT-B/16 et ViT-L/16 → **2 ✓** (comptage correct)

**9 figés + 3 affinés = 12 ✓** après correction (4 + 3 + 2 = 9 figés, + 3 affinés = 12).

**§3.2 vs §3.4 vs §5.4 — divergence train :**
- §3.2 L164 : « 49 433 tuiles, 61.6 % »
- §3.4 L182 : « de 492 à 49 281 tuiles »
- §5.4 L357 : « 49 281 tuiles d'entraînement au maximum, soit environ 69.5 % des ~70 000 tuiles totales »

Vérification : 1 % de 49 433 = 494 (pas 492). 100 % = 49 433 (pas 49 281).
L'hypothèse la plus probable : 49 281 = train **après exclusion complète de RHOL** (11cls), et 49 433 = train 12cls. Différence = 152 tuiles RHOL.

→ **BLOQUANT** — trois chiffres incompatibles dans trois sections différentes. À corriger en explicitant la différence 12cls vs 11cls.

**Tableau 1 (7998 segments) vs Tableau 2 (7866 segments) :**
Somme des segments par classe dans le Tableau 2 : 455 + 250 + 858 + 25 + 876 + 1240 + 1327 + 28 + 183 + 991 + 1144 + 489 = **7866**.
Le Tableau 1 annonce **7998 segments**.
Écart = 132 segments. Les « segments » du Tableau 1 peuvent provenir d'une subdivision des polylignes qui ne correspond pas au comptage par classe du Tableau 2.

→ **BLOQUANT** — 7998 ≠ 7866. La définition de « segment » doit être explicitée.

---

### c. Régimes d'affinage — incohérence structurelle

**Résumé §3.4 :** « quatre régimes d'affinage (complet, tête d'attention seule, ExPLoRA/LoRA, entraînement depuis zéro) »
**Figure 2 :** Montre ExPLoRA/LoRA.
**Tableau 3 :** Aucune ligne ExPLoRA/LoRA. Seulement FT full (×2) et FT MHSA (×1).
**§3.3 :** Ne liste que 3 modèles affinés (ResNet-50 full, ViT-B/16 full, ViT-B/16 MHSA).

ExPLoRA/LoRA fait partie de la courbe de données (Figure 2) mais n'est PAS dans le benchmark principal (Tableau 3). Le résumé le présente comme un régime évalué du benchmark.

→ **SIGNALÉ — décision de périmètre.** L'utilisateur doit clarifier : ExPLoRA/LoRA est-il un régime du benchmark (→ ajouter au Tableau 3) ou seulement de la courbe de données (→ corriger le résumé et §3.4) ?

---

### d. Résumé vs corps

**« DINOv2 » dans le résumé :** Le résumé dit « génuinistes (DINOv2/v3, ResNet, ViT ImageNet) ». **DINOv2 n'apparaît nulle part dans le corpus.** Seuls DINOv3, SimDINOv2, et ViT ImageNet y sont. SimDINOv2 est un modèle **bio-spécialisé**, pas un « génuiniste » (il est pré-entraîné sur des images de plantes).

→ **BLOQUANT** — mention d'un modèle absent du corpus.

**p < 10⁻³ (résumé) vs p = 0.0010 (Tableau 5) vs p < 0.001 (Figure 3) vs p = 0.001 (§4.5) :**
- Tableau 5 : p = 0.0010 (source canonique `relance2_correlations_11cls_n12.json` : p = 0.000951)
- Résumé : p < 10⁻³
- Figure 3 : p < 0.001
- §4.5 : p = 0.001

La valeur canonique est p = 0.000951. **0.000951 < 0.001**, donc « p < 0.001 » est mathématiquement correct. Mais « p = 0.0010 » est un arrondi trompeur (0.000951 arrondi à 0.0010 = correct à 4 décimales, mais 0.0010 n'est pas < 0.001).

→ **CORRIGÉ :** Harmonisation de toutes les occurrences sur « p < 0.001 » (cohérent avec la valeur canonique p = 0.000951).

---

### e. Figures

**Figure 1 — valeurs 12cls sur graphique 11cls :**
Le texte interne de la figure dit « Δ = −0.0002, p = 0.483 (indistinguables) ». Ces valeurs (Δ = −0.000192, p = 0.483) proviennent de `significance_matrix_all12.json` → **schéma 12cls**. Mais la figure est titrée/référencée comme « schéma 11 classes » (§4.1, L201 : `fig1_ranking_f1_11cls.png`). **La figure porte des valeurs 12cls sur un graphique 11cls.**

→ **BLOQUANT** — la figure doit être régénérée avec les valeurs 11cls. Script à relancer : `make_figures.py` avec la source de données 11cls. **Non corrigé (hors scope, régénération figure).**

---

### f. Tableau 5 et corrections multiples

**Légende vs contenu :** La légende dit « seules les deux lignes présentant des p_bonf < 1 sont détaillées » mais la table affiche les 11 lignes. → **CORRIGÉ :** « Les 11 métriques sont rapportées ; seules les deux premières (Silhouette, Davies-Bouldin) présentent des p_bonf < 1. »

**Bonferroni ×22 :** 11 métriques × 2 schémas = 22 ✓. Vérification de 3 p_bonf :
- Silhouette 11cls : p=0.0010 × 22 = 0.022 → plafonné à **0.0209** (le code utilise min(p×N, 1.0), la table donne 0.0209). **Suspicion :** 0.0010×22 = 0.022, pas 0.0209. Si p=0.000951, alors 0.000951×22 = 0.02092 → 0.0209 ✓. **L'arrondi est correct si on utilise la p brute réelle.**
- Davies-Bouldin 11cls : p=0.2652 × 22 = 5.834 → plafonné à **1.0000** ✓
- Rang effectif 11cls : p=0.9312 × 22 = 20.49 → plafonné à **1.0000** ✓

**p_BH — vérification :** Benjamini-Hochberg sur 22 p-valeurs. La plus petite (silhouette 11cls, p=0.000951) : rang 1, seuil BH = 1×0.05/22 = 0.00227. p=0.000951 < 0.00227 → significatif. La p_BH affichée = 0.0209 (22×p). **Le format d'affichage est p×22, pas le seuil BH.** → **MAJEUR** — la colonne « p_BH » n'affiche pas la p-value corrigée BH mais p×22 (identique à p_bonf pour les petites valeurs). C'est incorrect. La procédure BH donne des seuils ajustés, pas p×N.

**† (leave-one-out) :** Les marqueurs † sont sourcés dans `relance2_correlations_11cls_n12.json` (champ `outlier_warning.is_carried_by_single_outlier`). Vérification :
- Davies-Bouldin 11cls : `is_carried_by_single_outlier: true`, `worst_outlier_model: resnet50_imagenet` → † correct
- Rang effectif 11cls : `is_carried_by_single_outlier: true` → † correct
- Stable rank 11cls : `is_carried_by_single_outlier: true` → † correct
- Participation ratio 11cls : `is_carried_by_single_outlier: true` → † correct
- α spectral 11cls : `is_carried_by_single_outlier: true` → † correct
→ **Sourcés ✓**

---

### g. §5.4 — Bugs connus

**« un scheduler de régularisation figé à C = 0.01 au lieu du meilleur C canonique » :** Cette phrase fusionne deux bugs distincts :
1. Bug du scheduler cosine annealing steppé par epoch au lieu de par batch (commit `7682c50`)
2. Asymétrie de grille C entre pipelines affinés et figés (documentée dans `known_issues.md` §4.3)

Un « scheduler de régularisation » n'existe pas — le scheduler pilote le learning rate, pas C. La régularisation C est un hyperparamètre du probe linéaire, pas du scheduler.

→ **SIGNALÉ — reformulation proposée (non appliquée, décision auteur) :**
« (i) un scheduler de learning rate cosine annealing incorrectement steppé par epoch au lieu de par batch dans certaines expériences d'affinage (commit 7682c50), corrigé avant la production des résultats présentés ici ; (ii) une asymétrie de grille de régularisation C entre les pipelines affinés et figés, harmonisée dans le benchmark canonique par une grille unique C ∈ {10⁻⁴, …, 10}. »

**« corrigé avant la production des résultats présentés ici » :**
- Bug scheduler (7682c50) : corrigé ✓ (le commit est antérieur aux résultats canoniques)
- Asymétrie de grille C : `known_issues.md` §4.3 la documente comme **encore ouverte** (« Une comparaison fine-tuné vs frozen n'est valide que si les deux probes utilisent la même grille »). → **BLOQUANT** — le manuscrit affirme que l'asymétrie est corrigée, mais `known_issues.md` suggère qu'elle persiste comme précaution méthodologique. L'asymétrie est-elle corrigée (grille unique) ou non ? Si elle l'est, `known_issues.md` doit être mis à jour. Si elle ne l'est pas, le manuscrit ment.

---

### h. Travaux connexes vs résultats

**LogME [18] :** Introduit en §2.5 comme méthode d'estimation de transférabilité. Aucun résultat LogME n'apparaît dans le corps de l'article. Le code LogME existe (`src/transfer.py`, `src/_vendor/LogME.py`) et des résultats archivés existent (`_anciennes_experiences/nightly_2026-06-14/data/logme_vs_f1.json`). Soit le résultat manque à l'appel, soit la référence prépare une analyse absente.

→ **SIGNALÉ** — référence orpheline. Ajouter le résultat LogME ou supprimer la référence.

**RankMe [17] vs Rang effectif (Tableau 5) :** RankMe est introduit en §2.5, mais le Tableau 5 utilise « Rang effectif » (`global_effective_rank`) qui est une métrique apparentée mais distincte. Le lecteur ne peut pas faire le lien.

→ **SIGNALÉ** — ajouter une note précisant que « Rang effectif » est une variante de la métrique RankMe [17].

---

### i. Références croisées

**§5.1 L338 :** « La Section 5.2 montre au contraire… » → C'est **§4.2** qui montre la comparaison à architecture égale.
→ **CORRIGÉ :** « Section 5.2 » → « Section 4.2 »

**Vérification systématique des `\ref` :**
- `\ref{sec:sota_methods}` → §3.4 ✓
- `\ref{sec:latent}` → §4.5 ✓
- `\ref{tab:dataset}` → Tableau 1 ✓
- `\ref{tab:classes}` → Tableau 2 ✓
- `\ref{tab:schemeA}` → Tableau 3 ✓
- `\ref{tab:schemeB}` → Tableau 4 ✓
- `\ref{fig:ranking}` → Figure 1 ✓
- `\ref{sec:main_result}` → §4.1 ✓
- `\ref{fig:curve}` → Figure 2 ✓
- `\ref{fig:scatter}` → Figure 3 ✓
- `\ref{tab:corr}` → Tableau 5 ✓
- `\ref{fig:heatmap}` → Figure 4 ✓
- `\ref{sec:discussion}` → §5 ✓
- `\ref{sec:limits}` → §5.4 ✓
- `\ref{sec:sota_methods}` → §3.4 ✓

**§5.2 L338 :** La référence « Section 5.2 » est une référence en clair (pas un `\ref`), pointant vers elle-même. C'est §4.2.
**§5.3 L363 :** « §5.3 » → auto-référence, devrait être « §5.3 » (la section courante). ✓ (correct, c'est une auto-référence délibérée)
**§5.2 L338 :** « §5.2 » → **CORRIGÉ en « §4.2 ».**

---

### j. Divers

**« génuiniste » :** Néologisme. L'usage français standard serait « généraliste ». Employé partout (7 occurrences). → **MINEUR — signalé, non corrigé (décision délibérée possible).**

**Auteur « Étudiant-chercheur∗ » :** Placeholder d'anonymisation pour IEEE TGRS (double aveugle). → **SIGNALÉ — à remplacer par le vrai nom après acceptation.**

**Cohérence des arrondis :** Tous les F1 sont à 4 décimales. OK. Les p-values varient entre 2 et 4 décimales. Cohérent avec les conventions statistiques.

**Références [1]-[18] :** Toutes citées dans le texte. Vérification rapide : [1] DINOv2, [2] DINOv3, [3] SatMAE, [4] Scale-MAE, [5] SimDINOv2/plantes, [6] Kornblith, [7] VTAB, [8] Corley/nobody knows, [9] GEO-Bench, [10] Arctic-TVC, [11] ResNet, [12] ViT, [13] DINO, [14] Sosa/effective, [15] BioCLIP, [16] ExPLoRA, [17] RankMe, [18] LogME. Toutes citées. Aucune référence orpheline.

---

## TABLEAU RÉCAPITULATIF DES ANOMALIES

| # | Ligne .tex | Citation | Nature | Preuve | Sévérité | Statut |
|---|-----------|----------|--------|--------|----------|--------|
| T1 | 52,279,282,287,369 | « sans aucune étiquette de tâche » | Faux — silhouette sur labels terrain | T2a_kmeans_control + code | BLOQUANT | **CORRIGÉ** |
| T2 | 329 | « cohérentes avec inter-modèles » | Faux — Rang effectif ρ=−0.028 | Tableau 5 + layerwise_probe.json | BLOQUANT | **CORRIGÉ** |
| T3 | 52 | « p≈0.52 » | Proxy RidgeClassifier, pas canonique | T2c p_two_sided=0.8972 | BLOQUANT | **CORRIGÉ** |
| T4 | 215-216 | IC95 12cls dans tableau 11cls | Pas de bootstrap 11cls 12-modèles | Aucun fichier trouvé | BLOQUANT | **BLOQUÉ** |
| V2a.7 | 220 | dinov3_vitl16_sat F1=0.4620 | 0.4620 vs 0.4617 consolidé | relance2 vs probe_knn_cgrid | MAJEUR | **SIGNALÉ** |
| V2a.8 | 222 | resnet50_arctic F1=0.4619 | 0.4619 vs 0.4620 consolidé | relance2 vs probe_knn_cgrid | MAJEUR | **SIGNALÉ** |
| V2a.10 | 224 | scalemae F1=0.4480 | 0.4480 vs 0.4483 consolidé | relance2 vs probe_knn_cgrid | MAJEUR | **SIGNALÉ** |
| V2a.12 | 226 | resnet50 F1=0.4076 | 0.4076 vs 0.4077 consolidé | relance2 vs probe_knn_cgrid | MAJEUR | **SIGNALÉ** |
| V2a.IC | 262 | IC95 [0.4728,0.4847] | ≠ Tableau 3 [0.4727,0.4844] | Deux sources bootstrap | MAJEUR | **SIGNALÉ** |
| V2a.Δ | 262 | Δ=+0.0084 p=0.003 | Source non trouvée | Pas de JSON canonique | MAJEUR | **SIGNALÉ** |
| V2b.1 | 175 | « trois génuinistes » | 4 modèles listés | Dénombrement | MAJEUR | **CORRIGÉ** |
| V2b.2 | 175 | « deux géospatiaux » | 3 modèles listés | Dénombrement | MAJEUR | **CORRIGÉ** |
| V2b.3 | 164,182,357 | 49433 vs 49281 vs 69.5% | Incompatibles | Calcul arithmétique | BLOQUANT | **SIGNALÉ** |
| V2b.4 | 124,143-158 | 7998 vs 7866 segments | Incompatibles | Tableau 1 vs 2 | BLOQUANT | **SIGNALÉ** |
| V2c | 52,76,182 | ExPLoRA/LoRA absent Tableau 3 | Périmètre incohérent | Tableau 3 | MAJEUR | **SIGNALÉ** |
| V2d.1 | 52 | « DINOv2 » dans résumé | Modèle absent du corpus | Tableau 3 | BLOQUANT | **SIGNALÉ** |
| V2d.2 | 52,282,301 | p<10⁻³ vs p=0.0010 vs p<0.001 | Incohérence d'arrondi | relance2 p=0.000951 | MAJEUR | **CORRIGÉ** |
| V2e.1 | 201 | Fig 1 : Δ=−0.0002, p=0.483 | 12cls sur figure 11cls | significance_matrix_all12 | BLOQUANT | **SIGNALÉ** |
| V2f.1 | 294 | Légende « 2 lignes », table 11 | Légende ≠ contenu | Tableau 5 | MINEUR | **CORRIGÉ** |
| V2f.2 | 299 | Colonne p_BH = p×22 | Pas la correction BH | Calcul | MAJEUR | **SIGNALÉ** |
| V2g | 351 | « scheduler de régularisation figé à C=0.01 » | Fusionne 2 bugs distincts | known_issues.md + git log | BLOQUANT | **SIGNALÉ** |
| V2g | 351 | « corrigé avant la production » | Faux pour asymétrie C | known_issues.md §4.3 | BLOQUANT | **SIGNALÉ** |
| V2h.1 | 103 | LogME [18] sans résultat | Référence orpheline | Aucun LogME dans résultats actifs | MAJEUR | **SIGNALÉ** |
| V2h.2 | 103,308 | RankMe vs Rang effectif | Lien non explicité | src/latent.py | MINEUR | **SIGNALÉ** |
| V2i | 338 | « Section 5.2 » → §4.2 | Référence croisée fausse | Structure du document | MAJEUR | **CORRIGÉ** |
| V2j.1 | — | « génuiniste » | Néologisme | Usage français | MINEUR | **SIGNALÉ** |
| V2j.2 | 42 | « Étudiant-chercheur∗ » | Placeholder anonymisation | IEEE TGRS double aveugle | MINEUR | **SIGNALÉ** |

---

## BLOCAGES

1. **T4 — Bootstrap 11cls sur 12 modèles inexistant.** Fichier à produire : `results/bootstrap_ci_11cls_12models.json`. Nécessite les embeddings 11cls complets (dont fine-tunés sur Colab Drive).
2. **V2b.3 — 49 433 vs 49 281 vs 69.5%.** Trois chiffres incompatibles pour la taille d'entraînement. À clarifier : 49 433 = 12cls, 49 281 = 11cls ?
3. **V2b.4 — 7998 vs 7866 segments.** Tableau 1 vs Tableau 2 incompatibles.
4. **V2e.1 — Figure 1 avec valeurs 12cls.** Régénération nécessaire.
5. **V2g — « corrigé avant la production » pour asymétrie C.** Contredit `known_issues.md`.

---

## ESCALADE (décisions scientifiques à trancher)

1. **T1** : Accepter la reformulation « sans affinage ni évaluation supervisée » ? (appliquée)
2. **T2** : Accepter la reformulation de §4.6 (divergence, pas cohérence) ? (appliquée)
3. **V2c** : ExPLoRA/LoRA dans le benchmark (Tableau 3) ou seulement courbe de données ?
4. **V2g** : L'asymétrie de grille C est-elle corrigée ou non ? (détermine si le manuscrit ment)
5. **V2a** : Quelle source F1 11cls fait autorité — `probe_knn_cgrid.json` ou `relance2_correlations` ?
6. **V2j** : Garder « génuiniste » ou passer à « généraliste » ?
7. **V2h** : Garder LogME comme référence ou ajouter le résultat manquant ?
8. **V2f** : Corriger la colonne p_BH (afficher vraie correction BH, pas p×22) ?

---

## INCERTAIN

- **Δ = +0.0084, p = 0.003 (§4.2)** : source JSON non trouvée. Probablement calculé ad hoc.
- **IC95 Tableau 4 (8cls)** : source `results/8cls/bootstrap_ci.json` absente du working tree.
- **49 281** : hypothèse train 11cls (49 433 − 152 RHOL), non confirmée par un fichier.
- **7998 vs 7866** : divergence non résolue, probablement due à une différence de comptage polylignes vs segments.
- **p_BH dans Tableau 5** : la colonne étiquetée « p_BH » affiche en réalité p×22 (identique à p_bonf pour les valeurs < 1/22).

---

## LIVRABLES

1. `paper_arctic_fm_benchmark.tex` — patché (corrections T1, T2, T3, corrections mécaniques Volet 2)
2. `docs/audit/corrections_manuscrit_2026-07.md` — ce rapport
