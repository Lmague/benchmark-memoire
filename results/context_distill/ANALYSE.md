# Analyse — Self-distillation contexte→tuile sur Arctic-TVC (ctx 1024)

*Date : 2026-08-31. Sources : `results/context_distill/runs/`, `controls/`, `geometry/` (test v3, split spatial).*

## 1. Question et protocole

Deux designs testent l'apport du **contexte spatial** (fenêtre 1024px centrée sur la tuile, redimensionnée à 224) pour la classification de végétation arctique par drone (DINOv3-B/16 LVD + LoRA r=2 α=4, blocs 6-11, 3 seeds, split spatial, **test = v3 identique à toutes les baselines**) :

- **Design A** (R1 teacher DINOv3-L ; R3 teacher EMA-self) : contexte **au train seulement** (self-distillation), **tuile seule (768)** à l'inférence.
- **Design B** (R2) : même entraînement mais tête de classification sur `[tuile;contexte]` (**1536**) — le contexte est disponible **à l'inférence aussi**.

## 2. Résultats F1 (3 seeds, test v3)

| Design | Teacher | F1_macro ± std | F1_8cls | BAcc |
|---|---|---|---|---|
| **B (fused)** | DINOv3-L | **0.5080 ± 0.0013** | **0.6985** | **0.5077** |
| A | DINOv3-L | 0.4870 ± 0.0011 | 0.6696 | 0.4913 |
| A | EMA-self | 0.4836 ± 0.0014 | 0.6650 | 0.4877 |

**Constats immédiats**
- La **distillation seule (Design A) ne bat pas les baselines LoRA** (0.4870 vs 0.4835-0.4844) : le gain est dans le bruit inter-seed (~0.002-0.008). Le contexte au train n'améliore pas la représentation tuile-seule.
- Le teacher externe n'apporte rien de mesurable par rapport à l'EMA self (+0.003, bruit).
- **Design B explose : +0.021 à +0.024 vs Design A, +0.025 vs la meilleure baseline LoRA (0.4835)** — très au-dessus des seuils d'interprétabilité (0.01, AGENTS.md §4.4) et de l'écart inter-seed (~0.002).

## 3. Attribution du gain (matrice, seed0, même test v3, même sonde)

| Modèle | tile (768) | fused (1536) | Δ contexte |
|---|---|---|---|
| DINOv3-B **gelé** | 0.4716 | 0.4862 | +0.015 |
| LoRA r2 (v3train) | 0.4875 | 0.4946 | +0.007 |
| LoRA r8a16 (spatial) | 0.4887 | 0.4945 | +0.006 |
| R1 dA_tL (distill A) | 0.4856 | 0.4952 | +0.010 |
| **R2 dB_tL (distill B)** | **0.4779** | **0.5098** | **+0.032** |

*R2/fused = moyenne 3 seeds (`metrics.json`) ; les contrôles = seed0. Sanity : frozen/tile 0.4716 ≈ canonique 0.4712 ; LoRA spatial/tile 0.4887 ≈ csv spatial 0.4885.*

**Décomposition du gain de R2 (0.508) par rapport au frozen tile (0.472) :**
1. **Entraînement (tile seul)** : +0.015 (frozen 0.472 → entraînés 0.486-0.489).
2. **Contexte dans la sonde** : +0.006 à +0.010 pour tous les modèles → plafond ~0.495 pour quiconque n'a pas été entraîné en fusion (y compris le **gelé-fusionné = 0.486**).
3. **Entraînement Design B (tête sur features fusionnées)** : +0.015 de plus → **seul R2 franchit 0.51**.

**Point décisif — R2 est le pire en tuile seule (0.4779, < R1 0.4856, < LoRA 0.4875-0.4887)** mais le meilleur en fusion. L'entraînement Design B **réorganise le backbone pour dépendre du contexte** : il sacrifie la représentation tuile-seule et encode l'information conjointe tuile⊕contexte.

## 4. Géométrie de l'espace latent (test v3, subsample 20k, seed 42)

| Modèle | repr | dim | RankMe | RankMe/D | Aniso | α-ReQ | NESum |
|---|---|---|---|---|---|---|---|
| **FROZEN** | tile | 768 | 357.6 | 0.466 | **+0.587** | 1.66 | 5.85 |
| dA_tL (R1) | tile | 768 | 355.6 | 0.463 | +0.334 | 1.72 | 4.73 |
| dA_tEMA (R3) | tile | 768 | 342.7 | 0.446 | +0.349 | 1.73 | 4.40 |
| dB_tL (R2) | **fused** | 1536 | 701.6 | 0.457 | +0.379 | 1.76 | 4.24 |
| dB_tL (R2) | tile_only | 768 | 340.0 | 0.443 | **+0.449** | 1.74 | 4.28 |

- **Le fine-tuning déplie l'espace latent** : anisotropie 0.587 (frozen) → 0.33-0.45 (entraînés), à RankMe/D quasi constant (~0.44-0.47). C'est l'adaptation à la tâche classique.
- **La branche tuile-seule de R2 est plus anisotrope (0.449) que celle de R1 (0.334)** : le backbone B ne « déplie » pas la tuile seule — cohérent avec sa dépendance au contexte. Le **fused rétablit l'isotropie** (0.379 < 0.449).
- **RankMe/D du fused (0.457) ≈ tuile (0.44-0.46)** : la fusion n'ajoute pas de richesse spectrale par dimension — le gain n'est pas un artefact de dimensionnalité.

## 5. Interprétation mécanistique

Le contexte 1024px contient la végétation **environnante** de la tuile. Pour des classes de toundra dont la discrimination repose sur le voisinage (mosaïques de lichen vs mousse vs arbustes, transition DRYI↔MOSS↔WILL), cette information est fortement discriminante — mais elle n'est exploitable que si le modèle a été **entraîné à la fusionner** avec la tuile (Design B). Un simple concat à la sonde sur un backbone non entraîné en fusion (gelé : 0.486 ; LoRA : 0.495 ; même R1 distillé : 0.495) ne suffit pas : le backbone n'a pas appris à aligner les deux vues.

R2 est donc une **borne supérieure théorique** : non déployable en terrain (il exige le contexte 1024px au moment de prédire), mais preuve que l'information spatiale voisine existe et vaut ~+0.03 de F1 si on l'exploite correctement.

## 6. Limites et caveats (à mentionner dans le manuscrit)

1. **best_C asymétrique** : R2/fused sélectionne C=0.0001 vs C=0.001 pour les contrôles (régularisation plus forte sur 1536 dims). Sélection canonique (val), mais la comparaison n'est pas à régularisation égale.
2. **Contrôles = seed0 unique** ; R2 = moyenne 3 seeds. Le test v3 étant identique, la comparaison reste valide, mais la variance single-run des contrôles (~0.002) s'applique.
3. **Dimension 1536 vs 768** : le contrôle frozen-fused (aussi 1536) ne fait que 0.486 → la dimension seule n'explique pas le 0.51, mais elle peut amplifier.
4. **Métrique** : `f1_macro_pres` = `f1_macro_all` ici (les 11 classes sont présentes au test v3) — coïncidence vérifiée, à documenter.
5. **Non déployable** : Design B exige le contexte au moment de prédire (double forward).

## 7. Implications et recommandations

- **Pour la thèse** : ne pas présenter R2 (0.508) comme un « modèle déployable meilleur » — le présenter comme la **borne d'information spatiale** : « le contexte voisin porte +0.025 de F1, mais son exploitation exige d'entraîner la fusion, pas de concaténer à la sonde ».
- **Contrôles complémentaires suggérés** (par ordre de valeur/coût) :
  1. **MLP de fusion** (1536→512→11) à la place du concat linéaire pendant l'entraînement B — la littérature (bilinear pooling, gated fusion, FiLM, GeoCLIP) suggère qu'un gating implicite peut encore gagner sur des vues redondantes imbriquées.
  2. **Sweep de taille de contexte** (512/1024/2048) en **frozen-fused** d'abord (courbe d'apport d'info, sans entraînement) pour tester saturation ; puis éventuellement Design B entraîné si la courbe le justifie.
  3. **Interprétabilité** : attention spatiale / grad-CAM du contexte pour montrer QUELLE information voisine est utilisée (argument qualitatif fort).
