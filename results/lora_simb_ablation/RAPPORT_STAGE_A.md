# STAGE A — Ablation LoRA/PEFT SimDINOv2-B : rapport

*Date : 2026-09-07/08. 13 bras × 3 seeds = 42 runs, fraction 100 %, split spatial v3, 11 cls.
Sources : `results/lora_simb_ablation/runs/*/metrics.json` (39) + `results/lora_simb_ablation_qkv/runs/` (3, bras QKV).
Recette : identique au canonique (`simdinov2_vitb16_lora.yaml`), probe interne datacurve
(sélection C sur val, refit, f1_macro_pres sur test). Ancre : canonique r8a8 TOUS blocs Q/V
de la campagne 3models = **0.4781 ± 0.0028** (non re-entraîné ici ; le bras QKV porte le même
tag mais est Q+K+V — pas de duplication QV).*

## 1. Tableau maître (trié par F1 test)

*Mise à jour 2026-09-08 : F1 **canonique** (reprobe mono-thread §4.8,
`results/simb_stageA_probe_CANONICAL.json` — coïncide avec le probe interne à ±0,0002
près sauf QKV seed2, voir note C). best_C/époque/val : probe interne (`metrics.json`).*

| Rang | Bras | F1 canonique (moy ± std) | seeds | val | best_ep |
|---|---|---|---|---|---|
| 1 | **b911** (r8a8, blocs 9-11) | **0.4804 ± 0.0002** | 0.4803 / 0.4806 / 0.4803 | 0.4640 | 18,18,18 |
| 2 | **b611** (r8a8, blocs 6-11) | 0.4802 ± 0.0027 | 0.4828 / 0.4805 / 0.4774 | 0.4750 | 18,18,13 |
| 3 | r2a2 (all) | 0.4793 ± 0.0024 | 0.4777 / 0.4820 / 0.4781 | 0.4759 | 13,18,13 |
| 4 | r8a8 QKV (all) | 0.4788 ± 0.0015 | 0.4770 / 0.4795 / 0.4799 | 0.4781 | 13,13,13 |
| 5 | r4a4 (all) | 0.4785 ± 0.0049 | 0.4743 / 0.4839 / 0.4774 | 0.4753 | 11,32,13 |
| 6 | **norm_tuning** (normes + head) | **0.4783 ± 0.0004** | 0.4778 / 0.4786 / 0.4784 | 0.4675 | 18,18,18 |
| — | *ancre canonique r8a8 all QV* | *0.4781 ± 0.0028* | — | — | — |
| 7 | b05 (r8a8, blocs 0-5) | 0.4770 ± 0.0024 | 0.4743 / 0.4790 / 0.4777 | 0.4708 | 28,32,41 |
| 8 | r16a16 (all) | 0.4769 ± 0.0016 | 0.4754 / 0.4785 / 0.4768 | 0.4774 | 13,13,13 |
| 9 | r8a16 (all, s=2) | 0.4760 ± 0.0018 | 0.4747 / 0.4781 / 0.4753 | 0.4764 | 13,18,18 |
| 10 | r8 rsLoRA (all, s=2.83) | 0.4753 ± 0.0023 | 0.4750 / 0.4778 / 0.4733 | 0.4753 | 18,13,18 |
| 11 | r32a32 (all) | 0.4750 ± 0.0017 | 0.4746 / 0.4769 / 0.4736 | 0.4799 | 13,13,13 |
| 12 | r16a32 (all, s=2) | 0.4749 ± 0.0026 | 0.4751 / 0.4774 / 0.4721 | 0.4789 | 13,13,18 |
| 13 | r16 rsLoRA (all, s=4) | 0.4737 ± 0.0035 | 0.4767 / 0.4744 / 0.4698 | 0.4809 | 13,13,18 |

**Étendue du palier canonique : 0.0067** (r16 rsLoRA → b911).

**Note C (QKV seed2) — instabilité de sélection du C.** Le probe *interne* du run QKV
seed2 sélectionne C=0,0001 (F1 test 0,4873, moyenne interne 0,4812) ; le reprobe
canonique (sélection sur sous-échantillon 20k) sélectionne C=0,001 (F1 test 0,4799,
moyenne canonique 0,4788). Un basculement de C déplace ce seed de **0,0074** — le même
phénomène qu'en 2026-07-26 (grille C de LoRA r=8 DINOv3, ±0,003 sur le test). Leçon :
au sommet du plateau, le choix de C pèse autant que le choix du bras ; la valeur
canonique est retenue pour le benchmark, l'écart est tracé par la table des
conventions du compendium (`_conv_gaps`).

## 2. Lecture par axe

### Position (seuls bras restreints) — même ordre que DINOv3
b911 (0.4804) ≥ b611 (0.4802) > ancre all (0.4781) > b05 (0.4770).
- L'ordre DINOv3 se transpose : blocs HAUTS ≥ tous blocs ≫ blocs bas.
- **b911 = 3 derniers blocs, moitié du budget b611, variance la plus faible de tout le
  job (± 0.0002)** — candidat parsimonieux pour le déploiement.
- ⚠️ MAIS val ↔ test divergent : b911 a la val la plus BASSE des bras position
  (0.4640 vs 0.4750 b611). Sur val, le « gagnant » serait r16a64 (val 0.4809, test le
  PIRE, 0.4736). Les classements val et test sont quasi indépendants → aucune
  sélection d'arm ne peut être justifiée par la val seule ; cf. caveat §4.

### Rang (scaling 1, tous blocs) — pente douce, pas de falaise
r2 (0.4794) > r4 (0.4785) > r8 (0.4781) > r16 (0.4769) > r32 (0.4751). **Monotone
décroissante** (−0.0043 de r2 à r32). Même direction que DINOv3 mais sans le
décrochage brutal : sur SimB, le rang pénalise doucement.

### Scaling / rsLoRA — hypothèse INFIRMÉE, et le contraire de rsLoRA
- r8 : s=1 → 0.4781, s=2 → 0.4761, s=2.83 → 0.4753 (monotone ↓).
- r16 : s=1 → 0.4769, s=2 → 0.4750, s=4 → 0.4736 (monotone ↓).
- **Plus de scaling = systématiquement pire, aux deux rangs.** rsLoRA (le scaling
  devrait croître en √r pour les grands rangs) est donc réfuté sur ces données, et le
  décrochage r=16/r=32 observé sur DINOv3 n'était **pas** un artefact de scaling —
  c'est un effet de rang réel. Convention α=r (scaling 1) validée pour SimB.

### Type Q+K+V vs Q+V
QKV canonique (0.4788 ± 0.0015) vs ancre QV (0.4781 ± 0.0028) : Δ = +0.0007, DANS LE BRUIT
(§4.4) ; le seed interne isolé (0,4873, C=0,0001) retombe à 0,4799 en canonique (C=0,001, note C). K n'apporte rien de mesurable.

### NormTuning — LE résultat structurel de la Stage A
**0.4783 ± 0.0004** : identique à l'ancre LoRA r8a8 au millième près (Δ = +0.0002),
variance la plus serrée des 42 runs, sur ~0.1 % des paramètres (LayerNorms + head,
aucun adaptateur). **LoRA n'apporte rien qu'une mise à jour des normes ne donne
déjà.** Cohérent avec DEFLECT (NormTuning ≈ oracle à 0.03 % sur 5 tâches RS) et avec
le chapitre contexte (le plafond SimB est informationnel, pas méthodologique).

## 3. Réponse à la question du pipeline

> Le « meilleur LoRA » DINOv3-B se transpose-t-il à SimB ?

**Non, et la question est vide : il n'y a pas de meilleur LoRA sur SimB.** Tout le
palier canonique tient dans 0.0067, sous le seuil d'interprétabilité individuel (σ
inter-seed jusqu'à 0.0054) ; ni rang, ni α, ni type, ni position ne produit de gain
net. Ce qui se transpose : l'ordre position (hauts ≫ bas) et la pente rang (plus de
rang = légèrement pire).

## 4. Caveats

1. **Divergence val/test** : classements quasi indépendants (r16a64 : val max / test
   min ; b911 : test n°2 / val min). Sélectionner un bras sur la val serait du bruit.
   Le test v3 est la référence (identique à toutes les baselines du projet).
2. **σ de QKV (0.0054)** porté par un seed (0.4873) — ne pas citer sa moyenne seule.
3. Le probe interne (datacurve) diffère du probe canonique du dépôt ; l'ancre 0.4781
   vient du probe canonique. Pour tout chiffre publié : refaire le probe canonique
   (mono-thread §4.8) sur les embeddings rapatriés — les 42 embeddings sont locaux.
4. best_C = 0.001 quasi partout (sauf QKV s2 : 0.0001) → comparaison à régularisation
   homogène, mieux que le chapitre contexte (fused C=1e-4).

## 5. Recommandations pour la suite

1. **Stage B pleine grille : NE PAS la lancer** — rendement attendu nul au vu du
   palier plat (tout est déjà dans le bruit). La seule combinaison non testée au-dessus
   du bruit est type×position : **QKV sur b911 et/ou b611** (2 configs × 3 seeds ≈
   9 GPU-h) si on veut épuiser le design — optionnel.
2. **Tester formellement le plateau** : bootstrap apparié ancre-vs-{QKV, b911,
   norm_tuning} sur les embeddings locaux (mêmes indices de rééchantillonnage que
   `significance_tier.py`) — transforme « tout est dans le bruit » en résultat
   statistique citable.
3. **Stage C (fusion)** : candidat = b911 (parsimonie + stabilité) ou QKV (meilleure
   moyenne). La fusion ne changera pas le F1 (identité mathématique), elle sert la
   géométrie et le récit de déploiement. Priorité néanmoins au bootstrap du point 2.
4. **Pour le mémoire** : la lecture à écrire est « le paysage PEFT de SimB est plat ;
   l'adaptation de SimB vaut ce que vaut NormTuning (~0.1 % des params) ; sa valeur
   est dans le contexte spatial (0.5059-0.5080), pas dans l'adaptation des poids ».
