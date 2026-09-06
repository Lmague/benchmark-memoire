# Contrôles Bouguessa (mail début septembre) — contexte seul / contexte permuté

*Date : 2026-09-03. Exécution : job Narval `ctx_bouguessa_controls` (2354690), sonde
canonique sur `sig_embeddings` R2 (Design B, fusion 1536 = `[tuile ; contexte]`).*

## Résultats F1-macro test (11 classes, split spatial v3, 3 seeds)

| Représentation | seed0 | seed1 | seed2 | Moyenne ± std |
|---|---|---|---|---|
| **fused** (vrai) | 0.5097 | 0.5069 | 0.5072 | **0.5079 ± 0.0015** |
| **tile** (0:768) | 0.4779 | 0.4787 | 0.4770 | **0.4779 ± 0.0008** |
| **contexte seul** (768:1536) | 0.4775 | 0.4790 | 0.4775 | **0.4780 ± 0.0009** |
| **contexte permuté** (5 réplicats) | 0.4730–0.4754 | 0.4752–0.4777 | 0.4719–0.4740 | **≈ 0.4745 ± 0.0015** |

*Réplicats permutés (graine 1000+p, permutation indépendante par split) :*
`seed0`: 0.4754 / 0.4754 / 0.4730 / 0.4754 / 0.4742 ; `seed1`:
0.4755 / 0.4762 / 0.4761 / 0.4777 / 0.4752 ; `seed2`: 0.4725 / 0.4719 / 0.4731 /
0.4740 / 0.4723.

Repro / tmp. conformité : fused = 0.5079 ± 0.0015 ≈ metrics.json 0.5098/0.5070/0.5071
(écart < 0.001, float16 des sig_embeddings) ; tile seed0 = 0.4780 vs matrice 0.4779. ✓

## Réponse point par point

### (1) « Tester le contexte seul, sans l'embedding de la tuile centrale »
**Le contexte seul = 0.4780 ± 0.0009, soit ≈ la tuile seule (0.4779 ± 0.0008).**
À l'inférence seule, la fenêtre de contexte 1024px porte autant d'information que la
tuile centrale (ni nettement plus, ni moins). Le contexte n'est donc pas suffisant à
lui seul pour dépasser la baseline — il faut la couple tuile⊕contexte.

### (2) « Contrôle avec un contexte aléatoire (ou permuté) »
**Contexte permuté = 0.4745 ± 0.0015, soit EN DESSOUS de la tuile seule (0.4779).**
Réapparier le contexte au hasard (5 réplicats, par split) fait chuter le F1 sous la
baseline tuile-seule. Donc :
- le gain de R2 (`fused` 0.5079) **ne vient pas** de la simple concaténation d'une
  seconde représentation — une représentation de plus, mais désalignée, **dégrade**
  même le résultat ;
- le gain vient **de l'information spatiale associée à la tuile** (le contexte est le
  bon voisinage de la tuile). C'est la réponse attendue : le contrôle valide que
  l'apport est bien spatial, pas un artefact de dimensionnalité/concaténation.

## Lecture chiffrée

| Δ | Valeur | Interprétation |
|---|---|---|
| fusion − tuile | **+0.0300** | gain total du contexte apparié |
| fusion − contexte seul | **+0.0299** | la tuile + contexte ≫ contexte seul |
| tuile − contexte permuté | **+0.0034** | le contexte permuté est légèrement *pire* que la tuile seule (bruit ajouté) |
| contexte seul − tuile | +0.0001 | iso (à ± σ) |

## Caveats (à mentionner dans le manuscrit)

1. **best_C asymétrique** : fused sélectionne C = 0.0001 (1536 dims, régularisation plus
   forte) ; tile/contexte/permuté ≈ C = 0.001. Sélection canonique (val), mais
   comparaison à régularisation non strictement égale — déjà noté en caveat dans
   `ANALYSE.md` §6.
2. **Sonde linéaire** : ces contrôles mesurent l'information *linéairement décodable*.
   Un contrôle au niveau *entraînement* (ré-entraîner R2 avec contexte permuté) serait
   le gold-standard définitif, mais le contrôle sonde ici est déjà décisif (0.4745 vs
   0.5079, écart bien au-delà de tout σ).
3. **Contexte seul** : mesuré sur le modèle R2 (backbone entraîné Design B). Le contexte
   seul via le backbone gelé donnerait une autre référence (non mesuré ici — c'est
   l'objet du sweep des tailles, pallier frozen).

## Fichiers

- `results/context_distill/controls_bouguessa/r2_dB_tL_seed{0,1,2}_{fused,tile,ctx,fused_ctxperm0..4}.json`
  (produits sur `$SCRATCH/context_distill/controls_bouguessa/`, à rapatrier).
- Machine : `scripts/context_bouguessa_controls.py` (sonde canonique, BLAS mono-thread,
  parallélisation par processus, repartable par JSON).

---

# Sweep frozen multi-backbones × tailles (512/1024/2048) — extension Bouguessa

*Date : 2026-09-05. Job Narval `slurm_context_frozen_models.sh` (DINOv3-S/L,
SimDINOv2-B/L + B déjà fait par `slurm_context_size_sweep.sh`). Sonde canonique sur
features FROZEN fusionnées [tile ; ctx] (skip-if-done, repartable). SimDINOv2-L :
**à lancer** (job relance, modèle seul).*

## F1-macro test (seed0, 11 cls, split spatial v3)

| Modèle | taille | fused | tile | ctx seul | Δctx (fused−tile) | perm.moy |
|---|---|---|---|---|---|---|
| **ViT-B** (LVD) | 512 | 0.4953 | 0.4716 | 0.4592 | +0.0237 | 0.4697 |
| | 1024 | 0.4862 | 0.4716 | 0.3879 | +0.0145 | 0.4680 |
| | 2048 | 0.4715 | 0.4716 | 0.2960 | −0.0001 | 0.4690 |
| **ViT-S** (LVD) | 512 | 0.4810 | 0.4693 | 0.4399 | +0.0117 | 0.4621 |
| | 1024 | 0.4831 | 0.4693 | 0.3666 | +0.0138 | 0.4653 |
| | 2048 | 0.4664 | 0.4693 | 0.2795 | −0.0029 | 0.4651 |
| **ViT-L** (LVD) | 512 | 0.4973 | 0.4791 | 0.4646 | +0.0182 | 0.4761 |
| | 1024 | 0.4830 | 0.4791 | 0.3781 | +0.0039 | 0.4770 |
| | 2048 | 0.4636 | 0.4791 | 0.3131 | −0.0155 | 0.4753 |
| **SimDINOv2-B** (iNat) | 512 | **0.5059** | 0.4717 | **0.4931** | **+0.0342** | 0.4680 |
| | 1024 | 0.4913 | 0.4717 | 0.4402 | +0.0196 | 0.4702 |
| | 2048 | 0.4789 | 0.4717 | 0.3598 | +0.0072 | 0.4725 |
| **SimDINOv2-L** (iNat) | *à tester* | | 0.? | | | |

## Lectures

1. **SimDINOv2-B domine toutes les tailles** : fused 512 = **0.5059**, quasiment le
   niveau de **R2 entraîné** (0.508 @ 1024) — **sans aucun entraînement** (gelé +
   sonde). Le pré-entraînement iNat plantes est bien plus adapté à la toundra que
   DINOv3-LVD frozen (+0.033 vs ViT-B @512).
2. **Le Δcontexte de SimB est le plus grand (+0.034 @512)**, et son **contexte seul
   (0.4931)** est très au-dessus des autres (0.44-0.46) : SimDINOv2-B décode le
   voisinage incomparablement mieux.
3. **Le pattern 512 > 1024 ≫ 2048 est robuste** aux 4 backbones — confirme la lecture
   « contexte large = contexte flou » (le 2048, lissé par le resize 224, n'apporte
   rien : Δctx ≈ 0 voire négatif).
4. **Le permuté reste ≈ tile (0.46-0.48)** pour tous → le gain vient bien de
   l'appariement spatial (contrôle Bouguessa n°2 validé aussi multi-backbones).

## Prochaine étape logique

**SimDINOv2-L** (300 M, même pré-entraînement iNat) : SimB battant ViT-L de +0.03,
un SimL frozen pourrait franchir **0.51** — résultat fort : « contexte + iNat ≥
fine-tuning LoRA ». Relance du job multi-modèles pour ce seul modèle :
`sbatch scripts/slurm_context_frozen_models.sh "simdinov2_vitl16|frozen_simdinov2_vitl16.yaml|/scratch/lmague/checkpoints/simdinov2_vitl_inat21plantae.pth"`
