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
