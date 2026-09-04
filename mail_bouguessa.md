# Suivi des échanges avec M. Bouguessa

## Mail de M. Bouguessa (reçu début septembre 2026 — date exacte à préciser)

Bonjour,

Je pense que tu as maintenant suffisamment de matière expérimentale pour le
mémoire. Il faut probablement commencer à rédiger. C'est là qu'on commencera
vraiment à voir une structure pour la suite.

Toutefois, afin de compléter l'analyse du modèle avec contexte, je te suggère
simplement deux expériences :

1. Tester le contexte seul, sans l'embedding de la tuile centrale, afin de mesurer
   ce que le contexte apporte à lui seul.
2. Faire un contrôle avec un contexte aléatoire (ou permuté), afin de vérifier que
   le gain vient bien de l'information spatiale associée à la tuile et non
   simplement de la concaténation d'une deuxième représentation.

Tu peux également regarder les différentes tailles de contexte que tu proposes
(512, 1024, 2048), avant de passer à des modèles DINOv3 plus grands.

Bonne journée,

-M.B.

Traitement : points 1 et 2 par sonde canonique sur les embeddings déjà extraits des
runs R2 (Design B, fusion 1536 = [tuile ; contexte], `results/context_distill/sig_embeddings/`)
— `scripts/context_bouguessa_controls.py` → `results/context_distill/controls_bouguessa/`
+ rapport `results/context_distill/CONTROLES_BOUGUESSA.md`. Tailles de contexte :
pallier frozen (sans entraînement) via `scripts/context_size_sweep.py` +
`scripts/slurm_context_size_sweep.sh` (prérequis local : crops val/test 512/2048 —
cf. `NEEDS_HUMAN.md`) ; pallier entraîné (R2 à 512/2048) en option si la courbe le
justifie. Message clé de la mise en garde : « commencer à rédiger » — la priorité
dévient le manuscrit, pas de nouvelles expériences lourdes.

---

## Mail du 30 juillet 2026 (documents du 30 juillet, OneDrive)

Bonjour,

J'ai regardé tes documents du 30 juillet (sur le OneDrive). Puisqu'il s'agit d'un
travail expérimental et afin d'enrichir davantage ton mémoire, je propose que tu
considères ce qui suit :

1. Refaire la courbe de quantité de données avec un sous-échantillonnage
   spatial/orthomosaïque, pas seulement par tuiles → une courbe complémentaire où
   la quantité de données est contrôlée par orthomosaïque ou par blocs spatiaux.
2. Réaliser une ablation du rang LoRA → LoRA r=8 est la seule adaptation qui
   améliore significativement DINOv3-B, il faut faire une petite ablation du rang
   LoRA (r = 2, 4, 8, 16 ou même 32).
3. Ajouter une comparaison coût/performance DINOv3-B + LoRA vs DINOv3-H+ gelé,
   par exemple.
4. Est-ce qu'il est possible d'exploiter davantage la comparaison géométrique
   contrôlée DINOv3-B gelé/Full/MHSA/LoRA (pas nécessaire d'ajouter d'autres
   métriques) ?

Bonne journée,
-M.B.

Traitement : fichier `rapport_bouguessa/analyse_encadrant.pdf` (réponses point par
point), sections ajoutées dans performances.pdf / datacurves.pdf / geometrie.pdf,
design spatial dans docs/spatial_datacurve_design.md + scripts/ (splits,
slurm_datacurve_spatial.sh).

---

## Objet : Suivi benchmark Arctic-TVC — 17 juillet 2026 (précédent)

Salut Mohamed,

Depuis notre dernier échange, trois changements concrets sur le benchmark :

1. Le comparateur fine-tuné précédent (vitb16_fulft_arctic, F1=0.4796) était une seed unique
   chanceuse — il a été supplanté par le pipeline Narval Full FT 3 seeds (0.4708 ± 0.0079).
   Tous les modèles FT sont désormais en 3 seeds (Full, MHSA, ExPLoRA, Scratch).

2. Le « plateau » des courbes d'apprentissage autour de 0.45-0.46 est un artefact de
   grille C — la data curve utilise C ∈ {0.01..10} (best_C=0.01 saturé 21/21), alors
   que le benchmark utilise {1e-4..10}. L'écart Δ = +0.021 est entièrement attribuable
   à la régularisation, pas à un plafond de données. Détail : fiche Performances, tableau 1.

3. Les corrélations géométrie↔F1 ont été recalculées sur un corpus canonique propre
   (n=9 frozen, modèles douteux exclus). Résultat : seule la silhouette est corrélée
   (ρ = 0.983, p < 10⁻⁴, survit à Benjamini-Hochberg). Toutes les métriques spectrales
   sont non significatives — leur significativité dans le corpus n=16 était portée par
   le contraste scratch (rang effectif ≈ 4) vs pré-entraînés (rang effectif ≈ 40-80).
   Détail : fiche Géométrie, tableau 1.

Résultat central actuel : DINOv3 ViT-L/16 LVD frozen (0.4792) vs ViT-B Full FT SOTA
3 seeds (0.4708 ± 0.0079), Δ = +0.0084 en faveur du frozen. L'ancien comparateur
(vitb16_fulft_arctic, 0.4796) était une seed unique chanceuse — il est supplanté.
Bootstrap bloqué — les prédictions par tuile sont sur Narval, pas de GPU local.

Points ouverts / à trancher avec toi :
- Bootstrap seed-aware : pour une comparaison FT (3 seeds) vs frozen, il faut un
  rééchantillonnage hiérarchique (seeds avec remise, puis tuiles). Le bootstrap actuel
  est seed-naïf. Faut-il le refaire ? Besoin d'accès Narval.
- ResNet-50 FT : aucun équivalent fiable n'existe (le checkpoint resnet50_arctic a le
  même problème de provenance que vitb16_fulft_arctic). Relancer un entraînement
  ResNet-50 sur Narval ou documenter l'absence comme limite ?
- Grille C data curve : les courbes actuelles sont sur grille tronquée. Re-probe
  nécessaire pour les rendre comparables au benchmark. Priorité ?
- Affiliation Mila de Laliberté : à confirmer pour le manuscrit.
- Plan SSL continu : les hyperparamètres sont listés dans la fiche 4 — à valider.

Fiches jointes (PDF, 2-5 pages chacune, zéro interprétation — définitions + tableaux +
figures uniquement) :
- 01_resultats.pdf — tableaux maîtres 11cls/8cls, barplots, paire vedette/contrôlée,
  modèles exclus
- 02_geometrie.pdf — définitions formelles des 10 métriques, corrélations n=9 avec BH,
  table n=16 non-canonique
- 03_performances.pdf — paramètres probe/bootstrap, grilles C, data curve avec référence
  benchmark
- 04_hyperparametres.pdf — pipeline SSL/fine-tuning actuel, plan CABO, table décidé/à
  discuter

É.
