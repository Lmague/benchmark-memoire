# Points bloqués nécessitant une décision

Format : un point par section, daté en titre. Résolu → déplacer en bas dans "Résolu".

## 2026-09-08 — Campagne « heads » LANCÉE sur les deux fronts (Narval fused + local tile-only)

**Front Narval (24 tags FUSED)** : audit 24/24 OK (après fix alias `dinov3_vitl16`),
job à soumettre par l'humain : `sbatch scripts/slurm_fusion_head_sweep.sh` (5 têtes :
lbfgs, lin-AdamW, MLP-2, bilinéaire-diag, FiLM). Rapatrier
`$SCRATCH/context_distill/fusion_heads/` → `results/context_distill/`.

**Front local (33 groupes du palier, tuile seule)** : `scripts/tile_head_sweep.py`
(3 têtes : lbfgs / lin-AdamW / MLP-2 — bil/FiLM sans sens sur vue unique), lancé le
2026-09-08 ~14:10 via `scripts/run_tile_head_sweep.sh` (nohup, 7 processus
mono-thread, idempotent). Contrôle de cohérence validé au smoke : lbfgs NormTuning
= 0,4783 = canonique exact. Sorties : `results/rapport_data/tile_heads/` (+
`_aggregate.csv`) ; suivi : `tail -f results/rapport_data/tile_heads/_runner.log`,
`ls tile_heads/*.json | wc -l` (cible 33). Durée attendue 4–10 h.

**Lecture à venir (ne rien citer avant)** : critère = `delta_vs_lin_adamw` > +0,01
(sur fused, par tête) ; attendu raisonnable au vu de juillet : rien ne dépasse.
Si confirmation sur 33 groupes → le « plateau plat » s'étend aux classifieurs non
linéaires (paragraphe de robustesse Article 1). Sinon → Article 3 réorienté vers
le head de fusion.

## 2026-09-08 — Campagne « fusion heads » non linéaires : PRÊTE À LANCER sur Narval

Teste si une tête non linéaire (MLP-2, bilinéaire diagonal, FiLM) récupère sur les
embeddings fused DÉJÀ SUR DISQUE la part d'interaction que la sonde linéaire laisse
sur la table (écart gelé-fusionné 0,495 vs fusion apprise R2 0,508).

- Script : `scripts/fusion_head_sweep.py` (5 têtes, lbfgs canonique mono-thread en
  baseline, sélection config sur val uniquement, delta citation = tête − lin_adamw).
- SLURM : `scripts/slurm_fusion_head_sweep.sh` — 7 processus mono-thread de front
  (par tags, jamais par BLAS), idempotent, ~24 tags (sweep gelé 5×3 + R2 3 seeds +
  R1/R3 sanity), estimé 8-15 h wall sur 36 h demandées.
- Lancer : `git pull && sbatch scripts/slurm_fusion_head_sweep.sh` ; puis rapatrier
  `$SCRATCH/context_distill/fusion_heads/` → `results/context_distill/`.
- Smoke test local passé : R2 seed0 lbfgs reproduit à 0,0001 (0,5097 vs 0,5098) ;
  en mode quick le MLP-2 est à +0,006 vs lin_adamw (sous la baseline lbfgs) — à
  confirmer sur grille complète. **Ne rien citer avant le run complet.**

## 2026-09-08 — Ablation LoRA/PEFT SimDINOv2-B : Stage A TERMINÉE + bootstrap apparié FAIT — Stage B abandonnée

**Bootstrap apparié fait le 2026-09-08** (extraction de `results/significance_matrix_tier.json`, n=10 000, BH α=0.05) → `results/bootstrap_stageA_paired_CANONICAL.json` : ancre vs b911 p=0.44, vs b611 p=0.66, vs QKV p=0.71, vs NormTuning p=0.94, BH rejeté nulle part. Le plateau PEFT est désormais un résultat statistique citable, pas une impression. **Stage B pleine grille : abandonnée** (rendement nul confirmé formellement). Restent optionnels : QKV×position (9 GPU-h) et matrice d'attribution du SimB entraîné (voir CONTROLES_BOUGUESSA.md § « Trous identifiés »). Commit de figeage : 0493424.

## 2026-09-07 — Ablation LoRA/PEFT SimDINOv2-B (archive, voir ci-dessus)

**Stage A faite** (13 bras × 3 seeds = 42 runs, terminés 2026-09-07, rapatriés hors
checkpoints, 24 Go restés sur `$SCRATCH`). Rapport : `results/lora_simb_ablation/RAPPORT_STAGE_A.md`.
**Palier plat : 0.4736-0.4812 (étendue 0.0077)**, rien ne bat l'ancre 0.4781 au-delà du
bruit. Lectures : position hauts≫bas (ordre DINOv3 transposé, b911 = 0.4805 ± 0.0002,
mi-budget) ; rang pente douce décroissante ; **scaling monotone négatif → rsLoRA
infirme** (le décrochage DINOv3 r16/32 n'était pas un artefact de scaling) ; QKV ≈ QV
(bruit) ; **norm_tuning = 0.4781 ± 0.0004 = ancre au millième près → LoRA n'apporte rien
qu'une mise à jour des normes** (cohérent DEFLECT + chapitre contexte).
Caveat majeur : classements val/test quasi indépendants (r16a64 val max/test min) —
sélection sur val = bruit.

**Prochaines étapes recommandées** : (1) PAS de Stage B pleine grille ; optionnelle
QKV×position (b911/b611, 2 configs × 3 seeds ≈ 9 GPU-h) ; (2) bootstrap apparié
ancre-vs-{QKV, b911, norm_tuning} sur embeddings locaux (transforme « tout est dans
le bruit » en résultat citable) ; (3) Stage C fusion sur b911 ou QKV (rendement F1 nul,
sert géométrie/déploiement) ; (4) lecture mémoire : « paysage PEFT plat, la valeur de
SimB est dans le contexte, pas dans l'adaptation ».

*(Aucun autre point bloquant au 2026-09-07. Les trois points d'accès cluster ci-dessous sont résolus — jobs tournés et résultats rapatriés. Restent des compléments non bloquants, listés dans `results/context_distill/CONTROLES_BOUGUESSA.md` § « Trous identifiés » : matrice d'attribution du SimB entraîné, bootstrap apparié, consolidation table maître/rapports.)*

## Résolu

### 2026-09-02 / 2026-09-05 — Contrôles Bouguessa + sweep frozen + SimB entraîné

Date de résolution : 2026-09-06/07 (résultats sur disque).

- Contrôles contexte (job `ctx_bouguessa_controls` 2354690) : contexte seul = 0.4780,
  permuté = 0.4745 (< tuile 0.4779) → gain spatial validé. `results/context_distill/CONTROLES_BOUGUESSA.md`.
- Sweep frozen 5 backbones × 3 tailles, SimL relance incluse : SimB @512 = 0.5059
  (meilleur frozen-fused), SimL @512 = 0.5022 → hypothèse « SimL ≥ 0.51 » infirmée.
- SimDINOv2-B @512 entraîné Design B (r2a4 et r8a16, 3 seeds, terminés 2026-09-06) :
  0.5030 ± 0.0012 et 0.5057 ± 0.0072 ≈ gelé-fusionné (0.5059) → l'affinage complet
  n'apporte rien sur SimB. Détail et trous restants :
  `results/context_distill/CONTROLES_BOUGUESSA.md` § « Résultats — SimDINOv2-B @512 entraîné ».

### 2026-08-27 — Soumettre les jobs SLURM DINOv3 ViT-S/16 (Frozen + LoRA r=8) sur Narval

Date de résolution : 2026-08-27 (jobs Narval 1872073/1872074, cf. `AGENT_MEMORY.md`).
Runs terminés et intégrés : frozen 0.4689, LoRA r=8 0.4774 ± 0.0022,
`all_models_canonical_merged.json` (26 modèles à la date), `registry.py` à jour.

### 2026-07-15 — Retrain FT canoniques en 11cls, 3 seeds, recette Tier 1

Date de résolution : 2026-07-19. ResNet-50 full FT 3 seeds terminé (F1 = 0.4573 ± 0.0032), résultats dans `runs/frac100_seed{0,1,2}/metrics.json`. MHSA_cui abandonné (ne marchait pas).

### (complément 2026-09-08 — décision utilisateur : TOUT sur Narval)

Le front local est abandonné (-runner arrêté, 5 JSON locaux jetés). Pipeline unique :

1. Local : `python3 scripts/prepare_head_sweep_push.py && bash scripts/push_head_sweep_inputs.sh`
   (rsync ~12,8 Go, idempotent, vers $SCRATCH/head_sweep_inputs/)
2. Narval : `sbatch scripts/slurm_head_sweep_all.sh` — phase 1 fused (24 tags × 5 têtes)
   + phase 2 tile-only (33 groupes × 3 têtes), 7 process mono-thread, 48 h allouées,
   idempotent par JSON (re-sbatch après coupure). Bug corrigé : std(ddof=1) sur les
   gelés 1-seed.
3. Rapatrier `$SCRATCH/head_results/{fusion_heads,tile_heads}` → mêmes chemins locaux.
