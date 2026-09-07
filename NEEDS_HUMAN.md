# Points bloqués nécessitant une décision

Format : un point par section, daté en titre. Résolu → déplacer en bas dans "Résolu".

## 2026-09-07 — Ablation LoRA/PEFT SimDINOv2-B : **Stage A TERMINÉE** — Stage B à redéfinir après analyse

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
