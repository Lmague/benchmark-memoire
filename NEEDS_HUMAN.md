# Points bloqués nécessitant une décision

Format : un point par section, daté en titre. Résolu → déplacer en bas dans "Résolu".

## 2026-09-07 — Ablation LoRA/PEFT SimDINOv2-B : pipeline séquentiel A→B→C, à soumettre sur Narval

Question : le « meilleur LoRA » DINOv3-B (r=2-8, blocs 6-11, α=2r) se transpose-t-il
à SimB, et est-ce que LoRA est même la bonne méthode PEFT ? Design séquentiel :

- **Stage A — exploration large, 13 bras × 3 seeds** (`sbatch scripts/slurm_lora_simb_stageA.sh`,
  array 0-12, ≈ 58 GPU-h) — isolation stricte des axes : seuls les 3 bras POSITION
  restreignent les blocs, tous les autres (rang, α, type) tournent sur TOUS les blocs :
  b611 / b05 / b911 ; rang r2a2 / r4a4 / r16a16 / r32a32 ; α r8a16 / r16a32 (scaling 2) ;
  **rsLoRA r8_rslora / r16_rslora** (α = r^{3/2} → scaling = √r : échelles de scaling
  1/2/2.83 à r8 et 1/2/4 à r16 — si le scaling 4 rattrape le niveau r8, le décrochage
  r=16 de DINOv3 était un artefact de scaling, pas de rang) ; r8a8_qkv (type, OUT_DIR
  séparé — collision de tag) ; **norm_tuning** (nouveau régime ajouté à src/models.py
  le 2026-09-07 : LayerNorms + head seulement, ~0.03 % params, réf. DEFLECT
  arXiv 2504.17397 ; config `configs/simdinov2_vitb16_norm.yaml`, lr.norm=1e-4 —
  les normes SONT l'adaptation). r=3 corrigé en r=4 (demande explicite 2026-09-07).
  Référence gratuite : canonique r8a8 tous blocs = 0.4781 ± 0.0028.
  Note biblio : rsLoRA PAS indexé dans Fusion — tentative library_add du 2026-09-07 :
  2 fausses correspondances (MindDiffuser doi 10.48550/arxiv.2303.14139 et LoRA-GA
  doi 10.48550/arxiv.2407.05000 portent la raison « rsLoRA » par erreur, à nettoyer
  dans /home/erazal/fusion/data/library/) ; ajouter rsLoRA à la main (Kalajdzievski,
  « The Impact of Scaling on LoRA », arXiv 2023 — ID exact à vérifier sur arXiv).
  Lecture (§4.4) : moyennes ± std, |Δ| < 0.005 = ex æquo → parsimonie.
- **ANALYSE COMMUNE** (humain + agent) → Stage B sur mesure : configs générées par
  `scripts/gen_simb_lora_grid.py --position <gagnant> [--qkv]`, soumission
  `sbatch scripts/slurm_lora_simb_stageB.sh` (array 0-7 no-op au-delà de la liste,
  liste surchargeable `--export=ALL,VARIANTS="..."`, 3 seeds, réutilise les seeds
  déjà faits via skip-if-done).
- **Stage C — fusion** : `scripts/merge_lora_simb.py --ckpt <tag>_best.pth --config
  <config du run> --out <tag>_merged.pth` — fusionne les adaptateurs dans les poids
  (via merge_lora_state_dict), 3 contrôles de non-régression (équivalence numérique
  module ≤ 1e-4, zéro clé LoRA résiduelle, rechargement par build_frozen_extractor),
  sortie au format `{"teacher": {"backbone.…"}}` directement exploitable pour
  extraction/géométrie sans adaptateurs.

Préflight in-job (comptage params) sur les deux stages. Git push OBLIGATOIRE avant
sbatch (incident 2026-08-30).

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
