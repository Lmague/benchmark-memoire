# Points bloqués nécessitant une décision

Format : un point par section, daté en titre. Résolu → déplacer en bas dans "Résolu".

## 2026-09-02 — Contrôles Bouguessa (contexte) : jobs Narval à soumettre

Demande Bouguessa (mail début sept.) : (1) contexte seul, (2) contexte permuté,
(3) tailles de contexte 512/1024/2048. Préparé, **accès cluster requis** (`lmague@narval3`) :

1. **Points 1-2 sur R2 — JOB CPU pur** (`sbatch scripts/slurm_context_bouguessa_controls.sh`,
   ~2 h sur 4 workers, repartable) : sonde canonique (fused / contexte-seul /
   contexte-permuté ×5 / tile) sur les sig_embeddings DÉJÀ présents sur
   `$SCRATCH/context_distill/sig_embeddings/` (extraits par
   `slurm_context_distill_extract_sig.sh`). Aucun GPU, aucun zip à dézipper.
   Puis rapatrier : `scp -r narval:$SCRATCH/context_distill/controls_bouguessa results/context_distill/`
2. **Tailles de contexte, pallier frozen** — deux jobs séquentiels :
   a. EN LOCAL d'abord : générer les crops val/test manquants (train crops
      512/1024/2048 déjà sur `$SCRATCH` ; 1024 a déjà val/test) :
      ```bash
      python scripts/context_crop.py \
          --split-csv spatial_datacurve/splits/frac100_seed0/val.csv spatial_datacurve/splits/frac100_seed0/test.csv \
          --context-sizes 512,2048 --out-size 224 --out-dir out/context
      cd out/context && zip -qr ../../context_512_valtest.zip context_512 \
                       && zip -qr ../../context_2048_valtest.zip context_2048
      scp context_512_valtest.zip context_2048_valtest.zip narval:$SCRATCH/
      ```
   b. Sur Narval : `sbatch scripts/slurm_context_size_sweep.sh` (extraction GPU
      seule, fusionne les zips train+val/test dans le job) PUIS
      `sbatch scripts/slurm_context_size_sweep_probes.sh` (probes CPU : fused /
      tile / ctx / perm×3 par taille).
3. **Optionnel** (si la courbe frozen justifie R2 entraîné à 512/2048) — reconstruire
   les zips complets puis soumettre les entraînements :
   ```bash
   cd $SCRATCH && for S in 512 2048; do
     mkdir -p m$S && unzip -q context_$S.zip -d m$S && unzip -q context_${S}_valtest.zip -d m$S
     (cd m$S && zip -qr ../context_${S}_full.zip context_$S) && rm -rf m$S
     mv context_$S.zip context_${S}_trainonly.zip && mv context_${S}_full.zip context_$S.zip
   done
   sbatch scripts/slurm_context_distill.sh 512 B
   sbatch scripts/slurm_context_distill.sh 2048 B
   ```

⚠️ Avant tout sbatch : `git push` ici puis `git pull` sur Narval (`$HOME/benchmark-memoire`)
— cf. incident 2026-08-30 (§17 du README contexte) : un script non poussé = job à vide.

## 2026-08-27 — Soumettre les jobs SLURM DINOv3 ViT-S/16 (Frozen + LoRA r=8) sur Narval

Infrastructure prête et vérifiée par relecture (code, configs, scripts SLURM — cf. `AGENT_MEMORY.md` 2026-08-27) : aucun accès GPU/cluster depuis la session qui a fait la relecture, donc rien n'a pu être exécuté. Reste à faire, humain requis (accès Narval `lmague@narval3`) :

1. `sbatch scripts/slurm_extract_vits16.sh` (extraction frozen, ~10-20 min)
2. `sbatch scripts/slurm_lora_dinov3_vits16.sh` (LoRA r=8, 3 seeds, ~1-2h)
3. Vérifier `ls $SCRATCH/embeddings/dinov3_vits16_lvd_{val,test,train}.npy`, ajouter `dinov3_vits16_lvd` à `configs/probe_all.yaml` et `configs/benchmark_12models.yaml` (`models:`), relancer le probe canonique, mettre à jour `all_models_canonical_merged.json`.

## Résolu

### 2026-07-15 — Retrain FT canoniques en 11cls, 3 seeds, recette Tier 1

Date de résolution : 2026-07-19. ResNet-50 full FT 3 seeds terminé (F1 = 0.4573 ± 0.0032), résultats dans `runs/frac100_seed{0,1,2}/metrics.json`. MHSA_cui abandonné (ne marchait pas).
