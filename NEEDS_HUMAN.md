# Points bloqués nécessitant une décision

Format : un point par section, daté en titre. Résolu → déplacer en bas dans "Résolu".

## 2026-09-02 — Sweep des tailles de contexte (512/1024/2048) : crops val/test locaux puis jobs Narval

Demande Bouguessa (mail début sept.) : regarder les tailles de contexte avant de
passer à des DINOv3 plus grands. Pallier frozen (sans entraînement) préparé :
`scripts/context_size_sweep.py` + `scripts/slurm_context_size_sweep.sh` (extraction
frozen fusionnée + sonde canonique fused / ctx-seul / ctx-permuté, machinerie de
`scripts/context_bouguessa_controls.py` — les points 1-2 du mail sur R2 sont calculés
en LOCAL, sans Narval). Accès cluster requis (`lmague@narval3`) :

1. **EN LOCAL** — générer les crops val/test manquants (les train crops 512/1024/2048
   sont déjà sur `$SCRATCH` ; 1024 a déjà val/test) :
   ```bash
   python scripts/context_crop.py \
       --split-csv spatial_datacurve/splits/frac100_seed0/val.csv spatial_datacurve/splits/frac100_seed0/test.csv \
       --context-sizes 512,2048 --out-size 224 --out-dir out/context
   cd out/context && zip -qr ../../context_512_valtest.zip context_512 \
                    && zip -qr ../../context_2048_valtest.zip context_2048
   scp context_512_valtest.zip context_2048_valtest.zip narval:$SCRATCH/
   ```
2. **Sur Narval** — le pallier frozen : `sbatch scripts/slurm_context_size_sweep.sh`
   (fusionne les zips train+val/test dans le job, ~12 h slice MIG, repartable par JSON).
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

## 2026-08-27 — Soumettre les jobs SLURM DINOv3 ViT-S/16 (Frozen + LoRA r=8) sur Narval

Infrastructure prête et vérifiée par relecture (code, configs, scripts SLURM — cf. `AGENT_MEMORY.md` 2026-08-27) : aucun accès GPU/cluster depuis la session qui a fait la relecture, donc rien n'a pu être exécuté. Reste à faire, humain requis (accès Narval `lmague@narval3`) :

1. `sbatch scripts/slurm_extract_vits16.sh` (extraction frozen, ~10-20 min)
2. `sbatch scripts/slurm_lora_dinov3_vits16.sh` (LoRA r=8, 3 seeds, ~1-2h)
3. Vérifier `ls $SCRATCH/embeddings/dinov3_vits16_lvd_{val,test,train}.npy`, ajouter `dinov3_vits16_lvd` à `configs/probe_all.yaml` et `configs/benchmark_12models.yaml` (`models:`), relancer le probe canonique, mettre à jour `all_models_canonical_merged.json`.

## Résolu

### 2026-07-15 — Retrain FT canoniques en 11cls, 3 seeds, recette Tier 1

Date de résolution : 2026-07-19. ResNet-50 full FT 3 seeds terminé (F1 = 0.4573 ± 0.0032), résultats dans `runs/frac100_seed{0,1,2}/metrics.json`. MHSA_cui abandonné (ne marchait pas).
