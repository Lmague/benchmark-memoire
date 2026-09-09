#!/usr/bin/env python3
"""Audit des embeddings disponibles pour la campagne fusion-heads.

Répond à : « ai-je TOUT ce qu'il faut sur Narval pour lancer
slurm_fusion_head_sweep.sh ? » — sans rien calculer, lecture seule, ~30 s.

Pour chaque tag attendu, vérifie :
  1. existence du dossier sig_embeddings/<tag>/ ;
  2. les 6 fichiers train/val/test(.npy + _labels.npy) ;
  3. cohérence : dim train==val==test, dim paire (2 vues) pour les tags fused,
     n_train ~49k / n_val ~13k / n_test 17 598, labels ∈ 0..10 ;
  4. non-vide / non-corrompu (le premier bloc se lit).

Attendus (d'après les rapports de campagne — see CONTROLES_BOUGUESSA.md) :
  - sweep gelé  : {dinov3_vits16,dinov3_vitb16_lvd,dinov3_vitl16_lvd,
                   simdinov2_vitb16,simdinov2_vitl16}_FROZEN_fused_ctx{512,1024,2048}
                  _frac100_seed0                      → 15 tags, fused
  - R2 entraîné : dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed{0,1,2}
                  → 3 tags, fused 1536
  - sanity A    : dinov3_vitb16_lvd_ctxdistill_dA_{tL,tEMA}_ctx1024_r2a4
                  _frac100_seed{0,1,2}                → 6 tags, vue seule 768
  - HORS PÉRIMÈTRE (à récupérer AVANT de pouvoir les sonder) : les 6 runs
    SimDINOv2-B @512 entraînés (ctxdistill_dB_tSL_ctx512_{r2a4,r8a16}_seed{0,1,2})
    — signalés « pas d'embeddings rapatriés » dans CONTROLES_BOUGUESSA.md.
    Le script les cherche quand même et les signale MANQUANT ou PRÉSENT.

Usage (sur Narval, dans le dépôt cloné) :
    python3 scripts/check_fusion_heads_inputs.py --sig-dir $SCRATCH/context_distill/sig_embeddings
Sortie : rapport par tag (OK / MANQUANT / INCOHERENT) + compte final + code
d'exit 0 si tous les tags du périmètre sont OK, 1 sinon.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

VUE_FROZEN = {"dinov3_vits16": (384, ["dinov3_vits16"]),
              "dinov3_vitb16_lvd": (768, ["dinov3_vitb16_lvd"]),
              "dinov3_vitl16_lvd": (1024, ["dinov3_vitl16_lvd", "dinov3_vitl16"]),
              "simdinov2_vitb16": (768, ["simdinov2_vitb16"]),
              "simdinov2_vitl16": (1024, ["simdinov2_vitl16"])}
# Aliases : l'ancien batch (slurm_context_frozen_models.sh) écrit ViT-L sous
# `dinov3_vitl16` (sans suffixe _lvd) — même contenu, tag différent.
N_TRAIN_MIN, N_VAL_MIN, N_TEST = 40_000, 10_000, 17_598


def expected_tags(sig_dir: str = ""):
    tags = []
    for m, (d, aliases) in VUE_FROZEN.items():
        chosen = m
        if sig_dir:
            for a in aliases:
                for size in (512, 1024, 2048):
                    if os.path.isdir(os.path.join(
                            sig_dir, f"{a}_FROZEN_fused_ctx{size}_frac100_seed0")):
                        chosen = a
                        break
        for size in (512, 1024, 2048):
            tags.append((f"{chosen}_FROZEN_fused_ctx{size}_frac100_seed0", 2 * d))
    base = "dinov3_vitb16_lvd_ctxdistill"
    for seed in (0, 1, 2):
        tags.append((f"{base}_dB_tL_ctx1024_r2a4_frac100_seed{seed}", 1536))
    for des in ("dA_tL", "dA_tEMA"):
        for seed in (0, 1, 2):
            tags.append((f"{base}_{des}_ctx1024_r2a4_frac100_seed{seed}", 768))
    hors = []
    for r in ("r2a4", "r8a16"):
        for seed in (0, 1, 2):
            hors.append(f"simdinov2_vitb16_ctxdistill_dB_tSL_ctx512_{r}_frac100_seed{seed}")
    return tags, hors


def check(tag_dir: str, exp_dim: int | None) -> tuple[bool, str]:
    missing = [f for s in ("train", "val", "test")
               for f in (f"{s}.npy", f"{s}_labels.npy")
               if not os.path.exists(os.path.join(tag_dir, f))]
    if missing:
        return False, "fichiers manquants: " + ",".join(missing)
    dims, ns = set(), {}
    for s in ("train", "val", "test"):
        try:
            X = np.load(os.path.join(tag_dir, f"{s}.npy"), mmap_mode="r")
        except Exception as e:
            return False, f"{s}.npy illisible: {e}"
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            return False, f"{s}.npy vide/malformé shape={X.shape}"
        dims.add(X.shape[1]); ns[s] = X.shape[0]
        _ = X[0, :min(8, X.shape[1])].sum()  # premier bloc se lit (pas de trou)
        y = np.load(os.path.join(tag_dir, f"{s}_labels.npy"), mmap_mode="r")
        if y.shape[0] != X.shape[0]:
            return False, f"{s}: labels {y.shape[0]} != feats {X.shape[0]}"
        if y.max() > 10 or y.min() < 0:
            return False, f"{s}: labels hors 0..10 (max={int(y.max())})"
    (dim,) = dims
    if exp_dim and dim != exp_dim:
        return False, f"dim {dim} != attendue {exp_dim}"
    if exp_dim and dim % 2:
        return False, f"tag fused mais dim impaire {dim}"
    if ns["test"] != N_TEST:
        return False, f"n_test {ns['test']} != {N_TEST}"
    if ns["train"] < N_TRAIN_MIN or ns["val"] < N_VAL_MIN:
        return False, f"n_train/n_val trop petits {ns}"
    return True, f"dim={dim} n={ns['train']}/{ns['val']}/{ns['test']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sig-dir", required=True)
    args = ap.parse_args()
    tags, hors = expected_tags(args.sig_dir)
    n_ok = 0
    print(f"[audit] sig-dir = {args.sig_dir}\n")
    for tag, exp_dim in tags:
        ok, msg = check(os.path.join(args.sig_dir, tag), exp_dim)
        n_ok += ok
        print(f"  {'OK ' if ok else '✗  '} {tag}: {msg}")
    print()
    for tag in hors:
        present = os.path.isdir(os.path.join(args.sig_dir, tag))
        print(f"  {'PRÉSENT' if present else 'absent '} {tag} (HORS PÉRIMÈTRE — "
              "SimB entraîné, extraction sig manquante; à traiter avant citation)")
    total = len(tags)
    print(f"\n[audit] {n_ok}/{total} tags du périmètre prêts.")
    if n_ok < total:
        manquants = [t for t, d in tags
                     if not check(os.path.join(args.sig_dir, t), d)[0]]
        print("[audit] MANQUANTS/INCOHÉRENTS :")
        for t in manquants:
            print(f"  - {t}")
        print("[audit] → relancer l'extraction (slurm_context_frozen_models.sh / "
              "slurm_context_distill_extract_sig.sh) AVANT sbatch slurm_"
              "fusion_head_sweep.sh, ou accepter que le sweep tourne partiel.")
        sys.exit(1)
    print("[audit] → GO : sbatch scripts/slurm_fusion_head_sweep.sh")


if __name__ == "__main__":
    main()
