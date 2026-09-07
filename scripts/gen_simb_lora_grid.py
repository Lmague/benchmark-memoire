#!/usr/bin/env python3
"""Génère les configs de la Stage B (grille r × α) pour l'ablation LoRA SimDINOv2-B.

Usage (APRÈS la Stage A, avec la position gagnante) :

    # grille complète à la position gagnante (ex. b611) :
    python scripts/gen_simb_lora_grid.py --position b611

    # grille + bras QKV au r8a8 de la position gagnante :
    python scripts/gen_simb_lora_grid.py --position b611 --qkv

    # sans restriction de blocs (tous) :
    python scripts/gen_simb_lora_grid.py --position none

Écrase les configs existantes du même nom (idempotent). Les tags de run encodent
r/α/blocs (datacurve_one_run.py) → aucun risque de collision entre positions ;
SEULE EXCEPTION : target_modules absent du tag → le bras --qkv doit tourner dans
un OUT_DIR séparé (géré par slurm_lora_simb_stageB.sh).
"""
from __future__ import annotations

import argparse
import os

TEMPLATE = """# SimDINOv2 ViT-B/16 (iNat21 Plantae) — Stage B (grille r × α), position : {pos_desc}
# Généré par scripts/gen_simb_lora_grid.py (pipeline A→B→C, cf. slurm_lora_simb_stageB.sh).
# Base : configs/simdinov2_vitb16_lora.yaml (canonique r8a8 TOUS blocs = 0.4781 ± 0.0028).
# n_full_ft_blocks: 0 obligatoire (défaut config = 2 → full-FT blocs {{5,11}} !).
checkpoint: "simdinov2_vitb_inat21plantae.pth"
model:
  name: simdinov2_vitb16
  num_classes: 11
  norm: simdino_inat

regime: lora

lora:
  r: {r}
  alpha: {alpha}
  target_modules: {targets}
  n_full_ft_blocks: 0
  full_ft_block_indices: []
  lora_block_indices: {blocks}
  dropout: 0.0

optim:
  weight_decay: 0.05
  lr:
    lora: 1.0e-4
    full_late: 5.0e-5
    norm: 1.0e-5
    head: 1.0e-4

schedule:
  warmup_epochs: 3

train:
  epochs: 50
  patience: 10
  head_only_epochs: 0
  amp_dtype: bfloat16
  grad_clip: 1.0
  early_stop_metric: f1_macro_pres

probe:
  C_grid: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
"""


def fmt_blocks(bl):
    if bl is None:
        return "null"
    return "[" + ",".join(str(i) for i in bl) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--position", required=True, choices=["b05", "b611", "b911", "none"],
                    help="bloc gagnant de la Stage A (b05 = [0..5], b611 = [6..11], "
                         "b911 = [9..11], none = tous les blocs)")
    ap.add_argument("--ranks", default="2,4,8,16")
    ap.add_argument("--alphas", default="r,2r",
                    help="'r' (scaling 1.0) et/ou '2r' (scaling 2.0), séparés par des virgules")
    ap.add_argument("--qkv", action="store_true",
                    help="ajoute une config Q+K+V au r8a8 de la position (OUT_DIR séparé !)")
    args = ap.parse_args()

    blocks = None
    pos_desc = "TOUS les blocs"
    if args.position != "none":
        lo = {"b05": 0, "b611": 6, "b911": 9}[args.position]
        blocks = list(range(lo, 12))
        pos_desc = f"blocs {lo}-11"

    ranks = [int(x) for x in args.ranks.split(",")]
    alphas = []
    for a in args.alphas.split(","):
        a = a.strip().lower()
        if a == "r":
            alphas.append("r")
        elif a == "2r":
            alphas.append("2r")
        else:
            alphas.append(float(a))  # alpha explicite (ex. 32)

    written = []
    pos_short = "" if args.position == "none" else f"_{args.position}"
    for r in ranks:
        for a in alphas:
            alpha = float(r) if a == "r" else float(2 * r) if a == "2r" else float(a)
            suffix = f"r{r}a{int(alpha)}{pos_short}"
            path = f"configs/simdinov2_vitb16_lora_{suffix}.yaml"
            with open(path, "w") as f:
                f.write(TEMPLATE.format(pos_desc=pos_desc, r=r, alpha=alpha,
                                        targets="[q, v]", blocks=fmt_blocks(blocks)))
            written.append(suffix)

    if args.qkv:
        blocks_str = fmt_blocks(blocks)
        suffix = f"r8a8{pos_short}_qkv"
        path = f"configs/simdinov2_vitb16_lora_{suffix}.yaml"
        with open(path, "w") as f:
            f.write(TEMPLATE.format(pos_desc=pos_desc + " — bras type Q+K+V (tag identique au QV → OUT_DIR séparé)",
                                    r=8, alpha=8.0, targets="[q, k, v]", blocks=blocks_str))
        written.append(suffix + " (qkv — OUT_DIR séparé)")

    print(f"Position : {pos_desc}")
    for s in written:
        print("  écrit :", s)


if __name__ == "__main__":
    main()
