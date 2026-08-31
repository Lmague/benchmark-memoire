#!/usr/bin/env python3
"""Extraction des embeddings train/val/test pour un modèle context_distill.

Prépare les données du bootstrap canonique (`scripts/rapport/significance_tier.py`) :
chaque run écrit un dossier ``<out>/sig_embeddings/<tag>/`` avec
``train.npy``, ``val.npy``, ``test.npy`` (+ ``*_labels.npy``) au format attendu
par ``load_split(kind="ft", path, seed)`` (labels en 11 classes, RHOL retirée).

Design A : tuile seule (768). Design B : fusionné tuile⊕contexte (1536) — la
représentation qui donne le F1 0,508.

Usage (via scripts/slurm_context_distill_extract_sig.sh) :
    python scripts/context_distill_extract_sig.py \\
        --config configs/context_distill_dinov3b.yaml \\
        --context-dir $SLURM_TMPDIR/context_1024 \\
        --out-dir $SCRATCH/context_distill \\
        --ckpt-path $SCRATCH/context_distill/checkpoints/<tag>_best.pth \\
        --tag <tag> --fused
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import sys

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from src.config import load_config
from src.models import load_finetuned_ssl_backbone
from datacurve_one_run import _extract_backbone_embeddings, _apply_11cls_remap
from context_distill import _extract_fused_embeddings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--context-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="$SCRATCH/context_distill")
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--tag", required=True, help="nom de sortie (tag du run, avec seed)")
    ap.add_argument("--fused", action="store_true", help="Design B : features fusionnées 1536")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_key = cfg.model.name
    if not _os.path.exists(args.ckpt_path):
        print(f"[extract-sig] ERROR checkpoint introuvable : {args.ckpt_path}", flush=True)
        sys.exit(1)

    out_dir = _os.path.join(args.out_dir, "sig_embeddings", args.tag)
    _os.makedirs(out_dir, exist_ok=True)

    if args.fused:
        feats = _extract_fused_embeddings(model_key, args.ckpt_path, cfg, args.context_dir,
                                          splits=("train", "val", "test"))
        for s, (E, L) in feats.items():
            np.save(_os.path.join(out_dir, f"{s}.npy"), E.astype(np.float32))
            np.save(_os.path.join(out_dir, f"{s}_labels.npy"), L)
            print(f"  [fused] {s}: {E.shape}", flush=True)
    else:
        feats12 = _extract_backbone_embeddings(model_key, args.ckpt_path, cfg,
                                               pretrain_checkpoint=None,
                                               splits=("train", "val", "test"))
        for s, (E12, L12) in feats12.items():
            E, L = _apply_11cls_remap(E12, L12)
            np.save(_os.path.join(out_dir, f"{s}.npy"), E.astype(np.float32))
            np.save(_os.path.join(out_dir, f"{s}_labels.npy"), L)
            print(f"  [tile] {s}: {E.shape}", flush=True)

    print(f"[extract-sig] → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
