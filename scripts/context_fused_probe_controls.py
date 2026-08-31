#!/usr/bin/env python3
"""Contrôles de la fusion tuile⊕contexte (Design B) — modèles SANS distillation.

BUT : attribuer le gain de R2 (f1_macro_pres_test ≈ 0.508, features fusionnées
1536 dims) à la bonne cause. R2 compare un backbone ENTRAÎNÉ par distillation
(contexte→tuile) sur des features fusionnées — mais on ne sait pas encore si le
gain vient (a) du contexte ajouté dans la sonde, (b) de l'entraînement, ou
(c) des 1536 dims + de la régularisation C choisie.

Ce script calcule la MÊME sonde canonique que context_distill.py (mêmes zips,
mêmes splits spatiaux frac100_seed{0,1,2}, même grille C ∈ {1e-4..10}, même
métrique f1_macro_pres) sur DEUX contrôles :

  1. DINOv3-B GELÉ  (--frozen)  : fusion pure, AUCUN entraînement.
  2. LoRA spatial de base r8a16 (--ckpt-path) : entraîné SANS contexte
     (baseline spatiale canonique ≈ 0.4827, results/spatial_datacurve_CANONICAL.csv).

Pour chaque contrôle, on sonde les DEUX représentations :
  - ``tile``  (768)  → doit reproduire le chiffre canonique (sanity check),
  - ``fused`` (1536) → exactement l'entrée de R2.

ATTRIBUTION (hypothèses) :
  - frozen/fused ≈ 0.508 → le gain de R2 vient du CONTEXTE DANS LA SONDE, la
    distillation n'apporte rien de plus (le 0.508 n'est pas un score de modèle
    entraîné, c'est une borne « info contexte »).
  - frozen/fused ≈ 0.487 et lora_base/fused ≈ 0.508 → c'est l'entraînement
    (LoRA ou distillation) qui débloque l'usage du contexte.

Sortie : <out_dir>/controls/fused_probe_{tag}_seed{seed}.json

Usage (via scripts/slurm_context_fused_probe_controls.sh, ne pas lancer à la main) :
    python scripts/context_fused_probe_controls.py \\
        --config configs/context_distill_dinov3b.yaml \\
        --context-dir $SLURM_TMPDIR/context_1024 \\
        --out-dir $SCRATCH/context_distill \\
        --seed 0 --tag frozen --frozen
    python scripts/context_fused_probe_controls.py \\
        --config configs/context_distill_dinov3b.yaml \\
        --context-dir $SLURM_TMPDIR/context_1024 \\
        --out-dir $SCRATCH/context_distill \\
        --seed 0 --tag lora_base_r8a16 \\
        --ckpt-path $SCRATCH/sota_screening/lora_spatial_v2/frac100/checkpoints/\\
            dinov3_vitb16_lvd_lora_r8a16_frac100_seed0_best.pth
"""
from __future__ import annotations

import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine dépôt
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # scripts/ (sibling import)

from src.config import load_config
from context_distill import (_read_split_12cls_filtered_remapped,
                             _run_probe_with_balanced_acc)


def _extract(model, forward_fn, cfg, split, context_dir, fused,
             batch_size: int, num_workers: int):
    """Extraction tuile seule (fused=False) ou tuile⊕contexte (fused=True), split donné.

    Mêmes transform eval déterministes et même appariement (``fp`` identique pour
    la tuile et le contexte) que ``_make_eval_loader_with_context`` — aucun écart
    de protocole par rapport aux runs.
    """
    import torch
    from PIL import Image
    from src.data import build_transforms
    from src.utils import get_normalization, get_device

    mean, std = get_normalization(cfg.model.norm)
    tf = build_transforms("eval", mean, std, cfg.data.image_size)
    fps, labels_11 = _read_split_12cls_filtered_remapped(
        _os.path.join(cfg.paths.csv_dir, f"{split}.csv"))
    device = get_device()
    model = model.to(device).eval()

    embs = []
    with torch.no_grad():
        for i in range(0, len(fps), batch_size):
            bfps = fps[i:i + batch_size]
            tiles = torch.stack([tf(Image.open(_os.path.join(cfg.paths.tiles_dir, fp)).convert("RGB"))
                                 for fp in bfps]).to(device)
            f_tile = forward_fn(model, tiles)
            if fused:
                ctxs = torch.stack([tf(Image.open(_os.path.join(context_dir, fp)).convert("RGB"))
                                    for fp in bfps]).to(device)
                f = torch.cat([f_tile, forward_fn(model, ctxs)], dim=1)
            else:
                f = f_tile
            embs.append(f.detach().cpu().to(torch.float16).numpy())
    return np.concatenate(embs, axis=0), np.asarray(labels_11)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--context-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", required=True, help="frozen | lora_base_r8a16 | ...")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frozen", action="store_true", help="DINOv3-B gelé, aucun ckpt")
    src.add_argument("--ckpt-path", default=None, help="checkpoint LoRA (baseline spatiale)")
    ap.add_argument("--reps", choices=("both", "tile", "fused"), default="both",
                    help="représentations à sonder (défaut both ; R2 en tuile seule : --reps tile)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_key = cfg.model.name

    if args.frozen:
        from src.models import build_frozen_extractor
        model, forward_fn, dim, norm_key = build_frozen_extractor(model_key, None)
        src_desc = "frozen (aucun entraînement)"
    else:
        if not _os.path.exists(args.ckpt_path):
            print(f"[controls] ERROR checkpoint introuvable : {args.ckpt_path}", flush=True)
            sys.exit(1)
        from src.models import load_finetuned_ssl_backbone
        model, forward_fn, dim, norm_key = load_finetuned_ssl_backbone(model_key, None, args.ckpt_path)
        src_desc = f"ckpt={args.ckpt_path}"

    print(f"[controls] tag={args.tag} seed={args.seed} modèle={src_desc} dim={dim}", flush=True)

    results = {"tag": args.tag, "seed": args.seed, "model": model_key,
               "src": src_desc, "schema": "11cls_no_rhol", "split": "spatial"}
    for rep, fused in (("tile", False), ("fused", True)):
        if args.reps == "tile" and fused:
            continue
        if args.reps == "fused" and not fused:
            continue
        feats = {}
        for s in ("train", "val", "test"):
            E, L = _extract(model, forward_fn, cfg, s, args.context_dir, fused,
                            cfg.data.batch_size, cfg.data.num_workers)
            feats[s] = (E, L)
            print(f"  [extract-{rep}] {s}: {E.shape}", flush=True)
        m = _run_probe_with_balanced_acc(
            {s: feats[s] for s in ("val", "test")}, feats["train"],
            list(cfg.probe.C_grid), cfg.probe.max_iter)
        results[rep] = m
        print(f"  [probe-{rep}] dim={E.shape[1]} f1_macro_pres_test={m['f1_macro_pres_test']:.4f} "
              f"bal_acc={m['balanced_accuracy_test']:.4f} best_C={m['best_C']}", flush=True)

    out_dir = _os.path.join(args.out_dir, "controls")
    _os.makedirs(out_dir, exist_ok=True)
    out_path = _os.path.join(out_dir, f"fused_probe_{args.tag}_seed{args.seed}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[controls] → {out_path}", flush=True)


if __name__ == "__main__":
    main()
