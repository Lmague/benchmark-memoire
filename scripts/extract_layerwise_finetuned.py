#!/usr/bin/env python3
"""Extraction layerwise (CLS token par bloc transformer) depuis un checkpoint FINE-TUNÉ
(ExPLoRA), pour comparaison directe avec l'extraction layerwise frozen déjà faite.

Réutilise src.models.load_finetuned_ssl_backbone (merge LoRA + injection des poids
fine-tunés dans l'architecture frozen standard) — même chemin de chargement déjà validé
par datacurve_one_run.py ("211/211 clés chargées" dans les logs d'entraînement ExPLoRA).

Sauvegarde au format attendu par src.features.load_layerwise / count_layerwise_layers :
  {emb_dir}/{out_key}_{split}_layer{idx:02d}.npy   (fp16, CLS token par bloc)
  {emb_dir}/{out_key}_layerwise_{split}_labels.npy

Usage :
  python scripts/extract_layerwise_finetuned.py \\
      --config configs/frozen_dinov3_lvd.yaml \\
      --ft-checkpoint /scratch/lmague/sota_screening/dinov3_vitb16_lvd_explora/checkpoints/dinov3_vitb16_lvd_explora_frac100_seed0_best.pth \\
      --out-key dinov3_vitb16_lvd_explora_seed0 \\
      --splits train val test
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.models import get_transformer_blocks, load_finetuned_ssl_backbone
from src.utils import get_device, get_normalization

SPLITS_ALL = ("train", "val", "test")


def _eval_loader(cfg, split, mean, std):
    # Réutilise la même fonction que src/features.py pour rester bit-compatible
    # (ordre des tuiles, batch_size, resize/crop identiques à l'extraction frozen).
    from src.features import _eval_loader as _base_loader
    return _base_loader(cfg, split, mean, std)


def extract_layerwise_finetuned(cfg, model_key: str, ft_checkpoint: str, out_key: str,
                                splits=SPLITS_ALL) -> int:
    emb_dir = cfg.paths.emb_dir
    os.makedirs(emb_dir, exist_ok=True)

    pretrain_ckpt = cfg.raw.get("checkpoint")  # None pour DINOv3 (poids HF standard)
    model, forward_fn, dim, norm_key = load_finetuned_ssl_backbone(
        model_key, pretrain_ckpt, ft_checkpoint)
    device = get_device()
    model = model.to(device).eval()
    blocks = get_transformer_blocks(model)
    sel = list(range(len(blocks)))
    print(f"[layerwise-ft] {out_key}: {len(blocks)} blocs, dim={dim}, "
          f"checkpoint={ft_checkpoint}")

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in sel:
        def _hook(_m, _inp, out, li=li):
            o = out[0] if isinstance(out, tuple) else out
            captured[li] = o[:, 0, :].detach().cpu().to(torch.float16)  # token CLS

        handles.append(blocks[li].register_forward_hook(_hook))

    mean, std = get_normalization(norm_key)
    try:
        for s in splits:
            loader = _eval_loader(cfg, s, mean, std)
            per_layer = {li: [] for li in sel}
            lbls = []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device, non_blocking=True)
                    _ = forward_fn(model, x)
                    for li in sel:
                        per_layer[li].append(captured[li])
                    lbls.append(y)
            L = torch.cat(lbls).numpy().astype(np.int64)
            np.save(os.path.join(emb_dir, f"{out_key}_layerwise_{s}_labels.npy"), L)
            for li in sel:
                E = torch.cat(per_layer[li]).numpy().astype(np.float16)
                np.save(os.path.join(emb_dir, f"{out_key}_{s}_layer{li:02d}.npy"), E)
            print(f"[layerwise-ft] {out_key}/{s}: {len(sel)} couches × {L.shape[0]} ex sauvées.")
    finally:
        for h in handles:
            h.remove()

    return len(sel)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML frozen_* (architecture de base)")
    ap.add_argument("--ft-checkpoint", required=True, help="chemin .pth fine-tuné (ExPLoRA)")
    ap.add_argument("--out-key", required=True,
                    help="préfixe de sortie (ex: dinov3_vitb16_lvd_explora_seed0)")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS_ALL), metavar="SPLIT")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_key = cfg.model.name

    if not os.path.exists(args.ft_checkpoint):
        raise FileNotFoundError(f"checkpoint introuvable : {args.ft_checkpoint}")

    n = extract_layerwise_finetuned(cfg, model_key, args.ft_checkpoint, args.out_key,
                                    splits=tuple(args.splits))
    print(f"[done] {n} couches extraites pour {args.out_key} → {cfg.paths.emb_dir}")


if __name__ == "__main__":
    main()
