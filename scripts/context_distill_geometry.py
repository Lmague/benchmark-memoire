#!/usr/bin/env python3
"""Géométrie de l'espace latent des modèles context_distill (R1/R2/R3, ctx1024).

Recharge chaque checkpoint ``*_best.pth`` produit par ``scripts/context_distill.py``
et calcule les métriques de géométrie CANONIQUES du dépôt — RankMe, anisotropie,
α-ReQ, NESum — via ``src.latent.metrics_with_spectrum`` (le MÊME pipeline que
``analyze.py`` : split test, subsample 20 000, seed 42).

Représentations évaluées :
  - Design A : features tuile seule (768 dims) — la représentation déployable.
  - Design B : features fusionnées tuile⊕contexte (1536 dims) — la représentation
    sondée dans les metrics.json — ET la branche tuile seule (768 dims) du MÊME
    checkpoint, pour isoler l'apport du contexte (même backbone, seule la concat
    change ; les deux embeddings viennent du même ``*_best.pth``).

Ne ré-entraîne RIEN, lecture seule des runs existants. Le forward est fait sur le
device actif (GPU sur Narval via le wrapper SLURM).

Sorties (par seed) :
  ``<out_dir>/geometry/geometry_seed{S}_split{T}.json``
  ``<out_dir>/geometry/embeddings/<tag>_{tile|fused|tile_only}_{split}.npy`` (+ labels)
      si ``--save-embeddings`` — petits fichiers à rapatrier en local pour toute
      analyse ultérieure (les embeddings n'étaient jamais persistés par les runs).

Usage (via ``scripts/slurm_context_distill_geometry.sh``, ne pas lancer à la main) :
    python scripts/context_distill_geometry.py \\
        --config configs/context_distill_dinov3b.yaml \\
        --context-dir $SLURM_TMPDIR/context_1024 \\
        --out-dir $SCRATCH/context_distill \\
        --seed 0 --split test --save-embeddings
"""
from __future__ import annotations

# --- Mono-thread BLAS AVANT numpy/sklearn (AGENTS.md §4.8) -----------------------
# Les métriques elles-mêmes sont des SVD numpy (pas de solveur sensible aux threads),
# mais l'extraction passe par src.data/ArcticTVCDataset (PIL) et le standard du dépôt
# est de figer les threads — sans effet notable sur le forward GPU de toute façon.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine dépôt
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # scripts/ (sibling import)

from src.config import load_config
from src.latent import metrics_with_spectrum
from datacurve_one_run import _extract_backbone_embeddings, _apply_11cls_remap
from context_distill import _extract_fused_embeddings

DEFAULT_SUBSAMPLE = 20000
DEFAULT_N_PAIRS = 10000


def _tag_to_design(tag: str) -> str:
    """Design depuis le tag : '_dB_' → B, sinon A (convention de nommage du dépôt)."""
    return "B" if "_dB_" in tag else "A"


def _geometry_for(tag: str, ckpt_path: str, cfg, split: str, context_dir,
                  subsample_n: int, n_pairs: int, model_key: str):
    """Extrait + calcule la géométrie d'UN checkpoint. Retourne (result dict, embs dict)."""
    design = _tag_to_design(tag)
    res: dict = {"tag": tag, "design": design, "ckpt": ckpt_path, "split": split}
    embs: dict = {}

    if design == "B":
        # Représentation sondée (fused, déjà 11cls) — le chiffre comparable aux metrics.json.
        feats = _extract_fused_embeddings(model_key, ckpt_path, cfg, context_dir,
                                          splits=(split,))
        E, L = feats[split]
        res["fused"] = metrics_with_spectrum(E, E, n_pairs=n_pairs, subsample_n=subsample_n)
        embs["fused"] = (E, L)

        # Branche tuile seule du MÊME checkpoint — isole l'apport du contexte.
        feats12 = _extract_backbone_embeddings(model_key, ckpt_path, cfg,
                                               pretrain_checkpoint=None, splits=(split,))
        E12, L12 = feats12[split]
        E_tile, L_tile = _apply_11cls_remap(E12, L12)
        res["tile_only"] = metrics_with_spectrum(E_tile, E_tile, n_pairs=n_pairs,
                                                 subsample_n=subsample_n)
        embs["tile_only"] = (E_tile, L_tile)
        return res, embs

    # Design A : tuile seule (12cls brut → remap 11cls, comme les runs).
    feats12 = _extract_backbone_embeddings(model_key, ckpt_path, cfg,
                                           pretrain_checkpoint=None, splits=(split,))
    E12, L12 = feats12[split]
    E, L = _apply_11cls_remap(E12, L12)
    res["tile"] = metrics_with_spectrum(E, E, n_pairs=n_pairs, subsample_n=subsample_n)
    embs["tile"] = (E, L)
    return res, embs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--context-dir", required=True,
                    help="dossier context_<size> dézippé (requis Design B ; inutilisé Design A)")
    ap.add_argument("--out-dir", required=True,
                    help="dossier $SCRATCH/context_distill (contient checkpoints/ et runs/)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", default="test", help="split de géométrie (défaut test = canonique)")
    ap.add_argument("--subsample", type=int, default=DEFAULT_SUBSAMPLE)
    ap.add_argument("--n-pairs", type=int, default=DEFAULT_N_PAIRS)
    ap.add_argument("--save-embeddings", action="store_true",
                    help="persiste les .npy (fp16) + labels sous <out_dir>/geometry/embeddings/")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_key = cfg.model.name

    ckpt_dir = _os.path.join(args.out_dir, "checkpoints")
    pattern = _os.path.join(ckpt_dir, f"*_ctx*_r2a4_frac100_seed{args.seed}_best.pth")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        print(f"[geometry] aucun checkpoint pour seed={args.seed} : {pattern}", flush=True)
        sys.exit(1)
    print(f"[geometry] {len(ckpts)} checkpoint(s) seed={args.seed} split={args.split}", flush=True)

    results: dict = {}
    all_embs: dict = {}
    for ckpt in ckpts:
        tag = _os.path.basename(ckpt).removesuffix("_best.pth")
        print(f"\n[geometry] {tag}", flush=True)
        res, embs = _geometry_for(tag, ckpt, cfg, args.split, args.context_dir,
                                  args.subsample, args.n_pairs, model_key)
        results[tag] = res
        all_embs[tag] = embs
        for k, v in res.items():
            if isinstance(v, dict) and "rankme" in v:
                print(f"  {k:10s} dim={v['dim']:<5} RankMe={v['rankme']:>8.2f} "
                      f"Aniso={v['anisotropy']:+.4f} α-ReQ={v.get('alpha_req', float('nan')):.4f} "
                      f"NESum={v.get('nesum', float('nan')):.4f}", flush=True)

    out_dir = _os.path.join(args.out_dir, "geometry")
    _os.makedirs(out_dir, exist_ok=True)
    out_json = _os.path.join(out_dir, f"geometry_seed{args.seed}_split{args.split}.json")
    with open(out_json, "w") as f:
        json.dump({"split": args.split, "subsample_n": args.subsample,
                   "n_pairs": args.n_pairs, "models": results}, f, indent=2)
    print(f"\n[geometry] → {out_json}", flush=True)

    if args.save_embeddings:
        emb_dir = _os.path.join(out_dir, "embeddings")
        _os.makedirs(emb_dir, exist_ok=True)
        for tag, embs in all_embs.items():
            for k, (E, L) in embs.items():
                np.save(_os.path.join(emb_dir, f"{tag}_{k}_{args.split}.npy"),
                        E.astype(np.float16))
                np.save(_os.path.join(emb_dir, f"{tag}_{k}_{args.split}_labels.npy"), L)
        print(f"[geometry] embeddings → {emb_dir}", flush=True)


if __name__ == "__main__":
    main()
