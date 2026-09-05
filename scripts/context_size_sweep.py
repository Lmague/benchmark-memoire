#!/usr/bin/env python3
"""Sweep de la TAILLE du contexte (512/1024/2048) — DINOv3-B GELÉ, fusion sonde.

Demande de M. Bouguessa (mail ~sept. 2026) : « regarder les différentes tailles de
contexte (512, 1024, 2048) avant de passer à des modèles DINOv3 plus grands ».
Premier pallier NON ENTRAÎNÉ (cf. results/context_distill/ANALYSE.md §7) : pour
chaque taille, features FUSIONNÉES gelées [tile ; ctx] (2×768), puis sonde canonique
sur trois variantes réutilisant scripts/context_bouguessa_controls.py :

  - ``fused``          : la courbe d'apport d'information vs échelle de contexte ;
  - ``ctx``            : contexte SEUL (ce que chaque échelle porte à elle seule) ;
  - ``fused_ctxperm{p}`` : contexte réapparié au hasard — l'effet d'ensemble de la
    concaténation, soustrait du gain pour isoler l'information spatiale associée.

Extraction : ``_extract`` de ``context_fused_probe_controls.py`` (mêmes transforms
eval déterministes, même appariement tuile/contexte que les runs) sur DINOv3-B gelé
(``build_frozen_extractor``). Sauvegarde : ``sig_embeddings/<tag>/{split}.npy`` (même
layout que le bootstrap), float32 — repartable (skip si présent).

Sorties :
  - ``$SCRATCH/context_distill/sig_embeddings/dinov3_vitb16_lvd_FROZEN_fused_ctx<size>_frac100_seed0/``
  - ``$SCRATCH/context_distill/controls_bouguessa/frozen_ctx<size>_seed0_{variant}.json``

Usage (via scripts/slurm_context_size_sweep.sh, ne pas lancer à la main) :
    python scripts/context_size_sweep.py --config configs/context_distill_dinov3b.yaml \\
        --context-dir $SLURM_TMPDIR/context_512 --context-size 512 --out-dir $SCRATCH/context_distill
"""
from __future__ import annotations

# --- Mono-thread BLAS AVANT numpy/sklearn (AGENTS.md §4.8) -----------------------
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # scripts/ (sibling import)

from context_bouguessa_controls import C_GRID, _build_variant
from context_distill import _run_probe_with_balanced_acc
from context_fused_probe_controls import _extract  # extraction tuile⊕contexte, protocole identique aux runs


def _sig_tag(model_tag: str, context_size: int) -> str:
    """Tag du dossier sig_embeddings pour (modèle, taille). ``model_tag`` est un
    suffixe court propre au backbone (ex. ``dinov3_vits16``, ``simdinov2_vitb16``),
    PAS le nom complet — le nom complet peut contenir des suffixes FT.
    """
    return f"{model_tag}_FROZEN_fused_ctx{context_size}_frac100_seed0"


def _extract_and_save(cfg, context_dir: str, out_dir: str, context_size: int,
                      model_tag: str, checkpoint: str | None,
                      batch_size: int, num_workers: int) -> str:
    """Extraction frozen fusionnée des 3 splits → layout sig_embeddings. Retourne le tag."""
    from src.models import build_frozen_extractor

    tag = _sig_tag(model_tag, context_size)
    d = _os.path.join(out_dir, "sig_embeddings", tag)
    _os.makedirs(d, exist_ok=True)
    if all(_os.path.exists(_os.path.join(d, f"{s}.npy")) for s in ("train", "val", "test")):
        print(f"[sweep] sig_embeddings déjà présents : {d} — extraction sautée", flush=True)
        return tag

    model, forward_fn, _dim, _norm_key = build_frozen_extractor(cfg.model.name, checkpoint)
    for s in ("train", "val", "test"):
        t0 = time.time()
        E, L = _extract(model, forward_fn, cfg, s, context_dir, fused=True,
                        batch_size=batch_size, num_workers=num_workers)
        np.save(_os.path.join(d, f"{s}.npy"), E.astype(np.float32))
        np.save(_os.path.join(d, f"{s}_labels.npy"), L)
        print(f"[sweep ctx{context_size}] extract fused {s}: {E.shape} ({time.time()-t0:.0f}s)", flush=True)
    return tag


def _probe_task(args_tuple):
    (variant, sig_dir, tag, out_dir, max_iter, context_size, model_tag) = args_tuple
    out_path = _os.path.join(out_dir, "controls_bouguessa",
                             f"frozen_{model_tag}_ctx{context_size}_seed0_{variant}.json")
    if _os.path.exists(out_path):
        return out_path, "skip (déjà fait)"
    t0 = time.time()
    feats = {}
    for s in ("train", "val", "test"):
        feats[s] = (np.load(_os.path.join(sig_dir, tag, f"{s}.npy")),
                    np.load(_os.path.join(sig_dir, tag, f"{s}_labels.npy")))
    var = _build_variant(feats, variant)
    del feats
    d = var["train"][0].shape[1] // 2
    metrics = _run_probe_with_balanced_acc(
        {"val": var["val"], "test": var["test"]}, var["train"], C_GRID, max_iter)
    del var
    result = {
        "tag": f"frozen_{model_tag}_ctx{context_size}",
        "seed": 0,
        "variant": variant,
        "perm_seed": 1000 + int(variant.rsplit("perm", 1)[1])
        if variant.startswith("fused_ctxperm") else None,
        "dim": 2 * d if (variant == "fused" or variant.startswith("fused_ctxperm")) else d,
        "model": f"{model_tag} FROZEN (aucun entraînement), contexte redimensionné 224",
        "src": tag,
        "schema": "11cls_no_rhol",
        "split": "spatial_v3",
        "protocol": "StandardScaler float32 + make_canonical_lr lbfgs multinomial, "
                    f"C grille {C_GRID}, sélection val f1_macro_pres, refit best_C ; "
                    "BLAS mono-thread ; parallélisation par processus",
        "runtime_s": round(time.time() - t0, 1),
        **metrics,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path, f"f1_test={metrics['f1_macro_pres_test']:.4f} ({result['runtime_s']}s)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--context-dir", required=True)
    ap.add_argument("--context-size", type=int, required=True, help="512|1024|2048 (pour le nommage)")
    ap.add_argument("--model-tag", required=True,
                    help="suffixe court du backbone (ex. dinov3_vits16, dinov3_vitl16, "
                         "simdinov2_vitb16, simdinov2_vitl16) — préfixe des dossiers/tags")
    ap.add_argument("--checkpoint", default=None,
                    help="chemin absolu du .pth pour les backbones qui l'exigent (SimDINOv2 : "
                         "$SCRATCH/checkpoints/simdinov2_*.pth). None pour DINOv3 (HF).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--perm-reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--variants", default=None)
    ap.add_argument("--skip-probes", action="store_true",
                    help="extraction seule (les probes sont ensuite lancés en JOB CPU "
                         "séparé via scripts/context_bouguessa_controls.py — cf. "
                         "slurm_context_size_sweep_probes.sh)")
    args = ap.parse_args()

    from src.config import load_config
    cfg = load_config(args.config)

    tag = _extract_and_save(cfg, args.context_dir, args.out_dir, args.context_size,
                            args.model_tag, args.checkpoint,
                            cfg.data.batch_size, cfg.data.num_workers)
    if args.skip_probes:
        print(f"[sweep {args.model_tag} ctx{args.context_size}] --skip-probes : "
              f"extraction seule terminée ({tag}). Probes = job CPU séparé "
              f"(slurm_context_size_sweep_probes.sh).", flush=True)
        return
    sig_dir = _os.path.join(args.out_dir, "sig_embeddings")

    variants = ([v.strip() for v in args.variants.split(",")] if args.variants
                else ["fused", "ctx"] + [f"fused_ctxperm{p}" for p in range(args.perm_reps)])
    tasks = [(v, sig_dir, tag, args.out_dir, args.max_iter, args.context_size, args.model_tag)
             for v in variants]
    print(f"[sweep {args.model_tag} ctx{args.context_size}] {len(tasks)} probes : {variants} "
          f"(workers={args.workers})", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_probe_task, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            variant = futs[fut]
            try:
                _path, msg = fut.result()
                print(f"  [sweep ctx{args.context_size}] {variant} — {msg}", flush=True)
            except Exception as e:  # noqa: BLE001 — journaliser et continuer (repartable)
                print(f"  [ERREUR ctx{args.context_size}] {variant} : {type(e).__name__}: {e}", flush=True)
    print(f"[sweep ctx{args.context_size}] terminé.", flush=True)


if __name__ == "__main__":
    main()
