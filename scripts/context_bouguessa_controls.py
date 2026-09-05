#!/usr/bin/env python3
"""Contrôles demandés par M. Bouguessa (mail ~sept. 2026) sur le modèle à contexte (R2, Design B).

Deux expériences, calculées par sonde canonique sur les embeddings DÉJÀ extraits
(``sig_embeddings/dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100_seed{0,1,2}``,
fusionné 1536 = ``[feat_tuile ; feat_contexte]`` — ordre vérifié dans
``context_distill._extract_fused_embeddings`` : ``torch.cat([f_tile, f_ctx], dim=1)``) :

1. **Contexte SEUL** (``ctx`` : colonnes 768:1536) — ce que le contexte apporte à lui
   seul, sans l'embedding de la tuile centrale.
2. **Contexte PERMUTÉ** (``fused_ctxperm{p}`` : fusionné 1536 où le bloc contexte est
   réapparié au hasard, indépendamment dans chaque split) — vérifier que le gain de
   R2 (0.5080±0.0013) vient de l'information spatiale ASSOCIÉE à la tuile et non de
   la simple concaténation d'une seconde représentation (effet d'ensemble).

Deux variantes internes servent de témoins :
- ``fused`` (1536 tel quel) : doit reproduire les ``metrics.json`` (0.5098/0.5070/0.5071)
  — contrôle de validité de la séparation float16→splits ;
- ``tile`` (768, colonnes 0:768) : baseline tuile-seule par seed (le 0.4779 seed0 de la
  matrice d'attribution, plus les seeds 1-2 manquants).

Protocole : LA MÊME sonde canonique que les runs et les contrôles existants
(``context_distill._run_probe_with_balanced_acc`` : StandardScaler float32,
``make_canonical_lr``, grille C ∈ {1e-4..10}, sélection sur val f1_macro_pres, refit
best_C, split spatial v3). BLAS mono-thread (AGENTS.md §4.8) ; la parallélisation se
fait par PROCESSUS (``--workers``), jamais par threads BLAS.

Sortie : un JSON par (seed, variante) dans ``<out-dir>/`` — repartable (skip si déjà
présent, convention du dépôt). Agrégation/rapport : results/context_distill/CONTROLES_BOUGUESSA.md.

Usage :
    conda run -n arctic-tvc python scripts/context_bouguessa_controls.py --workers 2
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

from context_distill import _run_probe_with_balanced_acc  # sonde canonique, réutilisée telle quelle

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]  # configs/context_distill_dinov3b.yaml
MAX_ITER = 2000
PERM_SEED_BASE = 1000  # permutations p ∈ [0, perm_reps) → graine 1000+p (documenté dans le JSON)

RACINE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DEFAULT_SIG = _os.path.join(RACINE, "results", "context_distill", "sig_embeddings")
DEFAULT_OUT = _os.path.join(RACINE, "results", "context_distill", "controls_bouguessa")
DEFAULT_TAG = "dinov3_vitb16_lvd_ctxdistill_dB_tL_ctx1024_r2a4_frac100"


def _load_split(sig_dir: str, tag: str, seed: int, split: str):
    d = _os.path.join(sig_dir, f"{tag}_seed{seed}")
    E = np.load(_os.path.join(d, f"{split}.npy"))
    L = np.load(_os.path.join(d, f"{split}_labels.npy"))
    return E, L


def _build_variant(feats: dict, variant: str) -> dict:
    """Construit les (E, L) de la variante demandée depuis la fusion 1536.

    - fused          : tel quel (temoin de reproduction)
    - tile           : colonnes 0:768
    - ctx            : colonnes 768:1536  (Bouguessa #1 : contexte seul)
    - fused_ctxperm{p} : fusionné où les LIGNES du bloc contexte sont permutées
      indépendamment dans chaque split (graine PERM_SEED_BASE+p)  (Bouguessa #2)
    """
    if variant == "fused":
        return {s: feats[s] for s in feats}
    if variant == "tile":
        return {s: (feats[s][0][:, :768], feats[s][1]) for s in feats}
    if variant == "ctx":
        return {s: (feats[s][0][:, 768:], feats[s][1]) for s in feats}
    if variant.startswith("fused_ctxperm"):
        p = int(variant.rsplit("perm", 1)[1])
        out = {}
        for s, (E, L) in feats.items():
            rng = np.random.default_rng(PERM_SEED_BASE + p)
            idx = rng.permutation(E.shape[0])
            E2 = E.copy()
            E2[:, 768:] = E[idx, 768:]
            out[s] = (E2, L)
        return out
    raise ValueError(f"variante inconnue : {variant}")


def _run_one(args_tuple):
    (seed, variant, sig_dir, tag, out_dir, max_iter, json_prefix) = args_tuple
    out_dir = _os.path.join(out_dir, "controls_bouguessa")
    _os.makedirs(out_dir, exist_ok=True)
    out_path = _os.path.join(out_dir, f"{json_prefix}_seed{seed}_{variant}.json")
    if _os.path.exists(out_path):
        return out_path, "skip (déjà fait)"
    t0 = time.time()
    feats = {s: _load_split(sig_dir, tag, seed, s) for s in ("train", "val", "test")}
    var = _build_variant(feats, variant)
    del feats
    E_tr, L_tr = var["train"]
    metrics = _run_probe_with_balanced_acc(
        {"val": var["val"], "test": var["test"]}, (E_tr, L_tr), C_GRID, max_iter)
    del var, E_tr
    result = {
        "tag": "r2_dB_tL",
        "seed": seed,
        "variant": variant,
        "perm_seed": PERM_SEED_BASE + int(variant.rsplit("perm", 1)[1])
        if variant.startswith("fused_ctxperm") else None,
        "dim": metrics.get("dim", None) or _probe_dim(variant),
        "model": "dinov3_vitb16_lvd (LoRA r2a4 blocs 6-11, Design B, contexte 1024)",
        "src": f"{tag}_seed{seed} (sig_embeddings, fusion float16→float32)",
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


def _probe_dim(variant: str) -> int:
    if variant == "fused" or variant.startswith("fused_ctxperm"):
        return 1536
    return 768


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sig-dir", default=DEFAULT_SIG)
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variants", default=None,
                    help="liste csv parmi fused,tile,ctx,fused_ctxperm{p} (défaut : tout)")
    ap.add_argument("--perm-reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--json-prefix", default="r2_dB_tL",
                    help="préfixe des JSON de sortie (r2_dB_tL par défaut ; "
                         "le sweep des tailles utilise frozen_ctx<size>)")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
    else:
        variants = ["fused", "tile", "ctx"] + \
            [f"fused_ctxperm{p}" for p in range(args.perm_reps)]
    _os.makedirs(args.out_dir, exist_ok=True)

    tasks = [(seed, v, args.sig_dir, args.tag, args.out_dir, args.max_iter, args.json_prefix)
             for seed in seeds for v in variants]
    print(f"[controls-bouguessa] {len(tasks)} probes : seeds={seeds} × variantes={variants} "
          f"(workers={args.workers}, grille C={C_GRID}, max_iter={args.max_iter})", flush=True)

    done, skipped = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            seed, variant = futs[fut][0], futs[fut][1]
            try:
                path, msg = fut.result()
            except Exception as e:  # noqa: BLE001 — journaliser et continuer (repartable)
                print(f"  [ERREUR] seed{seed} {variant} : {type(e).__name__}: {e}", flush=True)
                continue
            if msg.startswith("skip"):
                skipped += 1
                print(f"  [skip] seed{seed} {variant} — {msg}", flush=True)
            else:
                done += 1
                print(f"  [ok] seed{seed} {variant} — {msg}", flush=True)
    print(f"[controls-bouguessa] terminé : {done} calculés, {skipped} déjà présents → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
