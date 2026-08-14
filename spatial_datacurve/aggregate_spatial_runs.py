#!/usr/bin/env python3
"""Agrège les metrics.json des runs LoRA SPATIAUX v2 (Narval) en une table.

Après les 21 runs (sbatch spatial_datacurve/slurm_datacurve_spatial_v2.sh),
exécuter sur Narval :

    python spatial_datacurve/aggregate_spatial_runs.py \
        --out-root "$SCRATCH/sota_screening/lora_spatial_v2" \
        --manifest spatial_datacurve/manifest.json

Sortie : {out_root}/results_spatial_summary.csv  (+ .md)

Colonnes :
  fraction_cible          — cible du niveau spatial (0.01 … 1.00)
  spatial_fraction_reelle — fraction SPATIALE réelle (tuiles brutes du split /
                            49433, cf. manifest) — l'abscisse de la courbe
  seed                    — seed du run (0/1/2)
  n_train_tiles_11cls     — volume réel d'entraînement APRÈS filtre RHOL
                            (schéma 11 classes) — metrics.json:n_train_tiles
  f1_macro_pres_test      — F1 macro test, classes présentes dans y_true test
  f1_macro_8cls_test      — F1 macro test, schéma 8 classes
  f1_macro_train_pres     — F1 macro test RESTREINT aux classes PRÉSENTES dans
                            le train (composition du split spatial) : effet
                            VOLUME pur, sans la pénalité des classes jamais
                            vues. Écart f1_macro_pres_test − train_pres = coût
                            de couverture d'habitats.
  best_epoch / best_C     — epoch et régularisation sélectionnés

NB : tous les tags de run internes portent `frac100` (artefact du
--fraction 1.0) ; le niveau spatial est encodé dans le NOM DU DOSSIER parent
(frac001 … frac100) sous --out-root, et c'est lui qui est joint au manifest.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np


def train_present_f1(metrics: dict, split_manifest: dict) -> float:
    """F1 macro test restreint aux classes présentes dans le train.

    ``metrics['f1_per_class']`` = {nom 11-class: F1 test} ; la présence au
    train vient du manifest du split (``classes_11cls``, comptes > 0).
    """
    f1_pc = metrics.get("f1_per_class_test") or metrics.get("f1_per_class") or {}
    present = [c for c, n in split_manifest.get("classes_11cls", {}).items()
               if n > 0 and c in f1_pc]
    if not present:
        return float("nan")
    return float(np.mean([f1_pc[c] for c in present]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True,
                    help="$SCRATCH/sota_screening/lora_spatial_v2")
    ap.add_argument("--manifest", default="spatial_datacurve/manifest.json")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    # tag (frac001…) → fraction cible + fraction spatiale réelle par seed
    level_info = {}
    for nv in manifest["niveaux"]:
        for s in nv["seeds"]:
            level_info[(nv["tag"], s["seed"])] = {
                "cible": nv["cible"],
                "fraction_reelle": s["fraction_reelle"],
                "n_tiles_brutes": s["n_tiles"],
            }

    rows = []
    missing = []
    splits_dir = os.path.join(os.path.dirname(os.path.abspath(args.manifest)),
                              "splits")
    for frac_dir in sorted(glob.glob(os.path.join(args.out_root, "frac*"))):
        tag = os.path.basename(frac_dir)
        for mf in sorted(glob.glob(os.path.join(frac_dir, "runs", "*", "metrics.json"))):
            with open(mf) as f:
                m = json.load(f)
            seed = m.get("seed")
            key = (tag, seed)
            if key not in level_info:
                missing.append((tag, seed))
                continue
            split_manifest = {}
            sm_path = os.path.join(splits_dir, f"{tag}_seed{seed}", "manifest.json")
            if os.path.exists(sm_path):
                with open(sm_path) as f:
                    split_manifest = json.load(f)
            rows.append({
                "fraction_cible": level_info[key]["cible"],
                "spatial_fraction_reelle": level_info[key]["fraction_reelle"],
                "seed": seed,
                "run_dir": os.path.relpath(os.path.dirname(mf), args.out_root),
                "n_train_tiles_11cls": m.get("n_train_tiles"),
                "f1_macro_pres_test": m.get("f1_macro_pres_test"),
                "f1_macro_8cls_test": m.get("f1_macro_8cls_test"),
                "f1_macro_train_pres": round(train_present_f1(m, split_manifest), 4),
                "best_epoch": m.get("best_epoch"),
                "best_C": m.get("best_C"),
            })

    rows.sort(key=lambda r: (r["fraction_cible"], r["seed"]))
    out_csv = os.path.join(args.out_root, "results_spatial_summary.csv")
    out_md = os.path.join(args.out_root, "results_spatial_summary.md")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out_md, "w") as f:
        f.write("# Courbe de données spatiale v2 — résultats agrégés\n\n")
        f.write(f"Runs trouvés : **{len(rows)}/21**\n\n")
        f.write("| Cible | Frac. spatiale | Seed | Tuiles (11cls) | "
                "F1 pres test | F1 8cls test | F1 train-pres | best_epoch | best_C |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['fraction_cible']:.0%} | "
                    f"{r['spatial_fraction_reelle']:.1%} | {r['seed']} | "
                    f"{r['n_train_tiles_11cls']} | "
                    f"{r['f1_macro_pres_test']:.4f} | "
                    f"{r['f1_macro_8cls_test']:.4f} | "
                    f"{r['f1_macro_train_pres']:.4f} | {r['best_epoch']} | "
                    f"{r['best_C']} |\n")

    print(f"OK : {len(rows)} runs agrégés → {out_csv}")
    if missing:
        print(f"ATTENTION : {len(missing)} runs sans correspondance manifest : {missing}")
    if len(rows) < 21:
        print(f"NOTE : {21 - len(rows)} runs manquants — relancer ou vérifier "
              f"--skip-if-done / sentinelles done.")


if __name__ == "__main__":
    main()
