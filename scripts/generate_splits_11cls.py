#!/usr/bin/env python3
"""Génère des CSV de splits pré-remappés en 11 classes (RHOL exclue, labels 0-10).

Lit ``{splits_dir}/{train,val,test}.csv`` (schéma brut 12 classes, RHOL=``utils.RHOL_IDX``),
filtre les lignes RHOL et remappe les labels restants via ``utils.LABEL_REMAP_12TO11``.
Écrit ``{out_dir}/{train,val,test}{suffix}.csv`` (même header/format que la source).

Usage :
    python scripts/generate_splits_11cls.py
    python scripts/generate_splits_11cls.py --force   # écrase une sortie déjà existante

Lecture seule sur les CSV sources. N'écrase JAMAIS un fichier de sortie déjà présent
sans ``--force`` explicite (garde-fou anti-écrasement silencieux).
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from src import utils

SPLIT_NAMES = ("train", "val", "test")


def remap_split(in_path: str, out_path: str) -> tuple[int, int, list[int], list[int]]:
    """Filtre RHOL + remappe labels 12->11 depuis ``in_path`` vers ``out_path``.

    Retourne ``(n_avant, n_apres, labels_avant, labels_apres)``.
    """
    rows_out: list[tuple[str, int]] = []
    labels_before: list[int] = []
    with open(in_path) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 2:
                continue
            fp, lb = row[0], int(row[1])
            labels_before.append(lb)
            if lb == utils.RHOL_IDX:
                continue
            rows_out.append((fp, utils.LABEL_REMAP_12TO11[lb]))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    labels_after = [lb for _, lb in rows_out]
    return len(labels_before), len(labels_after), labels_before, labels_after


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", default="splits",
                    help="dossier contenant train/val/test.csv (défaut: splits)")
    ap.add_argument("--out-dir", default=None,
                    help="dossier de sortie (défaut: identique à --splits-dir)")
    ap.add_argument("--suffix", default="_11cls",
                    help="suffixe des fichiers de sortie (défaut: _11cls)")
    ap.add_argument("--force", action="store_true",
                    help="autorise l'écrasement d'un fichier de sortie déjà existant")
    args = ap.parse_args()

    out_dir = args.out_dir or args.splits_dir

    print(f"[generate_splits_11cls] RHOL_IDX={utils.RHOL_IDX}")
    print(f"[generate_splits_11cls] LABEL_REMAP_12TO11={utils.LABEL_REMAP_12TO11}")

    for split in SPLIT_NAMES:
        in_path = os.path.join(args.splits_dir, f"{split}.csv")
        out_path = os.path.join(out_dir, f"{split}{args.suffix}.csv")
        if not os.path.exists(in_path):
            print(f"\n[generate_splits_11cls] SKIP {split}: {in_path} introuvable")
            continue
        if os.path.exists(out_path) and not args.force:
            raise FileExistsError(
                f"{out_path} existe déjà — relancer avec --force pour écraser "
                f"(garde-fou anti-écrasement silencieux)."
            )

        n_before, n_after, labels_before, labels_after = remap_split(in_path, out_path)
        uniq_before = sorted(int(x) for x in np.unique(labels_before))
        uniq_after = sorted(int(x) for x in np.unique(labels_after)) if labels_after else []
        over10 = [l for l in uniq_after if l > 10]

        print(f"\n[generate_splits_11cls] {split}:")
        print(f"  lignes avant filtrage RHOL : {n_before}")
        print(f"  lignes après filtrage RHOL : {n_after} (retiré {n_before - n_after})")
        print(f"  labels uniques avant : {uniq_before}")
        print(f"  labels uniques après : {uniq_after}")
        if over10:
            print(f"  !! ALERTE labels > 10 subsistants : {over10}")
        else:
            print("  OK : aucun label > 10 après remap")
        print(f"  écrit -> {out_path}")


if __name__ == "__main__":
    main()
