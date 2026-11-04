#!/usr/bin/env python3
"""Construit le manifeste de coordonnées et le split JSON du SPLIT CANONIQUE (35 orthos).

Entrées :
  - le manifeste existant (17 orthos : val + test intégralement couverts)
  - les coordonnées rejouées pour les 15 orthos de train (replay_coords.py)
  - splits/{train,val,test}_11cls.csv (définition canonique du split)

Sorties :
  - coords_manifest_35.json  (même schéma que out/context/coords_manifest.json)
  - canonical_split.json     (tiles[] avec class/label/ortho/split/row/col)

Échoue explicitement si une seule tuile du split n'a pas de coordonnées.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                  "RUBC", "SEDG", "TUSS", "WILL"]


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = str(Path(__file__).resolve().parents[3])
    ap.add_argument("--manifest17", default=os.path.join(repo, "out/context/coords_manifest.json"))
    ap.add_argument("--train-coords", default=os.path.join(base, "out/coords/train_coords.json"))
    ap.add_argument("--splits-dir", default=os.path.join(repo, "splits"))
    ap.add_argument("--out-manifest", default=os.path.join(base, "out/coords/coords_manifest_35.json"))
    ap.add_argument("--out-split", default=os.path.join(base, "out/canonical_split.json"))
    a = ap.parse_args()

    man17 = json.load(open(a.manifest17))["entries"]
    train_coords = json.load(open(a.train_coords))
    print(f"manifeste 17 orthos : {len(man17)} entrées")
    print(f"coords train rejouées : {sum(len(v) for v in train_coords.values())} tuiles "
          f"sur {len(train_coords)} orthos")

    entries: dict[str, dict] = {}
    tiles: list[dict] = []
    missing: list[str] = []
    per_split = collections.Counter()
    per_class = collections.defaultdict(collections.Counter)

    for split in ("train", "val", "test"):
        path = os.path.join(a.splits_dir, f"{split}_11cls.csv")
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            key = r["filepath"]
            parts = key.split("/")
            ortho, klass = parts[1], parts[2]
            idx = int(re.search(r"tile_(\d+)", parts[-1]).group(1))
            if key in man17:
                row_off = man17[key]["tile_row_off"]
                col_off = man17[key]["tile_col_off"]
            elif str(idx) in train_coords.get(ortho, {}):
                row_off, col_off = train_coords[ortho][str(idx)]
            else:
                missing.append(key)
                continue
            entries[key] = {"ortho": ortho, "tile_count": idx, "class": klass,
                            "tile_row_off": int(row_off), "tile_col_off": int(col_off)}
            tiles.append({"key": key, "ortho": ortho, "class": klass, "split": split,
                          "label": CLASS_NAMES_11.index(klass),
                          "tile_row_off": int(row_off), "tile_col_off": int(col_off)})
            per_split[split] += 1
            per_class[split][klass] += 1

    if missing:
        print(f"[ERREUR] {len(missing)} tuiles sans coordonnées (ex. {missing[:5]})", file=sys.stderr)
        sys.exit(2)

    # ordre trié par clé : identique à extract_features.py, exigé par run_arms.py
    tiles.sort(key=lambda t: t["key"])
    json.dump({"entries": entries}, open(a.out_manifest, "w"))
    report = {s: {"n": per_split[s], "class_counts": dict(sorted(per_class[s].items()))}
              for s in ("train", "val", "test")}
    json.dump({"created_utc": None, "n_tiles": len(tiles),
               "protocol": "split canonique 11 classes (splits/*_11cls.csv), "
                           "coordonnées = manifeste val/test + rejeu train",
               "source_manifest_sha256": sha256_file(a.out_manifest),
               "report": report, "tiles": tiles},
              open(a.out_split, "w"))
    print(json.dumps(report, indent=1))
    print(f"écrit {a.out_manifest} et {a.out_split}")


if __name__ == "__main__":
    main()
