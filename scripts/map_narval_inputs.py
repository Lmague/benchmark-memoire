#!/usr/bin/env python3
"""MAP les inputs du tile-sweep depuis NARVAL MEME (la source des embeddings).

Usage :  ~/ENV/bin/python3 scripts/map_narval_inputs.py   (SUR Narval)
Résolution des ambigus : éditer $SCRATCH/head_sweep_forced.tsv (lignes '<groupe>|seed<n> TAB <dir>') puis relancer — idempotent.

Les runs (sota_screening, ft_ssl, lora_3models, Stage A, vits16, context_distill)
ont été produits sur $SCRATCH et copies sur le laptop. Narval EST la source :
ce script ne remonte RIEN, il construit sur place le miroir de chemins que
`tile_head_sweep.py --root` attend :

    $SCRATCH/head_sweep_inputs/<même chemin relatif que le dépôt>/<split>.npy
        -> lien symbolique vers l'original trouvé dans $SCRATCH

Validation par fichier candidat (le nom seul ne suffit pas — cf. les pièges
connus du dépôt : `vitb16_full_*` = ResNet-50 2048d, dossiers `_explora`
contenant du LoRA r=8, etc.) :
  - les 6 fichiers train/val/test + labels présents ;
  - n_test == 17 598 exact, n_train >= 40 000, n_val >= 10 000 ;
  - dim identique sur les 3 splits ET == dim attendue du groupe (table ci-dessous) ;
  - labels dans 0..10 (schéma 11 classes).

Un candidat qui passe TOUT est lié ; sinon le groupe va dans le rapport MANQUANT
(avec les pistes où chercher). Sortie : rapport lisible +
$SCRATCH/head_sweep_inputs_MISSING.txt + compte OK/33.

Usage (SUR NARVAL) :
    cd ~/benchmark-memoire && git pull
    ~/ENV/bin/python3 scripts/map_narval_inputs.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "rapport"))

SCRATCH = os.environ.get("SCRATCH") or "/scratch/lmague"
MIRROR = os.path.join(SCRATCH, "head_sweep_inputs")

from significance_tier import GROUPS  # noqa: E402

# dims attendues (tableau maître / t_master) — discriminant anti-piège de noms
DIM = {
    "DINOv3-B LoRA r=8": 768, "DINOv3 ViT-H+16": 1280, "DINOv3-B MHSA": 768,
    "DINOv3 ViT-L16": 1024, "DINOv3-B Full": 768, "SimDINOv2 ViT-L16": 1024,
    "SimDINOv2 ViT-B16": 768, "DINOv3 ViT-B16": 768, "ViT-B/16 IN-Full": 768,
    "SimDINOv2-B MHSA": 768, "ViT-B/16 IN-MHSA": 768, "SimDINOv2-B Full": 768,
    "DINOv3-L LoRA r=8": 1024, "SimDINOv2-L LoRA r=8": 1024,
    "SimDINOv2-B LoRA r=8": 768, "Contexte R2 (B, fusion)": 1536,
    "Contexte R1 (A, tuile)": 768, "Contexte R3 (A, EMA)": 768,
    "DINOv3 ViT-S/16 LoRA r=8": 384, "DINOv3 ViT-S/16": 384,
}
for g in GROUPS:
    DIM.setdefault(g[0], 768)   # tous les bras Stage A sont 768d

# groupes dont le basename local est trop générique pour être cherché tel quel
TAG_ALIAS = {
    "runs_vits16/embeddings/seed{s}": "dinov3_vits16_lvd_lora_r8a16_frac100_seed{s}",
}
# groupes cherchés par motif de chemin (le basename seul est ambigu)
PATH_PAT = {
    "ViTB_L_LoRA/lora_3models/simdinov2_vitb16/embeddings/frac100_seed{s}":
        "*/simdinov2_vitb16/embeddings/frac100_seed*",
}

N_TEST = 17598
FORCED_FILE = os.path.join(SCRATCH, "head_sweep_forced.tsv")


def load_forced():
    """Résolutions humaines : lignes '<groupe>|seed<n>\t<chemin dir original>'.
    Utilisé pour les groupes AMBIGU (plusieurs candidats valides — le piège
    connu `..._explora_frac100_seed*` qui contient du LoRA r=8, ou l'inverse,
    ne se tranche pas à la forme : que l'humain choisisse)."""
    forced = {}
    if os.path.exists(FORCED_FILE):
        for line in open(FORCED_FILE):
            if "\t" in line:
                k, v = line.rstrip("\n").split("\t", 1)
                forced[k] = v
    return forced


def index_scratch():
    """Un seul find : tous les répertoires-run (contenant test.npy) + tous les
    profils plats *_train.npy (frozen)."""
    out = subprocess.run(
        ["find", SCRATCH, "-maxdepth", "7",
         "(", "-name", "test.npy", "-o", "-name", "*_test.npy", ")",
         "-not", "-path", "*SLURM*", "-printf", "%p\n"],
        capture_output=True, text=True, timeout=1200).stdout.splitlines()
    dirs = {os.path.dirname(p) for p in out if p.endswith("/test.npy")}
    flat = [p for p in out if p.endswith("_test.npy")]
    return dirs, flat


def valid_run_dir(d: str, dim: int) -> bool:
    try:
        shapes = {}
        for s in ("train", "val", "test"):
            X = np.load(os.path.join(d, f"{s}.npy"), mmap_mode="r")
            y = np.load(os.path.join(d, f"{s}_labels.npy"), mmap_mode="r")
            shapes[s] = (X.shape, y.shape, int(y.max()))
    except Exception:
        return False
    (nt, dt), (nv, dv), (ne, ddim) = shapes["train"], shapes["val"], shapes["test"]
    return (ne == N_TEST and nt >= 40_000 and nv >= 10_000
            and dt == dv == ddim == dim
            and shapes["test"][2] <= 10)


def link_mirror(src_dir: str, rel_tmpl: str, seed) -> None:
    """Lie les 6 fichiers de src_dir vers MIRROR/<relpath avec /{s} remplacé>."""
    rel_dir = rel_tmpl.format(s=seed)
    dst_dir = os.path.join(MIRROR, rel_dir)
    os.makedirs(dst_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        for suffix in (".npy", "_labels.npy"):
            f = s + suffix
            dst = os.path.join(dst_dir, f)
            if not os.path.lexists(dst):
                os.symlink(os.path.join(src_dir, f), dst)


def link_frozen(key: str, flat_dir: str) -> None:
    dst_dir = os.path.join(MIRROR, "embeddings")
    os.makedirs(dst_dir, exist_ok=True)
    for s in ("train", "val", "test"):
        for suffix in (".npy", "_labels.npy"):
            f = f"{key}_{s}{suffix}"
            src = os.path.join(flat_dir, f)
            dst = os.path.join(dst_dir, f)
            if os.path.exists(src) and not os.path.lexists(dst):
                os.symlink(src, dst)


def main() -> None:
    forced = load_forced()
    dirs, flat = index_scratch()
    flat_by_key = {}
    for p in flat:
        key = os.path.basename(p)[: -len("_test.npy")]
        flat_by_key.setdefault(key, os.path.dirname(p))

    n_ok, missing = 0, []
    for name, kind, path, seeds, _ in GROUPS:
        dim = DIM.get(name, 768)
        got = {}
        for sd in seeds:
            rel_tmpl = path
            tmpl = TAG_ALIAS.get(path, path).format(s=sd)
            basename = os.path.basename(tmpl)
            if path in PATH_PAT:
                pat = PATH_PAT[path].format(s=sd)
                cands_pre = [d for d in dirs if d.endswith(pat.lstrip("*/"))]
            else:
                cands_pre = None
            # 1) chemin job d'origine dans SCRATCH, tel quel
            c1 = os.path.join(SCRATCH, os.path.dirname(tmpl), basename) \
                 if "{" not in tmpl else None
            # 2) même chemin relatif sous le dépôt cloné (hasard de layout)
            # 3) recherche par basename dans l'index des run-dirs
            cands = cands_pre if cands_pre is not None \
                else [d for d in dirs if os.path.basename(d) == basename]
            if kind == "frozen":
                fd = flat_by_key.get(path)
                if fd and valid_run_dir_flat(fd, path, dim):
                    link_frozen(path, fd); got[sd] = fd
                else:
                    missing.append((name, "frozen", path, dim))
                continue
            chosen, valid = None, []
            for c in ([c1] if c1 else []) + cands:
                if valid_run_dir(c, dim):
                    valid.append(c)
            key = f"{name}|seed{sd}"
            if key in forced:
                if valid_run_dir(forced[key], dim):
                    chosen = forced[key]
                else:
                    print(f"  [forced] {key}: {forced[key]} ÉCHECHE la validation")
            elif len(valid) == 1:
                chosen = valid[0]
            elif len(valid) > 1:
                print(f"  AMBIGU {key}: {len(valid)} candidats — choisir dans "
                      f"{FORCED_FILE} (lignes '{key}\t<dir>')")
                for v in valid:
                    print(f"        - {v}")
            if chosen:
                link_mirror(chosen, rel_tmpl, sd)
                got[sd] = chosen
        if len(got) == len(seeds):
            n_ok += 1
            ex = next(iter(got.values()))
            print(f"  OK   {name:38s} -> {ex}")
        else:
            missing.append((name, kind, path, dim))
            print(f"  ✗    {name:38s} ({len(got)}/{len(seeds)} seeds trouvés)")

    with open(os.path.join(SCRATCH, "head_sweep_inputs_MISSING.txt"), "w") as f:
        for name, kind, path, dim in missing:
            f.write(f"{name}\t{kind}\t{path}\t{dim}\n")
    print(f"\n[map] {n_ok}/33 groupes mappés et liés sous {MIRROR}")
    print(f"[map] (les AMBIGUS comptent comme manquants tant que "
          f"{FORCED_FILE} n'est pas rempli)")
    if missing:
        print(f"[map] {len(missing)} MANQUANTS (détail dans head_sweep_inputs_MISSING.txt) :")
        for name, kind, path, dim in missing:
            print(f"  - {name} ({kind}, {path}, dim {dim})")
        print("[map] → pour ceux-là seulement, remonter du laptop :")
        print("    (laptop) python3 scripts/prepare_head_sweep_push.py && \\\n"
        "              MANIFEST=/tmp/head_sweep_missing.txt bash scripts/push_head_sweep_inputs.sh")
    else:
        print("[map] → GO : sbatch scripts/slurm_head_sweep_all.sh")


def valid_run_dir_flat(d: str, key: str, dim: int) -> bool:
    try:
        shapes = []
        for s in ("train", "val", "test"):
            X = np.load(os.path.join(d, f"{key}_{s}.npy"), mmap_mode="r")
            y = np.load(os.path.join(d, f"{key}_{s}_labels.npy"), mmap_mode="r")
            shapes.append((X.shape[0], X.shape[1], int(y.max())))
    except Exception:
        return False
    return (shapes[2][0] == N_TEST and shapes[0][0] >= 40_000
            and shapes[1][0] >= 10_000
            and {d1 for _, d1, _ in shapes} == {dim}
            and all(mx <= 11 for *_, mx in shapes))   # frozen: labels 12cls possibles


if __name__ == "__main__":
    main()
