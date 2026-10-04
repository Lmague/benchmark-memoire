#!/usr/bin/env python3
"""Dose-réponse d'alignement — extraction frozen des checkpoints SSL Léo sur Arctic-TVC.

CONTEXTE SCIENTIFIQUE
---------------------
Le SSL ExPLoRA (`leo-explora-ssl/`) adapte SimDINOv2-B/16 iNat-Plantae au domaine
aérien des orthomosaïques Léo (nadir, 8-17 mm/px, non labellisées). Pour mesurer
*comment l'alignement du pré-entraînement agit* sur la cible Arctic-TVC, on sonde
CHAQUE checkpoint sauvegardé (toutes les 5 époques) **sans aucun entraînement
supervisé** : la sonde linéaire gelée est le mètre de transférabilité, et la
trajectoire en fonction de la dose de SSL aérien est la dose-réponse.

Trois compteurs par checkpoint (tous dérivés de CETTE extraction) :
  - ``tile``  : F1 du classifieur linéaire sur les features de la tuile (768) ;
  - ``ctx``   : F1 sur le contexte SEUL (768) — le mètre qui discrimine le mieux les
                backbones alignés (SimB-iNat @512 : 0.4931 vs DINOv3-LVD : 0.4592) ;
  - ``fused`` : F1 sur la concaténation [tuile ; contexte] (1536) — la référence
                gelée à battre (SimB-iNat @512 = 0.5059).

On enregistre tuile ET contexte séparément (une seule passe forward) plutôt que le
fusionné : les trois compteurs sortent alors de la même extraction, sans repasser
deux fois sur les images.

FORMAT DU CHECKPOINT
--------------------
Identique à SimDINOv2 : clé ``teacher``, préfixe ``backbone.``, LoRA **fusionnée**
dans ``qkv.weight`` (cf. `leo-explora-ssl/README.md` et ``merged_teacher_state``).
``build_frozen_extractor`` -> ``_load_simdinov2`` (src/models.py) le charge donc tel
quel (vérifié : 176/176 clés, clés plates). Le .pth d'origine
``simdinov2_vitb_inat21plantae.pth`` est accepté aussi : c'est le point **époque 0**
de la trajectoire (LoRA initialisée à zéro ⇒ teacher ≡ SimB iNat).

SORTIE (par tag)
----------------
    <out-dir>/<tag>/<split>_tile.npy     float32 [N, 768]  features tuile 224px
    <out-dir>/<tag>/<split>_ctx.npy      float32 [N, 768]  features contexte (redim. 224)
    <out-dir>/<tag>/<split>_labels.npy   int64   [N]       labels 11 classes (RHOL retirée)
    <out-dir>/<tag>/meta.json            provenance complète

Repartable : un tag dont les 3 splits × (tile, ctx, labels) existent est sauté
sauf ``--force`` (convention du dépôt).

USAGE (Narval, via scripts/slurm_leo_ssl_arctic_extract.sh) :
    python scripts/leo_ssl_extract_arctic.py \\
        --ckpt $SCRATCH/leo_ssl/runs/leo_vitb16_ssl_seed0/checkpoints/ep004.pth \\
        --tiles-dir $SLURM_TMPDIR/tiles \\
        --context-dir $SLURM_TMPDIR/context_512 \\
        --csv-dir spatial_datacurve/splits/frac100_seed0 \\
        --out-dir $SCRATCH/leo_ssl/arctic_probe/sig_embeddings \\
        --tag leossl_b16_seed0_ep004
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))                 # import src.*
sys.path.insert(0, str(PROJ / "scripts"))     # import datacurve_one_run

SPLITS = ("train", "val", "test")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- labels
def read_split_11cls(csv_path: str) -> tuple[list[str], np.ndarray]:
    """CSV 12 classes → (filepaths, labels 11 classes).

    Réplique EXACTEMENT ``datacurve_one_run._apply_11cls_remap`` + la lecture de
    ``context_distill._read_split_12cls_filtered_remapped`` (filtrage RHOL puis
    remapping 12→11 via ``LABEL_REMAP_12TO11``) — importé, jamais redéfini, pour
    qu'aucun écart de schéma ne puisse s'introduire.
    """
    from datacurve_one_run import LABEL_REMAP_12TO11, _RHOL_IDX
    from src.utils import read_split_csv

    fps_all, labels_12 = read_split_csv(csv_path)
    keep = np.array([int(l) != _RHOL_IDX for l in labels_12], dtype=bool)
    fps = [fps_all[i] for i in np.nonzero(keep)[0]]
    labels_11 = np.array([LABEL_REMAP_12TO11[int(l)] for l in labels_12[keep]],
                         dtype=np.int64)
    return fps, labels_11


# --------------------------------------------------------------------- dataset
class TileCtxDataset:
    """(tuile, contexte) appariés par le MÊME chemin relatif.

    Protocole identique à ``context_distill._make_eval_loader_with_context`` : un
    seul transform (eval déterministe, résolution 224) appliqué aux deux vues, et
    le contexte lu sous le même ``relpath`` que la tuile (garanti par
    ``context_crop.py``). Aucun appariement aléatoire, aucun resize divergent.
    """

    def __init__(self, fps: list[str], tiles_dir: str, context_dir: str, transform):
        self.fps = fps
        self.tiles_dir = tiles_dir
        self.context_dir = context_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.fps)

    def __getitem__(self, i: int):
        from PIL import Image
        fp = self.fps[i]
        tile = Image.open(os.path.join(self.tiles_dir, fp))
        ctx = Image.open(os.path.join(self.context_dir, fp))
        if tile.mode != "RGB":
            tile = tile.convert("RGB")
        if ctx.mode != "RGB":
            ctx = ctx.convert("RGB")
        return self.transform(tile), self.transform(ctx)


def _sanity_check(model, forward_fn, dataset, dim: int, n: int = 8) -> None:
    """Garde-fou avant les 80k tuiles (mêmes seuils que src.features.sanity_check)."""
    import torch
    model.eval()
    xb = torch.stack([dataset[i][0] for i in range(min(n, len(dataset)))])
    device = next(model.parameters()).device
    with torch.no_grad():
        out = forward_fn(model, xb.to(device)).float().cpu()
    std_dim = float(out.std(0).mean())
    nrm = float(out.norm(dim=1).mean())
    log(f"[sanity] shape={tuple(out.shape)} std/dim={std_dim:.4f} ||v||={nrm:.2f}")
    if out.ndim != 2 or out.shape[1] != dim:
        raise RuntimeError(f"[sanity] shape {tuple(out.shape)} (dim attendue {dim})")
    if std_dim <= 0.01:
        raise RuntimeError(f"[sanity] std/dim={std_dim:.4f} <= 0.01 (collapse / CLS mal extrait)")
    if nrm <= 1.0:
        raise RuntimeError(f"[sanity] ||v||={nrm:.2f} <= 1.0")


def _ckpt_meta(path: str) -> dict:
    """Lit (best effort) l'époque SSL et un sous-ensemble de la config d'entraînement.

    Second ``torch.load`` du checkpoint (le premier est fait par
    ``build_frozen_extractor``) : ~5-10 s, accepté pour la traçabilité. Ne lève
    jamais — un checkpoint d'un autre format doit rester exploitable.
    """
    try:
        import torch
        raw = torch.load(path, map_location="cpu", weights_only=False)
        out: dict = {}
        if isinstance(raw, dict):
            if "epoch" in raw:
                out["ssl_epoch_0based"] = int(raw["epoch"])
            cfg = raw.get("config")
            if isinstance(cfg, dict):
                keys = ("seed", "epochs", "save_every_epochs", "batch", "lora_r",
                        "lora_alpha", "n_full_blocks", "n_global", "n_local",
                        "green_upsample", "lr_lora", "lr_late", "w_ibot", "w_koleo")
                out["ssl_config"] = {k: cfg[k] for k in keys if k in cfg}
            out["ckpt_top_keys"] = sorted(str(k) for k in raw.keys())
        return out
    except Exception as e:  # noqa: BLE001 — provenance best-effort
        return {"ckpt_meta_error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------- extraction
def _extract_split(model, forward_fn, dataset, labels, split: str, batch: int,
                   workers: int, device, amp: bool, torch_dtype, max_samples: int | None,
                   tag: str):
    import torch
    from torch.utils.data import DataLoader

    n = len(dataset) if not max_samples else min(max_samples, len(dataset))
    loader = DataLoader(dataset, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=(device.type == "cuda"))
    E_tile, E_ctx = [], []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for tile, ctx in loader:
            tile = tile.to(device, non_blocking=True)
            if amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    f_t = forward_fn(model, tile)
                    f_c = forward_fn(model, ctx.to(device, non_blocking=True))
            else:
                f_t = forward_fn(model, tile)
                f_c = forward_fn(model, ctx.to(device, non_blocking=True))
            E_tile.append(f_t.detach().cpu().to(torch_dtype).numpy())
            E_ctx.append(f_c.detach().cpu().to(torch_dtype).numpy())
            done += tile.shape[0]
            if done % (batch * 50) < batch:
                log(f"  [{tag}/{split}] {done}/{n} "
                    f"({done / max(1e-9, time.time() - t0):.0f} img/s)")
            if max_samples and done >= max_samples:
                break
    Et = np.concatenate(E_tile, axis=0)[:n]
    Ec = np.concatenate(E_ctx, axis=0)[:n]
    L = np.asarray(labels)[:n]
    log(f"  [{tag}/{split}] {Et.shape} + ctx {Ec.shape} ({time.time() - t0:.0f}s)")
    return Et, Ec, L


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint .pth : ExPLoRA-Léo (clé `teacher`, LoRA fusionnée) "
                         "ou SimDINOv2 d'origine (point époque 0)")
    ap.add_argument("--tag", required=True,
                    help="nom du dossier de sortie (convention : leossl_b16_seed{N}_ep{EEE})")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tiles-dir", required=True, help="tuiles Arctic 224px (ex. $SLURM_TMPDIR/tiles)")
    ap.add_argument("--context-dir", required=True,
                    help="dossier context_<size> produit par context_crop.py (ex. $SLURM_TMPDIR/context_512)")
    ap.add_argument("--csv-dir", required=True,
                    help="dossier des CSV de split (ex. spatial_datacurve/splits/frac100_seed0)")
    ap.add_argument("--backbone", default="simdinov2_vitb16",
                    choices=["simdinov2_vitb16", "simdinov2_vitl16",
                             "simdinov2_vitb16_imagenet", "simdinov2_vitl16_imagenet"])
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true",
                    help="autocast fp16 (plus rapide ; les features différent du run fp32 "
                         "à ~1e-3 — laisser OFF si l'on veut reproduire au plus près les "
                         "chiffres canoniques du chapitre contexte)")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                    help="dtype de sauvegarde (float32 = convention sig_embeddings)")
    ap.add_argument("--context-size", type=int, default=512, help="métadonnée de provenance")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="limiter N par split (smoke test)")
    ap.add_argument("--force", action="store_true", help="ré-extraire même si présent")
    args = ap.parse_args()

    import torch
    from src.data import build_transforms
    from src.models import build_frozen_extractor
    from src.utils import get_device, get_normalization

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out = Path(args.out_dir) / args.tag
    torch_dtype = torch.float32 if args.dtype == "float32" else torch.float16

    # --- repartable -----------------------------------------------------------
    expected = [f"{s}_{kind}.npy" for s in splits for kind in ("tile", "ctx", "labels")]
    if not args.force and all((out / f).exists() for f in expected):
        log(f"[skip] {args.tag} déjà complet ({out}) — utiliser --force pour refaire")
        return
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint introuvable : {args.ckpt}")
    out.mkdir(parents=True, exist_ok=True)

    device = get_device()
    t_start = time.time()
    log(f"tag={args.tag} | ckpt={args.ckpt} | device={device} | batch={args.batch} "
        f"amp={args.amp} dtype={args.dtype}")

    # --- modèle ---------------------------------------------------------------
    # Le checkpoint ExPLoRA a EXACTEMENT le format attendu par _load_simdinov2
    # (clé `teacher` + préfixe `backbone.` + LoRA fusionnée) : chargement direct.
    model, forward_fn, dim, norm_key = build_frozen_extractor(args.backbone, args.ckpt)
    model = model.to(device).eval()
    mean, std = get_normalization(norm_key)
    log(f"backbone={args.backbone} dim={dim} norm={norm_key}")

    # --- extraction -----------------------------------------------------------
    tf = build_transforms("eval", mean, std, 224)
    n_by_split: dict[str, int] = {}
    for s in splits:
        csv_path = os.path.join(args.csv_dir, f"{s}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV de split introuvable : {csv_path}")
        fps, labels = read_split_11cls(csv_path)
        ds = TileCtxDataset(fps, args.tiles_dir, args.context_dir, tf)
        if s == splits[0]:
            _sanity_check(model, forward_fn, ds, dim)
        Et, Ec, L = _extract_split(model, forward_fn, ds, labels, s, args.batch,
                                   args.num_workers, device, args.amp, torch_dtype,
                                   args.max_samples, args.tag)
        np.save(out / f"{s}_tile.npy", Et)
        np.save(out / f"{s}_ctx.npy", Ec)
        np.save(out / f"{s}_labels.npy", L)
        n_by_split[s] = int(len(L))
        log(f"[ok] {args.tag}/{s} : tile {Et.shape}, ctx {Ec.shape} → {out}")

    # --- provenance -----------------------------------------------------------
    meta = {
        "tag": args.tag,
        "backbone": args.backbone,
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_name": os.path.basename(args.ckpt),
        "ckpt_size_bytes": os.path.getsize(args.ckpt),
        "dim": int(dim),
        "norm": norm_key,
        "context_size": int(args.context_size),
        "tiles_dir": args.tiles_dir,
        "context_dir": args.context_dir,
        "csv_dir": os.path.abspath(args.csv_dir),
        "splits": splits,
        "n_by_split": n_by_split,
        "dtype": args.dtype,
        "amp": bool(args.amp),
        "batch": int(args.batch),
        "num_workers": int(args.num_workers),
        "protocol": ("features frozen, transforms eval déterministes build_transforms, "
                     "224px ; tuile et contexte appariés par relpath ; labels 11 classes "
                     "(RHOL retirée) ; aucune tête, aucun entraînement"),
        "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t_start, 1),
        **_ckpt_meta(args.ckpt),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"terminé : {out} ({meta['runtime_s']}s)")


if __name__ == "__main__":
    main()
