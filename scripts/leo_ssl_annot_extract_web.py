#!/usr/bin/env python3
"""Extraction GPU des embeddings des annotations Léo depuis les JPEG web (Narval).

CONTEXTE
--------
`scripts/leo_ssl_annot_prepare_index.py` (local) a résolu les coordonnées UTM des
annotations en indices de tuile (`idx = tr*260 + tc`, ortho Clairière). Ce script fait
le reste sur Narval, sur les tuiles **déjà présentes** dans `$SCRATCH/web/tiles/clairiere`
(355 486 JPEG canoniques, les mêmes que les embeddings gelés) : un forward par tuile, pour
la baseline SimDINOv2-B **et** chaque seed ExPLoRA-Léo. Aucun raster requis (131 Go, local).

⚠️ Les JPEG web sont **stretchés** (percentiles 2-98 par tuile, cf. `export_tiles.py`), pas
le raster brut. Les embeddings produits ici ne sont donc PAS bit-comparables à l'ancien
`embeddings/bank.npz` (raster brut) — mais ils sont comparables ENTRE TAGS, ce qui est ce
qui compte pour la comparaison gelé vs adapté. On ré-extrait donc aussi la baseline.

⚠️ Pas de contexte 512 px ici : les JPEG sont des tuiles 224 isolées, et le stretch
par tuile rend tout recollement de voisinage incohérent. Le rapport de séparabilité
(`embeddings/separability_report.md` §2) a mesuré que le contexte n'apporte **rien** sur
ce corpus (0.878 fusionné vs 0.875 tuile) → on enregistre `emb_ctx` nul et un drapeau
`ctx: false` dans `meta.json` ; la sonde se limite à la variante `tile`.

Sorties (par tag) :
    <out-dir>/<tag>/bank.npz        species, fid, x, y, tr, tc, emb_tile, emb_ctx(=0)
    <out-dir>/<tag>/negatives.npz   bg_tile/ctx, bg_x/y, rej_tile/ctx, rej_species, rej_x/y
    <out-dir>/<tag>/meta.json       provenance + ctx=false

Usage (via scripts/slurm_leo_ssl_annot_eval.sh) :
    python scripts/leo_ssl_annot_extract_web.py \\
        --ckpt $SCRATCH/checkpoints/simdinov2_vitb_inat21plantae.pth --tag base \\
        --index $SCRATCH/annot_leo/annot_index.npz --web-dir $SCRATCH/web \\
        --out-dir $SCRATCH/annot_leo/emb
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
sys.path.insert(0, str(PROJ))


def _require(mods_env: str) -> None:
    """Message clair si l'interpréteur n'est pas le bon (cf. docstring)."""
    missing = []
    for m in ("torch", "PIL", "numpy", "sklearn", "einops"):
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        raise SystemExit(
            f"[env] {', '.join(missing)} manquant(s) dans {sys.executable}.\n"
            f"      {mods_env}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class WebTiles:
    """tuiles 224px depuis `<web-dir>/tiles/<site>/<part>/<tr>/<idx>.jpg`."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i])
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True, help="base | ssl_seed0 | ssl_seed1 | ssl_seed2")
    ap.add_argument("--index", required=True, help="annot_index.npz (prepare_index.py)")
    ap.add_argument("--web-dir", required=True, help="racine web ($SCRATCH/web)")
    ap.add_argument("--site", default="clairiere")
    ap.add_argument("--manifest", default=None,
                    help="défaut : <web-dir>/manifest_<site>.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--backbone", default="simdinov2_vitb16",
                    choices=["simdinov2_vitb16", "simdinov2_vitl16"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    _require("Sur Narval : source $HOME/ENV/bin/activate. En local : "
             "/home/erazal/miniconda3/bin/python")
    import torch
    from src.data import build_transforms
    from src.models import build_frozen_extractor
    from src.utils import get_device, get_normalization
    from torch.utils.data import DataLoader

    out = Path(args.out_dir) / args.tag
    if not args.force and (out / "bank.npz").exists() and (out / "negatives.npz").exists():
        log(f"[skip] {args.tag} déjà extrait ({out})")
        return
    out.mkdir(parents=True, exist_ok=True)

    web = Path(args.web_dir)
    man_path = Path(args.manifest) if args.manifest else web / f"manifest_{args.site}.json"
    man = json.loads(man_path.read_text())["tiles"]
    log(f"manifest {man_path.name} : {len(man)} tuiles")

    idx = np.load(args.index, allow_pickle=True)
    t0 = time.time()
    device = get_device()
    model, forward_fn, dim, norm_key = build_frozen_extractor(args.backbone, args.ckpt)
    model = model.to(device).eval()
    mean, std = get_normalization(norm_key)
    tf = build_transforms("eval", mean, std, 224)
    log(f"tag={args.tag} ckpt={Path(args.ckpt).name} dim={dim} norm={norm_key} "
        f"device={device} amp={args.amp}")

    def relpath(i: int) -> str | None:
        m = man.get(str(int(i)))
        return None if m is None else str(m["f"])

    # ---- liste des tuiles à encoder (uniques) ------------------------------------
    bank_idx = idx["bank_idx"].astype(np.int32)
    idxs_bg = np.array([relpath(i) is not None for i in idx["bg_idx"]], dtype=bool)
    bg_idx = idx["bg_idx"].astype(np.int32)[idxs_bg]
    rej_idx_all = idx["rej_idx"].astype(np.int32)
    rej_keep = np.array([relpath(i) is not None for i in rej_idx_all])
    rej_idx = rej_idx_all[rej_keep]
    if len(bg_idx) < len(idx["bg_idx"]):
        log(f"[info] {len(idx['bg_idx']) - len(bg_idx)} tuiles de fond absentes du manifest "
            f"(tuiles vides) — retirées, identique pour tous les tags")
    if (~rej_keep).any():
        log(f"[info] {int((~rej_keep).sum())} candidats rejetés hors manifest — retirés")

    uniq = np.unique(np.concatenate([bank_idx, bg_idx, rej_idx]))
    missing_bank = [int(i) for i in np.unique(bank_idx) if relpath(i) is None]
    if missing_bank:
        raise SystemExit(f"[erreur] {len(missing_bank)} tuiles annotées absentes du manifest "
                         f"(ex. {missing_bank[:5]}) — annotations hors zone exportée ?")
    log(f"à encoder : {len(uniq)} tuiles uniques "
        f"(bank {len(np.unique(bank_idx))}, fond {len(np.unique(bg_idx))}, "
        f"rejetés {len(np.unique(rej_idx))})")

    # ---- extraction ---------------------------------------------------------------
    paths = [str(web / "tiles" / args.site / relpath(i)) for i in uniq]
    ds = WebTiles(paths, tf)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    feats = np.zeros((len(uniq), dim), dtype=np.float16)
    k = 0
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            if args.amp and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    f = forward_fn(model, xb)
            else:
                f = forward_fn(model, xb)
            n = xb.shape[0]
            feats[k:k + n] = f.detach().cpu().to(torch.float16).numpy()
            k += n
    pos = {int(v): i for i, v in enumerate(uniq)}
    log(f"features {feats.shape} en {time.time() - t0:.0f}s")

    # ---- assemblage ---------------------------------------------------------------
    bank_ok = np.array([relpath(i) is not None for i in bank_idx])
    bi = np.array([pos[int(i)] for i in bank_idx[bank_ok]])
    np.savez_compressed(
        out / "bank.npz",
        species=idx["bank_species"][bank_ok],
        fid=idx["bank_fid"][bank_ok].astype(np.int32),
        x=idx["bank_x"][bank_ok].astype(np.float64),
        y=idx["bank_y"][bank_ok].astype(np.float64),
        tr=idx["bank_tr"][bank_ok].astype(np.int32),
        tc=idx["bank_tc"][bank_ok].astype(np.int32),
        emb_tile=feats[bi],
        emb_ctx=np.zeros_like(feats[bi]),          # pas de contexte 512 (cf. docstring)
    )
    bg_pos = np.array([pos[int(i)] for i in bg_idx])
    rej_pos = np.array([pos[int(i)] for i in rej_idx])
    np.savez_compressed(
        out / "negatives.npz",
        bg_tile=feats[bg_pos], bg_ctx=np.zeros_like(feats[bg_pos]),
        bg_x=idx["bg_x"][idxs_bg], bg_y=idx["bg_y"][idxs_bg],
        rej_tile=feats[rej_pos], rej_ctx=np.zeros_like(feats[rej_pos]),
        rej_species=idx["rej_species"][rej_keep],
        rej_x=idx["rej_x"][rej_keep], rej_y=idx["rej_y"][rej_keep],
        bg_idx=bg_idx, rej_idx=rej_idx,
    )
    (out / "meta.json").write_text(json.dumps({
        "tag": args.tag, "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_name": Path(args.ckpt).name, "backbone": args.backbone,
        "dim": int(dim), "norm": norm_key, "site": args.site,
        "source": "JPEG web stretchés ($SCRATCH/web/tiles)", "ctx": False,
        "n_bank": int(bank_ok.sum()), "n_bg": int(len(bg_idx)), "n_rej": int(len(rej_idx)),
        "manifest": str(man_path), "index": str(Path(args.index).resolve()),
        "amp": bool(args.amp), "batch": int(args.batch), "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t0, 1),
    }, indent=2))
    log(f"[ok] {args.tag} : bank {int(bank_ok.sum())} pts, {len(bg_idx)} fond, "
        f"{len(rej_idx)} rejetés → {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
