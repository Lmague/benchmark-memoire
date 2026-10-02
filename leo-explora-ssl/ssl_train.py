#!/usr/bin/env python3
"""ExPLoRA vrai : extended pre-training SSL (DINO + iBOT + KoLeo) sur tuiles Léo.

Réf : Khanna et al., "ExPLoRA: Parameter-Efficient Extended Pre-Training
to Adapt Vision Transformers under Domain Shifts", arXiv:2406.10973.
Ici : SimDINOv2 ViT-B/16 (iNat-Plantae, photos au sol) -> orthomosaïques
aériennes 224px (nadir). Puis fine-tune supervisé LoRA (cf. train.py Mémoire).

Groupes de paramètres (fidèle au papier, U={2 derniers blocs}) :
  - blocs 0..9  : Q,V wrappés LoRA (r configurable), reste GELÉ
  - blocs 10,11 : full fine-tuning
  - toutes les LayerNorm (+ norm finale) : dégelées
  - patch_embed / pos_embed / cls_token / registers : GELÉS
  - heads DINO+iBOT neuves (random, le ckpt SimDINOv2 ne garde que le backbone)

Teacher = EMA du student (momentum cos 0.996->1.0). Objectifs :
  DINO cls (2 globales teacher <- 2 globales + N locales student),
  iBOT patchs masqués sur globales, KoLeo sur cls student.

Checkpoint sauvé au format SimDINOv2 (clé `teacher`, préfixe `backbone.`,
LoRA fusionnée) -> chargeable tel quel par get_model.load_simdino_state_dict.

Usage (Narval, 1x A100 40Go) :
    python ssl_train.py --config configs/leo_vitb16_ssl.json
    python ssl_train.py --config ... --dry-run   # construit, 10 steps test
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from simdinov2.loss.dino_clstoken_loss import DINOLoss
from simdinov2.loss.ibot_patch_loss import iBOTPatchLoss
from simdinov2.loss.koleo_loss import KoLeoLoss
from simdinov2.layers.dino_head import DINOHead
from simdinov2.models.vision_transformer import vit_base

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- LoRA Q,V
class QVLoRA(nn.Module):
    """Remplace un qkv Linear gelé : sortie + delta LoRA sur tranches Q et V.

    qkv : Linear(dim, 3*dim) ; tranches [0:dim]=Q, [dim:2dim]=K, [2dim:3dim]=V.
    Seules Q et V reçoivent un adaptateur rang-r (fidèle ExPLoRA).
    """

    def __init__(self, qkv: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.qkv = qkv
        for p in self.qkv.parameters():
            p.requires_grad = False
        dim = qkv.in_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A_q = nn.Parameter(torch.zeros(r, dim))
        self.lora_A_v = nn.Parameter(torch.zeros(r, dim))
        self.lora_B_q = nn.Parameter(torch.zeros(dim, r))
        self.lora_B_v = nn.Parameter(torch.zeros(dim, r))
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

    def forward(self, x):
        out = self.qkv(x)
        dim = self.qkv.in_features
        dq = (x @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling
        dv = (x @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling
        out = out.clone()
        out[..., 0:dim] += dq
        out[..., 2 * dim:3 * dim] += dv
        return out

    def merged_qkv(self) -> nn.Linear:
        """Retourne un Linear équivalent (pour export du checkpoint)."""
        lin = copy.deepcopy(self.qkv)
        with torch.no_grad():
            dq = (self.lora_B_q @ self.lora_A_q) * self.scaling
            dv = (self.lora_B_v @ self.lora_A_v) * self.scaling
            lin.weight[0:lin.in_features] += dq
            lin.weight[2 * lin.in_features:] += dv
        return lin


def apply_explora(backbone: nn.Module, r: int, alpha: float, n_full: int = 2):
    """Gèle tout puis ouvre : LoRA-QV (blocs sauf n_full derniers),
    full-FT (n_full derniers), LayerNorms partout."""
    n_blocks = len(backbone.blocks)
    for p in backbone.parameters():
        p.requires_grad = False
    # 1) LoRA Q,V sur les premiers blocs
    for i in range(n_blocks - n_full):
        blk = backbone.blocks[i]
        blk.attn.qkv = QVLoRA(blk.attn.qkv, r=r, alpha=alpha)
    # 2) derniers blocs full-FT
    for i in range(n_blocks - n_full, n_blocks):
        for p in backbone.blocks[i].parameters():
            p.requires_grad = True
    # 3) toutes les LayerNorm (blocs + norm finale + patch_embed si norm)
    for m in backbone.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True
    groups = {"lora": [], "full_late": [], "norm": [], "head": []}
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            groups["lora"].append(p)
        elif any(f"blocks.{i}." in n for i in range(n_blocks - n_full, n_blocks)):
            groups["full_late"].append(p)
        else:
            groups["norm"].append(p)
    return groups


# ---------------------------------------------------------------- dataset
class SSLTiles(Dataset):
    """n_global vues globales (augmentations INDÉPENDANTES) + n_local locales.

    Le collate par défaut empile chaque liste -> ( [g1_batch, g2_batch\n    , ...], [l1_batch, ...] ). Deux globales distinctes sont requises par DINO
    (le teacher est centré sur ses globales, le student prédit les locales).
    """
    def __init__(self, files, n_global=2, n_local=6):
        self.files = files
        self.n_global = max(1, int(n_global))
        self.n_local = n_local

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        globs = [global_view(img) for _ in range(self.n_global)]
        locals_ = [local_view(img) for _ in range(self.n_local)]
        return globs, locals_


def _norm(t):
    t = torch.from_numpy(np.asarray(t, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _jitter(img):
    import torchvision.transforms.v2 as T
    j = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02)
    return j(img)


def global_view(img):
    import torchvision.transforms.v2 as T
    g = T.RandomResizedCrop(224, scale=(0.4, 1.0),
                            interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True)(img)
    if random.random() < 0.5:
        g = T.functional.hflip(g)
    return _norm(_jitter(g))


def local_view(img):
    import torchvision.transforms.v2 as T
    l = T.RandomResizedCrop(96, scale=(0.05, 0.4),
                            interpolation=T.InterpolationMode.BICUBIC,
                            antialias=True)(img)
    if random.random() < 0.5:
        l = T.functional.hflip(l)
    return _norm(_jitter(l))


def random_mask(B, n_patches=196, ratio=0.3):
    n_mask = int(n_patches * ratio)
    m = torch.zeros(B, n_patches, dtype=torch.bool)
    for b in range(B):
        idx = torch.randperm(n_patches)[:n_mask]
        m[b, idx] = True
    return m


# ---------------------------------------------------------------- train
def build_ssl_backbone(r, alpha, n_full):
    bb = vit_base(patch_size=16, img_size=224, layerscale=1e-5,
                  block_chunks=0, num_register_tokens=4)
    groups = apply_explora(bb, r, alpha, n_full)
    return bb, groups


def load_backbone(ckpt: Path):
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw["teacher"] if isinstance(raw, dict) and "teacher" in raw else raw
    import re
    out = {}
    for k, v in sd.items():
        k2 = k.removeprefix("backbone.")
        if not k.startswith("backbone."):
            continue
        out[re.sub(r"blocks\.\d+\.(?=\d+)", "blocks.", k2)] = v
    # NOTE: on charge le ckpt d'origine AVANT d'injecter LoRA (clés qkv standard).
    bb = vit_base(patch_size=16, img_size=224, layerscale=1e-5,
                  block_chunks=0, num_register_tokens=4)
    bb.load_state_dict(out, strict=True)
    return bb


@torch.no_grad()
def ema_update(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def merged_teacher_state(teacher_bb):
    """State dict backbone compatible SimDINOv2 (LoRA fusionnée)."""
    lora_mods = {n for n, m in teacher_bb.named_modules() if isinstance(m, QVLoRA)}
    sd = {}
    for n, m in teacher_bb.named_modules():
        if isinstance(m, QVLoRA):
            lin = m.merged_qkv()
            sd[n + ".weight"] = lin.weight.detach().cpu()
            if lin.bias is not None:
                sd[n + ".bias"] = lin.bias.detach().cpu()
    for n, p in teacher_bb.named_parameters():
        modpath = n.rsplit(".", 1)[0]
        if modpath in lora_mods or any(
                modpath == lp or modpath.startswith(lp + ".") for lp in lora_mods):
            continue  # remplacé par la version fusionnée ci-dessus
        sd[n] = p.detach().cpu()
    return {"backbone." + k: v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg['seed'] ; suffixe cfg['out_dir'] par _seed{N}")
    ap.add_argument("--out-dir", default=None, help="override cfg['out_dir']")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    seed = int(cfg.get("seed", 0))
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    elif args.seed is not None:
        cfg["out_dir"] = f"{cfg['out_dir']}_seed{seed}"

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run] seed={seed} out_dir={cfg['out_dir']}", flush=True)

    # --- données (un manifest ou une liste) ---
    mans = cfg["manifest"] if isinstance(cfg["manifest"], list) else [cfg["manifest"]]
    files, green = [], []
    for mf in mans:
        man = json.loads(Path(mf).read_text())["tiles"]
        for idx, t in man.items():
            p = Path(cfg["tiles_dir"]) / t["f"]  # f = "{site}/{idx}.jpg"
            if p.exists():
                files.append(str(p))
                green.append(t.get("green", 0.0))
    green = np.array(green)
    w = np.ones(len(files))
    up = cfg.get("green_upsample", 4.0)
    w[green > float(np.median(green))] = up
    print(f"[data] {len(files)} tuiles (poids verts x{up} au-dessus de la médiane)")
    ds = SSLTiles(files, cfg.get("n_global", 2), cfg.get("n_local", 6))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(w), len(files), replacement=True)
    dl = DataLoader(ds, batch_size=cfg.get("batch", 128), sampler=sampler,
                    num_workers=cfg.get("workers", 8), pin_memory=True,
                    drop_last=True)
    if args.dry_run:
        print(f"[dry-run] {len(ds)} tuiles, {len(dl)} steps/époque")

    # --- modèle : student charge le ckpt, teacher = copie à l'identique ---
    student_bb = load_backbone(Path(cfg["simdinov2_ckpt"])).to(device)
    groups = apply_explora(student_bb, cfg.get("lora_r", 16),
                           cfg.get("lora_alpha", 16.0), cfg.get("n_full_blocks", 2))
    dim, out_dim = 768, cfg.get("head_dim", 16384)
    s_head = DINOHead(dim, out_dim).to(device)
    s_ihead = DINOHead(dim, out_dim).to(device)
    for p in list(s_head.parameters()) + list(s_ihead.parameters()):
        p.requires_grad = True
    groups["head"] = [p for p in list(s_head.parameters()) + list(s_ihead.parameters())]

    teacher_bb, _ = build_ssl_backbone(cfg.get("lora_r", 16),
                                       cfg.get("lora_alpha", 16.0),
                                       cfg.get("n_full_blocks", 2))
    teacher_bb.load_state_dict(student_bb.state_dict(), strict=True)
    teacher_bb.to(device)
    t_head = DINOHead(dim, out_dim).to(device)
    t_ihead = DINOHead(dim, out_dim).to(device)
    t_head.load_state_dict(s_head.state_dict())
    t_ihead.load_state_dict(s_ihead.state_dict())
    for m in [teacher_bb, t_head, t_ihead]:
        for p in m.parameters():
            p.requires_grad = False

    n_tr = sum(p.numel() for p in student_bb.parameters() if p.requires_grad)
    n_tr += sum(p.numel() for p in groups["head"])
    n_tot = sum(p.numel() for p in student_bb.parameters())
    print(f"[model] entraînables backbone: {n_tr:,} / {n_tot:,} "
          f"({100*n_tr/n_tot:.1f}%)")
    for g, ps in groups.items():
        print(f"  groupe '{g}': {len(ps)} tensors")

    dino_loss = DINOLoss(out_dim).to(device)
    ibot_loss = iBOTPatchLoss(out_dim).to(device)
    koleo = KoLeoLoss().to(device)

    opt = torch.optim.AdamW([
        {"params": groups["lora"], "lr": cfg.get("lr_lora", 1e-4)},
        {"params": groups["full_late"], "lr": cfg.get("lr_late", 5e-5)},
        {"params": groups["norm"], "lr": cfg.get("lr_norm", 1e-5)},
        {"params": groups["head"], "lr": cfg.get("lr_head", 1e-4)},
    ], weight_decay=cfg.get("weight_decay", 0.05), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    epochs = cfg.get("epochs", 50) if not args.dry_run else 1
    steps_per_ep = len(dl)
    warmup = cfg.get("warmup_epochs", 3) * steps_per_ep
    total = epochs * steps_per_ep
    t_temp = cfg.get("teacher_temp", 0.07)
    m0, m1 = cfg.get("ema_0", 0.996), cfg.get("ema_1", 1.0)

    out = Path(cfg["out_dir"]); (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    step, t0 = 0, time.time()
    n_G = int(cfg.get("n_global", 2))
    n_patches = (224 // 16) ** 2
    save_every = int(cfg.get("save_every_epochs", 5))
    # nombre de termes de la loss DINO (student x teacher, diagonale exclue) —
    # normalisation fidèle au vendor (ssl_meta_arch_sim.py, n_total_crops_loss_terms)
    n_dino_terms = n_G * max(0, n_G - 1) + n_G * cfg.get("n_local", 6)

    student_bb.train()
    for ep in range(epochs):
        for globs, locals_ in dl:
            step += 1
            frac = min(1.0, step / max(1, total))
            if step == 1:
                for pg in opt.param_groups:
                    pg["base_lr"] = pg["lr"]
            lr_scale = min(1.0, step / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * frac))
            for pg in opt.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale
            m = m1 - (m1 - m0) * 0.5 * (1 + math.cos(math.pi * frac))

            # globales : n_G vues par image, empilées -> (n_G*B, 3, 224, 224)
            global_crops = torch.cat(
                [g.to(device, non_blocking=True) for g in globs], dim=0)
            local_crops = [l.to(device, non_blocking=True) for l in locals_]
            B = global_crops.size(0) // n_G
            masks = random_mask(global_crops.size(0), n_patches=n_patches).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(device == "cuda")):
                # ---- teacher : n_G globales propres, UNE seule passe, sans masque ----
                with torch.no_grad():
                    tout = teacher_bb(global_crops, is_training=True)
                    t_cls = t_head(tout["x_norm_clstoken"])        # (n_G*B, out)
                    t_pl = t_ihead(tout["x_norm_patchtokens"])     # (n_G*B, N, out)
                    dino_loss.update_center(t_cls)
                    ibot_loss.update_center(t_pl)
                    t_cls_c = dino_loss.softmax_center_teacher(
                        t_cls, t_temp).view(n_G, B, -1)             # (n_G, B, out)
                    t_pl_c = ibot_loss.softmax_center_teacher(t_pl, t_temp)
                # ---- student : globales MASQUÉES (iBOT) + locales ----
                s_out_g = student_bb(global_crops, masks=masks, is_training=True)
                s_cls_g = s_head(s_out_g["x_norm_clstoken"])       # (n_G*B, out)
                s_pl_g = s_ihead(s_out_g["x_norm_patchtokens"])    # (n_G*B, N, out)
                s_cls_loc = [s_head(student_bb(l, is_training=True)["x_norm_clstoken"])
                             for l in local_crops]

                # DINO multi-crop : [globales..., locales...] vs les n_G globales teacher,
                # diagonale (même globale) exclue. Normalisé par le nb de termes.
                s_cls_list = list(s_cls_g.chunk(n_G)) + s_cls_loc
                l_dino, _ = dino_loss(s_cls_list, list(t_cls_c), no_diag=True)
                l_dino = l_dino / max(1, n_dino_terms)
                # iBOT : patchs masqués des globales (dense pondéré par le masque)
                l_ibot = ibot_loss(s_pl_g, t_pl_c, masks)
                # KoLeo : une fois par globale (pas entre les vues d'une même image)
                l_koleo = sum(koleo(c) for c in s_cls_g.chunk(n_G))
                loss = (l_dino + cfg.get("w_ibot", 1.0) * l_ibot
                        + cfg.get("w_koleo", 0.5) * l_koleo)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for ps in groups.values() for p in ps], 1.0)
            scaler.step(opt)
            scaler.update()
            ema_update(teacher_bb, student_bb, m)
            ema_update(t_head, s_head, m)
            ema_update(t_ihead, s_ihead, m)

            if step % cfg.get("log_every", 50) == 0:
                print(f"[ep{ep} step{step}/{total}] loss={loss.item():.4f} "
                      f"(dino={l_dino.item():.3f} ibot={l_ibot.item():.3f} "
                      f"koleo={l_koleo.item():.3f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if args.dry_run and step >= 10:
                print("[dry-run] OK — 10 steps passés, arrêt.")
                return
        if (save_every > 0 and (ep + 1) % save_every == 0) or ep == epochs - 1:
            state = {"teacher": merged_teacher_state(teacher_bb),
                     "epoch": ep, "config": cfg}
            torch.save(state, out / "checkpoints" / f"ep{ep:03d}.pth")
            torch.save(state, out / "checkpoints" / "last.pth")
    print(f"TERMINÉ: {out/'checkpoints'} (ep{epochs-1:03d}.pth + last.pth)")


if __name__ == "__main__":
    main()
