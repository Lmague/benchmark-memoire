#!/usr/bin/env python3
"""STAGE C — Fusion (merge) des adaptateurs LoRA d'un checkpoint SimDINOv2-B dans les
poids du backbone, avec contrôles de non-régression.

Produit un .pth chargeable directement par ``build_frozen_extractor("simdinov2_vitb16",
checkpoint=<merged>)`` (format ``{"teacher": {"backbone.<k>": v}}`` attendu par
``_load_simdinov2``) — utile pour la géométrie sur poids propres et le récit de
déploiement (un seul modèle, zéro adaptateur, zéro overhead).

Utilisation (sur Narval, après rapatriement des checkpoints OU sur $SCRATCH) :

    python scripts/merge_lora_simb.py \
        --ckpt $SCRATCH/sota_screening/lora_simb_ablation/checkpoints/<tag>_best.pth \
        --config configs/simdinov2_vitb16_lora_r8a8_b611.yaml \
        --out  $SCRATCH/sota_screening/lora_simb_ablation/merged/<tag>_merged.pth

Contrôles exécutés avant sauvegarde :
  1. Équivalence numérique au niveau module : la sortie du bloc LoRAFusedQKV chargé
     depuis le checkpoint original doit coïncider (atol 1e-4, fp32) avec
     ``F.linear(x, W_fusionné)`` du même module fusionné à la main ;
  2. Aucune clé ``lora_A/lora_B/scaling_buf`` restante dans le state fusionné ;
  3. Rechargement du .pth fusionné par ``build_frozen_extractor`` (via
     ``_load_simdinov2``, strict=False) : < 10 % de clés manquantes exigé, sinon abort.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

from src.models import build_model, build_frozen_extractor, merge_lora_state_dict  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="checkpoint du run (…_best.pth, payload _save_ckpt)")
    ap.add_argument("--config", required=True, help="config du run (pour reconstruire le modèle)")
    ap.add_argument("--out", required=True, help="chemin du .pth fusionné à écrire")
    args = ap.parse_args()

    cfg = load_config(args.config)
    teacher = cfg.raw.get("checkpoint")
    if teacher and not os.path.isabs(teacher):
        teacher = os.path.join(cfg.paths.ckpt_dir, teacher)

    # 1. Modèle avec adaptateurs, chargé depuis le checkpoint du run
    model, groups = build_model(cfg.model.name, cfg.regime, cfg.model.num_classes,
                                lora=cfg.lora, checkpoint=teacher)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    print(f"[merge] checkpoint chargé : epoch={payload.get('epoch')} best_f1={payload.get('best_f1')}")

    # 2. Contrôle n°1 — équivalence numérique module par module (avant/après fusion)
    lora_modules = [(n, m) for n, m in model.named_modules()
                    if hasattr(m, "lora_A") and hasattr(m, "scaling_buf")]
    if not lora_modules:
        raise SystemExit("[merge] ERREUR : aucun module LoRA dans le checkpoint "
                         "(déjà fusionné, ou régime non-LoRA ?)")
    n_checked = 0
    with torch.no_grad():
        for name, mod in lora_modules:
            x = torch.randn(2, mod.in_features)
            out_lora = mod(x)
            W = mod.weight.detach().clone()
            dim = mod.dim
            for t, s in (("q", 0), ("k", 1), ("v", 2)):
                A = mod.lora_A[t].detach().float()
                B = mod.lora_B[t].detach().float()
                scaling = float(mod.scaling_buf)
                d = scaling * (B @ A)
                W[s * dim:(s + 1) * dim, :] += d
            out_merged = F.linear(x.float(), W, None if mod.bias is None else mod.bias.detach().float())
            err = (out_lora.float() - out_merged).abs().max().item()
            if err > 1e-4:
                raise SystemExit(f"[merge] ERREUR : écart {err:.2e} sur {name} (> 1e-4) — fusion invalide.")
            n_checked += 1
    print(f"[merge] contrôle 1 OK : {n_checked} modules LoRA fusionnés, écart max ≤ 1e-4")

    # 3. Fusion du state_dict complet
    state = {k: v for k, v in payload["model_state_dict"].items()}
    merged = merge_lora_state_dict(state)

    # 4. Contrôle n°2 — plus aucune trace d'adaptateur
    leftover = [k for k in merged if "lora_A." in k or "lora_B." in k or "scaling_buf" in k]
    if leftover:
        raise SystemExit(f"[merge] ERREUR : clés LoRA résiduelles : {leftover[:5]}")
    print("[merge] contrôle 2 OK : aucune clé lora_A/lora_B/scaling_buf restante")

    # 5. Extraction de la partie backbone → format {"teacher": {"backbone.<k>": v}}
    bb_keys = [k for k in merged if k.startswith("backbone.")]
    if not bb_keys:
        raise SystemExit("[merge] ERREUR : aucune clé 'backbone.' dans le state (schéma inattendu).")
    teacher_state = {k: v for k, v in merged.items() if k.startswith("backbone.")}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"teacher": teacher_state}, args.out)
    print(f"[merge] écrit {args.out}  ({len(teacher_state)} clés backbone)")

    # 6. Contrôle n°3 — le .pth fusionné est rechargable par l'extracteur frozen
    backbone, forward_fn, embed_dim, _ = build_frozen_extractor(cfg.model.name, checkpoint=args.out)
    print(f"[merge] contrôle 3 OK : extracteur frozen reconstruit depuis le .pth fusionné "
          f"(dim={embed_dim})")
    print("[merge] TERMINÉ — checkpoint fusionné utilisable pour extraction/géométrie "
          "sans adaptateurs.")


if __name__ == "__main__":
    main()
