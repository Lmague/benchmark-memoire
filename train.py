#!/usr/bin/env python3
"""CLI fine-tuning : ``python train.py --config configs/<modele>_<regime>.yaml [--dry-run]``.

Toute la logique vit dans ``src/`` ; ce script orchestre. ``--dry-run`` construit dataset,
modèle, groupes de params, loss, optimizer et scheduler, imprime un résumé, puis s'arrête
SANS lancer l'entraînement.
"""
from __future__ import annotations

import argparse
import os

from src import engine, utils
from src.config import load_config
from src.data import make_loaders
from src.losses import build_class_weights, build_class_weights_effective_num, build_criterion
from src.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML modèle×régime (configs/...)")
    ap.add_argument("--dry-run", action="store_true",
                    help="construit tout sans entraîner")
    ap.add_argument("--resume", default=None, help="checkpoint .pth à reprendre")
    args = ap.parse_args()

    cfg = load_config(args.config)
    utils.set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    utils.maybe_mount_drive(cfg.env)
    device = utils.get_device()
    print(f"[train] model={cfg.model.name} regime={cfg.regime} env={cfg.env} device={device}")

    # --- checkpoint pré-entraîné (backbones SSL_FT_NAMES, ex. SimDINOv2) ---
    # Chemin relatif -> résolu contre ckpt_dir (même logique que extract.py) ; ignoré si absent
    # (DINOv3-HF n'a pas besoin de `checkpoint:`, chargé via AutoModel.from_pretrained).
    checkpoint = cfg.raw.get("checkpoint")
    if checkpoint and not os.path.isabs(checkpoint):
        checkpoint = os.path.join(cfg.paths.ckpt_dir, checkpoint)

    # --- modèle + groupes de params (valide le régime, ex. mhsa interdit sur CNN) ---
    model, groups = build_model(cfg.model.name, cfg.regime, cfg.model.num_classes,
                                lora=cfg.lora, checkpoint=checkpoint)
    model = model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] params total={n_total:,} entraînables={n_train:,} "
          f"({100 * n_train / max(1, n_total):.1f}%)")
    for g, ps in groups.items():
        print(f"[train]   groupe '{g}': {len(ps)} tensors @ lr={cfg.optim.lr.get(g)}")

    # --- loss pondérée 1/sqrt(n) depuis les effectifs du train ---
    _, train_labels = utils.read_split_csv(os.path.join(cfg.paths.csv_dir, "train.csv"))
    weighting = getattr(cfg.train, "weighting", "sqrt")
    beta = getattr(cfg.train, "cui_beta", 0.999)
    weights = (build_class_weights(train_labels, cfg.model.num_classes) if weighting == "sqrt"
            else build_class_weights_effective_num(train_labels, cfg.model.num_classes, beta))
    print(f"[train] weighting={weighting}" + (f" beta={beta}" if weighting != "sqrt" else ""))
    print("[train] class weights:",
        {c: round(float(w), 3) for c, w in zip(utils.CLASS_NAMES_11, weights)})
    criterion = build_criterion(train_labels, cfg.model.num_classes, device,
                                weighting=weighting, beta=beta)
                                
    # --- loaders ---
    loaders = make_loaders(cfg, train_aug=True)
    print("[train] tailles:", {s: len(dl.dataset) for s, dl in loaders.items()})

    if args.dry_run:
        optimizer = engine.build_optimizer(groups, cfg)
        scheduler = engine.build_scheduler(optimizer, cfg,
                                           steps_per_epoch=len(loaders["train"]))
        print(f"[dry-run] optimizer={type(optimizer).__name__} "
              f"({len(optimizer.param_groups)} groupes) | scheduler={type(scheduler).__name__}")
        print("[dry-run] OK — tout construit, entraînement NON lancé.")
        return

    engine.fit(cfg, model, groups, loaders, criterion, device, resume=args.resume)


if __name__ == "__main__":
    main()
