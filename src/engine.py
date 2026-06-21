"""Boucle d'entraînement : optimizer/scheduler, AMP, early stopping, checkpoints, test.

Logique canonique (notebooks + spec) :
- AdamW, LR différentiels par groupe (depuis ``config.optim.lr``).
- CosineAnnealingLR(eta_min=1e-7), calé sur les STEPS de gradient (pas les epochs) —
  garantit que deux runs avec des volumes de données différents (ex. courbe de
  données Q5) suivent la MÊME trajectoire de LR par step, donc sont comparables.
  Warmup linéaire + phase head-only = OPTIONNELS (défaut off).
- AMP float16 sur CUDA. Early stopping sur **val F1-Macro (12 classes)**.
- Anti-écrasement : checkpoints ``{model}_{regime}_best.pth`` / ``_last.pth`` ;
  résultats ``{model}_{regime}_results.json``.

torch/numpy importés au niveau module (engine n'est chargé que par train.py).
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.optim import AdamW

from .metrics import confusion, eval_classifier, per_class_count, per_class_f1
from .utils import CLASS_NAMES, ensure_dir, run_tag, save_json


# ----------------------------------------------------------------- optim / schedule
def build_optimizer(groups: dict, cfg) -> AdamW:
    """AdamW avec un groupe par entrée de ``groups`` ; LR pris dans ``cfg.optim.lr``."""
    lr_map = cfg.optim.lr
    missing = [g for g in groups if g not in lr_map]
    if missing:
        raise ValueError(
            f"LR manquant dans config.optim.lr pour les groupes {missing} "
            f"(disponibles: {list(lr_map)}). regime='{cfg.regime}'.")
    param_groups = [{"params": groups[g], "lr": float(lr_map[g]), "name": g}
                    for g in groups if len(groups[g]) > 0]
    if not param_groups:
        raise ValueError("Aucun paramètre entraînable — vérifier le régime / le modèle.")
    return AdamW(param_groups, weight_decay=cfg.optim.weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int = 1):
    """CosineAnnealingLR(eta_min), CALÉ SUR LES STEPS (pas sur les epochs).

    ``T_max`` et les milestones de warmup sont exprimés en steps de gradient
    (``steps_per_epoch × epochs``), pas en epochs. Avant ce fix, deux runs avec
    des volumes de données différents (ex. courbe de données Q5 : 270 steps/epoch
    à 70% vs 386 à 100%) recevaient le même nombre d'epochs mais un nombre de
    steps différent pour la même position dans le cycle cosine — ce qui rendait
    les runs incomparables et causait une instabilité accrue (et donc une chute
    de F1 test) au volume de données le plus élevé malgré un meilleur état au
    best_epoch en val. ``steps_per_epoch=1`` (défaut) reproduit l'ancien
    comportement epoch-par-epoch si jamais appelé sans le paramètre.
    """
    from torch.optim.lr_scheduler import (CosineAnnealingLR, LinearLR,
                                          SequentialLR)
    epochs = cfg.train.epochs
    warmup_epochs = cfg.schedule.warmup_epochs
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max(1, (epochs - warmup_epochs) * steps_per_epoch)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps,
                               eta_min=cfg.schedule.eta_min)
    if warmup_steps > 0:
        warm = LinearLR(optimizer, start_factor=1.0 / warmup_steps, end_factor=1.0,
                        total_iters=warmup_steps)
        return SequentialLR(optimizer, schedulers=[warm, cosine], milestones=[warmup_steps])
    return cosine


def _maybe_head_only(model, cfg, epoch: int) -> bool:
    """Optionnel : gèle le backbone pendant ``head_only_epochs`` epochs (défaut off)."""
    hoe = cfg.train.head_only_epochs
    if not hoe or hoe <= 0:
        return False
    in_phase = epoch < hoe
    for n, p in model.named_parameters():
        is_head = ("head" in n) or n.startswith("fc")
        p.requires_grad = True if is_head else (not in_phase)
    return in_phase


# --------------------------------------------------------------------- train / eval
def train_one_epoch(model, loader, optimizer, scaler, criterion, device,
                    scheduler=None, use_amp: bool = True, log_every: int = 50) -> float:
    """Entraîne une epoch. ``scheduler.step()`` appelé APRÈS CHAQUE BATCH (step-based)."""
    model.train()
    total, n = 0.0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item()
        n += 1
        if log_every and i % log_every == 0:
            print(f"    batch {i:>5}/{len(loader)} loss={loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return total / max(1, n)


@torch.no_grad()
def predict(model, loader, device, use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (preds, labels) sur tout le loader."""
    model.eval()
    preds, labels = [], []
    amp = use_amp and device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


# ------------------------------------------------------------------- checkpoint IO
def _save_ckpt(path, model, optimizer, scaler, epoch, best_f1, history) -> None:
    ensure_dir(os.path.dirname(path))
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_f1": best_f1,
        "history": history,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_ckpt(path, model, optimizer=None, scaler=None) -> tuple[int, float, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt.get("epoch", -1), ckpt.get("best_f1", 0.0), ckpt.get("history", _empty_history())


def _empty_history() -> dict:
    return {"train_loss": [], "val_f1_macro": [], "val_f1_weighted": [], "val_acc": [], "lr": []}


# -------------------------------------------------------------------------- fit
def fit(cfg, model, groups: dict, loaders: dict, criterion, device,
        resume: str | None = None, tag_override: str | None = None) -> dict:
    """Entraîne, early-stoppe sur val F1-Macro, évalue le best sur le test, sauve résultats."""
    optimizer = build_optimizer(groups, cfg)
    steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=steps_per_epoch)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    tag = tag_override or run_tag(cfg.model.name, cfg.regime)
    ensure_dir(cfg.paths.ckpt_dir)
    ensure_dir(cfg.paths.results_dir)
    best_path = os.path.join(cfg.paths.ckpt_dir, f"{tag}_best.pth")
    last_path = os.path.join(cfg.paths.ckpt_dir, f"{tag}_last.pth")

    history = _empty_history()
    start_epoch, best_f1, patience_left = 0, 0.0, cfg.train.patience
    if resume and os.path.exists(resume):
        start_epoch, best_f1, history = _load_ckpt(resume, model, optimizer, scaler)
        start_epoch += 1
        print(f"[{tag}] reprise à l'epoch {start_epoch + 1}, best_f1={best_f1:.4f}")

    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.train.epochs):
        _maybe_head_only(model, cfg, epoch)
        lrs = [g["lr"] for g in optimizer.param_groups]
        train_loss = train_one_epoch(model, loaders["train"], optimizer, scaler,
                                     criterion, device, scheduler=scheduler, use_amp=use_amp)
        preds, labels = predict(model, loaders["val"], device, use_amp)
        val = eval_classifier(labels, preds, cfg.model.num_classes)
        # scheduler.step() retiré ici — désormais appelé PAR BATCH dans train_one_epoch
        # (step-based cosine, cf. build_scheduler).

        history["train_loss"].append(train_loss)
        history["val_f1_macro"].append(val["f1_macro_all"])
        history["val_f1_weighted"].append(val["f1_weighted"])
        history["val_acc"].append(val["accuracy"])
        history["lr"].append(lrs)

        improved = val["f1_macro_all"] > best_f1
        print(f"[{tag}] epoch {epoch + 1}/{cfg.train.epochs} loss={train_loss:.4f} "
              f"valF1m={val['f1_macro_all']:.4f} valF1w={val['f1_weighted']:.4f} "
              f"acc={val['accuracy']:.4f}" + ("  *best*" if improved else ""), flush=True)

        if improved:
            best_f1 = val["f1_macro_all"]
            patience_left = cfg.train.patience
            _save_ckpt(best_path, model, optimizer, scaler, epoch, best_f1, history)
        else:
            patience_left -= 1
        if (epoch + 1) % cfg.train.save_every == 0:
            _save_ckpt(last_path, model, optimizer, scaler, epoch, best_f1, history)
        if patience_left <= 0:
            print(f"[{tag}] early stopping à l'epoch {epoch + 1} "
                  f"(val F1-Macro stagnante depuis {cfg.train.patience} epochs)")
            break

    _save_ckpt(last_path, model, optimizer, scaler, epoch, best_f1, history)

    # --- évaluation test sur le meilleur checkpoint ---
    if os.path.exists(best_path):
        _load_ckpt(best_path, model)
    test_preds, test_labels = predict(model, loaders["test"], device, use_amp)
    test = eval_classifier(test_labels, test_preds, cfg.model.num_classes)
    results = {
        "model": cfg.model.name,
        "regime": cfg.regime,
        "tag": tag,
        "classes": CLASS_NAMES,
        "config": {
            "epochs": cfg.train.epochs,
            "batch_size": cfg.data.batch_size,
            "weight_decay": cfg.optim.weight_decay,
            "lr": cfg.optim.lr,
            "eta_min": cfg.schedule.eta_min,
            "warmup_epochs": cfg.schedule.warmup_epochs,
            "head_only_epochs": cfg.train.head_only_epochs,
            "augmentation": "notebook_moummad",
        },
        "best_val_f1_macro": best_f1,
        "test": {**test,
                 "f1_per_class": per_class_f1(test_labels, test_preds),
                 "n_per_class": per_class_count(test_labels)},
        "confusion_matrix": confusion(test_labels, test_preds).tolist(),
        "history": history,
    }
    save_json(results, os.path.join(cfg.paths.results_dir, f"{tag}_results.json"))
    print(f"[{tag}] test F1m(12)={test['f1_macro_all']:.4f} "
          f"F1m(pres)={test['f1_macro_pres']:.4f} acc={test['accuracy']:.4f} → résultats sauvés")
    return results
