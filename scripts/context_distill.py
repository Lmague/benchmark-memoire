#!/usr/bin/env python3
"""Self-distillation contexte→tuile — DINOv3 student + LoRA, 3 axes de variation.

Correspond au tableau "plan final" (R1/R2/R3) de l'utilisateur — R1 = teacher externe
DINOv3-L + Design A (défauts CLI ci-dessous), R2 = ``--design B``, R3 = ``--teacher
ema_self``. Voir ``scripts/context_distill_README.md`` §1 pour la table complète et
la convention de nommage (R1/R2/R3 désignent CES 3 lignes, jamais une taille de
contexte, pour éviter la confusion vécue le 2026-08-29).

Un student DINOv3 ViT-B/16 LVD + LoRA apprend à encoder une TUILE (224px) en
s'aidant d'une fenêtre de CONTEXTE spatial plus large (512/1024/2048px, produite
par ``scripts/context_crop.py``). Trois axes de variation, indépendants :

1. ``--design {A,B}`` — QUAND le contexte est utilisé.
   - A (défaut) : contexte au TRAIN SEULEMENT. À l'inférence : un seul modèle, une
     seule entrée (la tuile 224px). Le contexte et le teacher ne servent qu'à
     régulariser l'entraînement (self-distillation).
   - B : contexte au train ET à l'inférence — "borne supérieure théorique" (pas
     forcément déployable), teste si le fait de DISPOSER du contexte au moment de
     prédire aide, par-delà l'effet régularisateur de la distillation seule. Garde
     l'architecture de A INTACTE (teacher, distillation, λ, focal) et change
     SEULEMENT l'entrée de la tête de classification : delta d'UNE variable par
     rapport à A, cf. §Design B ci-dessous.
2. ``--teacher`` — QUI encode le contexte pour la distillation.
   - Un backbone frozen externe (ex. ``dinov3_vitl16_lvd``, le teacher "riche" par
     défaut — R1 du tableau, et sa base pour R2/Design B).
   - ``ema_self`` — pas de modèle externe : le teacher est une copie EMA (momentum)
     du backbone du STUDENT lui-même (même archi DINOv3-B), mise à jour après
     chaque step. Isole si le gain vient d'un teacher plus riche (DINOv3-L) ou
     simplement d'un signal de self-distillation, cf. §EMA self-teacher.
3. ``--context-size`` — taille de la fenêtre de contexte (512/1024/2048).

Loss = focal(classif, logits) + λ · distill(proj(feat_tuile student) vs feat_contexte
teacher). Réutilise DIRECTEMENT (pas de réimplémentation) :
  - ``src.models.build_model(..., regime="lora", lora=cfg.lora)``     — injection LoRA
  - ``src.models.build_frozen_extractor``                             — chargement teacher externe
  - ``src.engine.build_optimizer`` / ``build_scheduler``               — AdamW + cosine step-based
  - ``src.losses.build_class_weights``                                 — pondération 1/sqrt(n)
  - ``src.utils.make_canonical_lr``                                    — sonde linéaire canonique
  - ``scripts.datacurve_one_run._extract_backbone_embeddings`` / ``_apply_11cls_remap``
    — extraction Design A (tuile seule), MÊME code que le pipeline LoRA canonique.

── Schéma de labels ────────────────────────────────────────────────────────────
``splits_spatial/*/train.csv`` (et val/test) sont en schéma BRUT 12 classes (RHOL
présent au train, absent du val/test) — vérifié empiriquement (152 lignes RHOL dans
train.csv, 0 dans val/test). Ce script filtre RHOL et remappe 12→11 classes EXACTEMENT
comme ``scripts/datacurve_one_run.py`` (même table ``LABEL_REMAP_12TO11``, réutilisée
par import).

── Token d'échelle (GSD) ───────────────────────────────────────────────────────
Un petit MLP sur ``log(context_size / 224)``, concaténé à la feature CLS TUILE du
student avant la tête de projection (embed_dim+scale)→teacher_dim. Conditionne
UNIQUEMENT la branche de distillation (jamais la classification) — cf. commentaire
détaillé dans la version précédente de ce fichier / docs, raison inchangée par
Design B ou l'EMA self-teacher : la scale sert à dire "cette cible de distillation
vient d'un contexte de taille X", peu importe qui l'a produite.

── Design B — un seul delta par rapport à A ─────────────────────────────────────
Design B garde EXACTEMENT l'architecture de A (teacher externe ou EMA, distillation,
focal, λ) et ajoute UNE SEULE chose : la tête de classification prend
``[feat_tuile ; feat_contexte]`` (2×embed_dim) au lieu de ``feat_tuile`` seule — les
deux features viennent du MÊME backbone LoRA student (poids partagés, deux forwards).
La branche de distillation reste identique à A (seule ``feat_tuile`` est projetée et
comparée au teacher) : Design B répond à "le contexte aide-t-il en plus à
l'inférence", pas "l'architecture de distillation change-t-elle". Casser cette
contrainte (ex. supprimer le teacher pour B) répondrait à une question différente
et rendrait A/B incomparables — voir la discussion du 2026-08-29 dans
``scripts/context_distill_README.md``.
Conséquence pratique : le CONTEXTE DOIT EXISTER POUR VAL ET TEST AUSSI sous Design B
(pas seulement train comme sous Design A) — ``context_crop.py`` doit avoir été relancé
sur ``val.csv``/``test.csv`` dans le MÊME ``--context-dir`` avant de lancer un run B.

── EMA self-teacher (``--teacher ema_self``) ────────────────────────────────────
Teacher = copie EMA (momentum, ``--ema-momentum``, défaut 0.999) des paramètres
ENTRAÎNABLES du backbone student (LoRA A/B + LayerNorm + éventuel full_late) — la
base DINOv3 gelée est identique par construction entre les deux copies (jamais mise
à jour côté student), donc l'EMA dessus serait un no-op ; exclue pour ne pas itérer
sur ~86M paramètres inutilement à chaque step. Mise à jour après chaque
``optimizer.step()``, sous ``torch.no_grad()``, jamais de gradient à travers le
teacher (comme le teacher externe). ``teacher_dim`` devient l'embed_dim du student
(768), pas 1024 (DINOv3-L) — la tête de projection s'adapte automatiquement.

── Contexte NON augmenté ───────────────────────────────────────────────────────
Le contexte reçoit un transform déterministe (resize + normalisation), jamais les
augmentations stochastiques de la tuile — évite un flip désynchronisé entre les deux
vues. Normalisation : la même clé ("imagenet") sert au teacher externe DINOv3-L, à
l'EMA self-teacher DINOv3-B et à la branche contexte du student sous Design B — les
trois partagent actuellement la même normalisation dans ce dépôt ; si un teacher
futur en utilisait une différente, il faudrait deux transforms de contexte distincts
(non implémenté, pas nécessaire aujourd'hui).

Usage :
    python scripts/context_distill.py \\
        --config configs/context_distill_dinov3b.yaml \\
        --context-size 1024 --context-dir $SLURM_TMPDIR/context_1024 \\
        --design A --teacher dinov3_vitl16_lvd \\
        --seed 0 --out-dir results/context_distill_2026-08-29
"""
from __future__ import annotations

# --- Mono-thread BLAS AVANT numpy/sklearn (AGENTS.md §4.8) -----------------------
# Ce script exécute la sonde canonique (make_canonical_lr) en fin de run et ses
# chiffres sont comparés DIRECTEMENT aux baselines canoniques (0.4835, blocs
# 6-11=0.4844, etc.) — donc soumis à la même exigence de reproductibilité que
# scripts/rapport/significance_tier.py. Sans effet notable sur l'entraînement GPU
# (le gros du calcul est CUDA, pas BLAS CPU).
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # racine dépôt
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # scripts/ (sibling import)

_TEACHER_TAG = {"dinov3_vitl16_lvd": "tL", "ema_self": "tEMA"}


def _teacher_tag(name: str) -> str:
    """Alias court pour le tag de run — cf. garde-fou anti-collision dans main().

    Fallback : nom brut si le teacher n'est pas dans la table (reste unique, juste
    plus long) — ne JAMAIS lever ici, seul le garde-fou de collision doit bloquer.
    """
    return _TEACHER_TAG.get(name, name)


# ─────────────────────────────────────────────────────────────────── Dataset

def _read_split_12cls_filtered_remapped(csv_path: str):
    """Lit un CSV 12cls, filtre RHOL, remappe 12→11 — réutilise la table de
    datacurve_one_run.py (même schéma, cf. docstring module)."""
    from datacurve_one_run import LABEL_REMAP_12TO11, _RHOL_IDX
    from src.utils import read_split_csv
    fps_all, labels_12 = read_split_csv(csv_path)
    keep = [i for i, l in enumerate(labels_12) if int(l) != _RHOL_IDX]
    fps = [fps_all[i] for i in keep]
    labels_11 = np.array([LABEL_REMAP_12TO11[int(labels_12[i])] for i in keep], dtype=np.int64)
    return fps, labels_11


class TileContextDataset:
    """Paire (tuile 224px augmentée, contexte pré-redimensionné, label 11cls).

    ``tiles_dir``/``context_dir`` partagent le même chemin relatif par construction
    (context_crop.py écrit sous le même ``relpath`` que le tile CSV d'origine).
    """

    def __init__(self, csv_path, tiles_dir, context_dir, tile_transform, context_transform):
        self.tiles_dir = tiles_dir
        self.context_dir = context_dir
        self.tile_transform = tile_transform
        self.context_transform = context_transform
        self.fps, self.labels = _read_split_12cls_filtered_remapped(csv_path)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, idx):
        from PIL import Image
        fp = self.fps[idx]
        tile = Image.open(_os.path.join(self.tiles_dir, fp)).convert("RGB")
        ctx = Image.open(_os.path.join(self.context_dir, fp)).convert("RGB")
        return self.tile_transform(tile), self.context_transform(ctx), int(self.labels[idx])


def _make_train_loader(cfg, context_dir, teacher_norm_key, batch_size, num_workers, seed):
    import torch
    from torch.utils.data import DataLoader
    from src.data import build_transforms
    from src.utils import get_normalization, seed_worker

    mean, std = get_normalization(cfg.model.norm)
    t_mean, t_std = get_normalization(teacher_norm_key)
    tile_tf = build_transforms("train", mean, std, cfg.data.image_size)
    ctx_tf = build_transforms("eval", t_mean, t_std, cfg.data.image_size)  # déterministe, cf. docstring

    ds = TileContextDataset(_os.path.join(cfg.paths.csv_dir, "train.csv"),
                            cfg.paths.tiles_dir, context_dir, tile_tf, ctx_tf)
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g,
                        worker_init_fn=seed_worker, num_workers=num_workers,
                        pin_memory=True, persistent_workers=num_workers > 0)
    return loader, ds.labels


def _make_eval_loader(cfg, split, batch_size, num_workers):
    """Design A : tuile SEULE, même transform eval que tous les autres régimes."""
    from torch.utils.data import DataLoader
    from src.data import build_transforms
    from src.utils import get_normalization

    mean, std = get_normalization(cfg.model.norm)
    tf = build_transforms("eval", mean, std, cfg.data.image_size)
    fps, labels_11 = _read_split_12cls_filtered_remapped(_os.path.join(cfg.paths.csv_dir, f"{split}.csv"))

    class _Wrapped:
        def __len__(self_inner):
            return len(fps)

        def __getitem__(self_inner, i):
            from PIL import Image
            img = Image.open(_os.path.join(cfg.paths.tiles_dir, fps[i])).convert("RGB")
            return tf(img), int(labels_11[i])

    ds = _Wrapped()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, labels_11


def _make_eval_loader_with_context(cfg, split, context_dir, batch_size, num_workers):
    """Design B SEULEMENT : tuile + contexte, transform eval déterministe pour les
    deux. Suppose que ``context_crop.py`` a déjà produit le contexte pour CE split
    (val/test, pas seulement train comme sous Design A) — sinon ``FileNotFoundError``
    explicite au premier ``Image.open`` manquant (pas de fallback silencieux).
    """
    from torch.utils.data import DataLoader
    from src.data import build_transforms
    from src.utils import get_normalization

    mean, std = get_normalization(cfg.model.norm)
    tf = build_transforms("eval", mean, std, cfg.data.image_size)
    fps, labels_11 = _read_split_12cls_filtered_remapped(_os.path.join(cfg.paths.csv_dir, f"{split}.csv"))

    class _Wrapped:
        def __len__(self_inner):
            return len(fps)

        def __getitem__(self_inner, i):
            from PIL import Image
            fp = fps[i]
            tile = Image.open(_os.path.join(cfg.paths.tiles_dir, fp)).convert("RGB")
            ctx = Image.open(_os.path.join(context_dir, fp)).convert("RGB")
            return tf(tile), tf(ctx), int(labels_11[i])

    ds = _Wrapped()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return loader, labels_11


# ─────────────────────────────────────────────────────────────────── Modèle

def build_projection_head(embed_dim: int, teacher_dim: int, scale_embed_dim: int = 32):
    """Tête de projection (branche distillation) — dimensionnée sur ``teacher_dim``
    (1024 pour DINOv3-L externe, ``embed_dim``=768 pour l'EMA self-teacher). Séparée
    de la construction du student (cf. ``main()``) : jamais attachée comme attribut
    de ``classifier`` (PyTorch l'enregistrerait en sous-module et polluerait
    ``classifier.state_dict()`` avec des clés "proj.*"/"scale_mlp.*" — le checkpoint
    ne resterait alors garanti "backbone.*/head.* pur" que par accident de filtrage
    de préfixe côté ``load_finetuned_ssl_backbone``, pas par construction).
    """
    import torch.nn as nn
    scale_mlp = nn.Sequential(
        nn.Linear(1, scale_embed_dim), nn.GELU(), nn.Linear(scale_embed_dim, scale_embed_dim),
    )
    proj = nn.Linear(embed_dim + scale_embed_dim, teacher_dim)
    return proj, scale_mlp


def _replace_head_for_fusion(classifier, groups, embed_dim: int, num_classes: int):
    """Design B : remplace la tête 768→11 (Design A) par une tête fusionnée
    (2×embed_dim)→11, entrée = [feat_tuile ; feat_contexte] concaténés.

    Met AUSSI à jour ``groups['head']`` — sinon l'optimizer garderait une référence
    vers les paramètres de l'ANCIENNE tête (jamais utilisée dans le forward une fois
    ``classifier.head`` réassigné) : un bug silencieux où le head entraîné ne serait
    jamais celui réellement appelé.
    """
    import torch.nn as nn
    new_head = nn.Linear(2 * embed_dim, num_classes)
    classifier.head = new_head
    groups = dict(groups)
    groups["head"] = list(new_head.parameters())
    return classifier, groups


def build_ema_teacher(classifier):
    """``--teacher ema_self`` : copie EMA (momentum) du backbone LoRA du student.

    Appelé APRÈS ``classifier.to(device)`` dans ``main()`` — ``copy.deepcopy`` d'un
    module déjà sur CUDA produit une copie CUDA, et l'appariement (teacher_param,
    student_param) référence directement les tenseurs post-``.to(device)`` (évite
    tout risque lié au comportement de ``Module.to()`` sur l'identité des
    ``Parameter`` selon la version de PyTorch).

    Restreint aux paramètres ENTRAÎNABLES du student (LoRA A/B + LayerNorm +
    éventuel full_late) : la base DINOv3 gelée est identique par construction entre
    les deux copies (jamais mise à jour côté student), donc l'EMA dessus serait un
    no-op — exclue pour ne pas itérer sur ~86M paramètres inutilement à chaque step.
    """
    import copy
    teacher_backbone = copy.deepcopy(classifier.backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad_(False)
    teacher_backbone.eval()
    student_named = dict(classifier.backbone.named_parameters())
    pairs = [(tp, student_named[n]) for n, tp in teacher_backbone.named_parameters()
             if student_named[n].requires_grad]
    return teacher_backbone, pairs


def ema_update(pairs, momentum: float) -> None:
    """``teacher = momentum·teacher + (1-momentum)·student``, in-place, sans grad."""
    import torch
    with torch.no_grad():
        for teacher_p, student_p in pairs:
            teacher_p.mul_(momentum).add_(student_p, alpha=1.0 - momentum)


def student_forward(classifier, proj, scale_mlp, x, log_scale):
    """Design A : ``(logits, proj_out, feat_tuile)``. ``log_scale`` : tenseur ``(B, 1)``."""
    feat = classifier._forward_fn(classifier.backbone, x)
    logits = classifier.head(feat)
    se = scale_mlp(log_scale)
    proj_out = proj(_torch_cat(feat, se))
    return logits, proj_out, feat


def student_forward_fused(classifier, proj, scale_mlp, tile, ctx, log_scale):
    """Design B : le student encode TUILE ET CONTEXTE via le MÊME backbone LoRA
    (poids partagés, deux forwards), concatène les deux CLS pour la classification.
    La branche de distillation reste identique à Design A (seule ``feat_tile`` est
    projetée et comparée au teacher) — cf. docstring module §Design B.
    """
    feat_tile = classifier._forward_fn(classifier.backbone, tile)
    feat_ctx = classifier._forward_fn(classifier.backbone, ctx)
    logits = classifier.head(_torch_cat(feat_tile, feat_ctx))
    se = scale_mlp(log_scale)
    proj_out = proj(_torch_cat(feat_tile, se))
    return logits, proj_out, feat_tile, feat_ctx


def _torch_cat(a, b):
    import torch
    return torch.cat([a, b], dim=1)


# ─────────────────────────────────────────────────────────────────── Losses

def build_focal_loss(class_weights, gamma: float):
    import torch.nn.functional as F

    def _loss(logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp().clamp(min=1e-6, max=1.0)
        w = class_weights.to(logits.device)[target]
        return (-w * (1.0 - pt).pow(gamma) * logpt).mean()

    return _loss


def distill_loss(student_proj, teacher_feat, kind: str = "cosine"):
    import torch.nn.functional as F
    if kind == "cosine":
        s = F.normalize(student_proj, dim=1)
        t = F.normalize(teacher_feat, dim=1)
        return (1.0 - (s * t).sum(dim=1)).mean()
    if kind == "mse":
        return F.mse_loss(student_proj, teacher_feat)
    raise ValueError(f"--distill-loss inconnu : {kind!r} (attendu: cosine|mse)")


# ─────────────────────────────────────────────────────────────────── Train / eval

def train_one_epoch(classifier, proj, scale_mlp, teacher, teacher_fwd, loader, optimizer,
                    criterion_cls, lambda_distill, distill_kind, log_scale_value, device,
                    scheduler, grad_clip: float, design: str = "A",
                    ema_pairs=None, ema_momentum: float = 0.999, log_every: int = 50):
    import torch
    classifier.train()
    proj.train()
    scale_mlp.train()
    teacher.eval()  # même en mode EMA : le teacher n'est jamais mis à jour par backprop

    total_cls, total_dist, n = 0.0, 0.0, 0
    t0 = time.time()
    log_scale = torch.full((loader.batch_size, 1), log_scale_value, device=device)
    for i, (tile, ctx, y) in enumerate(loader):
        tile = tile.to(device, non_blocking=True)
        ctx = ctx.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if tile.shape[0] != log_scale.shape[0]:
            log_scale = torch.full((tile.shape[0], 1), log_scale_value, device=device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            teacher_feat = teacher_fwd(teacher, ctx).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            if design == "B":
                logits, proj_out, _ft, _fc = student_forward_fused(
                    classifier, proj, scale_mlp, tile, ctx, log_scale)
            else:
                logits, proj_out, _feat = student_forward(classifier, proj, scale_mlp, tile, log_scale)
            loss_cls = criterion_cls(logits, y)
            loss_dist = distill_loss(proj_out.float(), teacher_feat, distill_kind)
            loss = loss_cls + lambda_distill * loss_dist
        loss.backward()
        if grad_clip > 0:
            params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        if ema_pairs is not None:
            ema_update(ema_pairs, ema_momentum)
        if scheduler is not None:
            scheduler.step()

        total_cls += loss_cls.item()
        total_dist += loss_dist.item()
        n += 1
        if log_every and i % log_every == 0:
            print(f"    batch {i:>5}/{len(loader)} loss_cls={loss_cls.item():.4f} "
                  f"loss_distill={loss_dist.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    return total_cls / max(1, n), total_dist / max(1, n)


def _predict_tile_only(classifier, loader, device):
    """Design A : validation sans contexte (comme à l'inférence)."""
    import torch
    classifier.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feat = classifier._forward_fn(classifier.backbone, x)
                logits = classifier.head(feat)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(labels)


def _predict_fused(classifier, loader, device):
    """Design B : validation avec tuile+contexte (comme à l'inférence sous B)."""
    import torch
    classifier.eval()
    preds, labels = [], []
    with torch.no_grad():
        for tile, ctx, y in loader:
            tile = tile.to(device, non_blocking=True)
            ctx = ctx.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feat_tile = classifier._forward_fn(classifier.backbone, tile)
                feat_ctx = classifier._forward_fn(classifier.backbone, ctx)
                logits = classifier.head(_torch_cat(feat_tile, feat_ctx))
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(np.asarray(y))
    return np.concatenate(preds), np.concatenate(labels)


# ─────────────────────────────────────────────────────────────────── Probe (extraction + LR)

def _extract_fused_embeddings(model_key, ckpt_path, cfg, context_dir, splits=("train", "val", "test")):
    """Design B : extrait les features FUSIONNÉES (tuile+contexte concat., 2×embed_dim)
    pour la sonde canonique. Recharge le backbone via ``load_finetuned_ssl_backbone``
    (IDENTIQUE à Design A — le merge LoRA ne dépend pas du design), mais forward
    TUILE ET CONTEXTE pour chaque split — contrairement à Design A qui n'a besoin du
    contexte qu'au train, Design B en a besoin partout (cf. docstring module).
    Retourne des labels DÉJÀ en 11cls (contrairement à ``_extract_backbone_embeddings``
    qui renvoie du 12cls brut) — ne PAS appeler ``_apply_11cls_remap`` sur le résultat.
    """
    import torch
    from src.models import load_finetuned_ssl_backbone
    from src.utils import get_device

    device = get_device()
    backbone, forward_fn, _dim, _norm_key = load_finetuned_ssl_backbone(model_key, None, ckpt_path)
    backbone = backbone.to(device).eval()

    out = {}
    for s in splits:
        loader, labels_11 = _make_eval_loader_with_context(cfg, s, context_dir,
                                                            cfg.data.batch_size, cfg.data.num_workers)
        embs = []
        with torch.no_grad():
            for tile, ctx, _y in loader:
                tile = tile.to(device, non_blocking=True)
                ctx = ctx.to(device, non_blocking=True)
                f_tile = forward_fn(backbone, tile)
                f_ctx = forward_fn(backbone, ctx)
                fused = torch.cat([f_tile, f_ctx], dim=1)
                embs.append(fused.detach().cpu().to(torch.float16).numpy())
        E = np.concatenate(embs, axis=0)
        out[s] = (E, labels_11)
        print(f"  [extract-fused] {s}: {E.shape}", flush=True)
    return out


def _run_probe_with_balanced_acc(feats_val_test: dict, feats_train: tuple, C_grid, max_iter: int) -> dict:
    """Variante de ``datacurve_one_run._run_probe_and_metrics`` (MÊME méthodologie :
    StandardScaler, ``make_canonical_lr``, sélection de C sur val f1_macro_pres) qui
    AJOUTE la balanced accuracy demandée par la mission (métrique primaire secondaire,
    moins sensible au déséquilibre extrême que f1_macro_pres seul). Dupliquée plutôt
    qu'importée+étendue car la fonction source ne renvoie pas les prédictions brutes.
    Agnostique à la dimensionnalité des features (768 Design A, 1536 Design B) —
    StandardScaler + LogisticRegression n'en dépendent pas.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
    from src.utils import make_canonical_lr
    from datacurve_one_run import CLASS_NAMES_11, LABELS_8CLS

    E_tr, L_tr = feats_train
    E_va, L_va = feats_val_test["val"]
    E_te, L_te = feats_val_test["test"]

    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr.astype(np.float32))
    X_va = sc.transform(E_va.astype(np.float32))
    X_te = sc.transform(E_te.astype(np.float32))

    best_c, best_f1v = C_grid[0], -1.0
    for c in C_grid:
        clf = make_canonical_lr(C=c, max_iter=max_iter)
        clf.fit(X_tr, L_tr)
        f1v = f1_score(L_va, clf.predict(X_va), average="macro", labels=list(range(11)), zero_division=0)
        if f1v > best_f1v:
            best_c, best_f1v = c, f1v

    clf_final = make_canonical_lr(C=best_c, max_iter=max_iter)
    clf_final.fit(X_tr, L_tr)
    preds_va = clf_final.predict(X_va)
    preds_te = clf_final.predict(X_te)

    present_va = sorted({int(v) for v in L_va})
    present_te = sorted({int(v) for v in L_te})

    f1_all = f1_score(L_te, preds_te, average=None, zero_division=0, labels=list(range(11)))
    return {
        "best_C": best_c,
        "f1_macro_pres_val": float(f1_score(L_va, preds_va, average="macro", labels=present_va, zero_division=0)),
        "f1_macro_pres_test": float(f1_score(L_te, preds_te, average="macro", labels=present_te, zero_division=0)),
        "f1_macro_8cls_test": float(f1_score(L_te, preds_te, average="macro", labels=LABELS_8CLS, zero_division=0)),
        "accuracy_test": float(accuracy_score(L_te, preds_te)),
        "balanced_accuracy_val": float(balanced_accuracy_score(L_va, preds_va)),
        "balanced_accuracy_test": float(balanced_accuracy_score(L_te, preds_te)),
        "f1_per_class_test": {CLASS_NAMES_11[i]: float(f1_all[i]) for i in range(11)},
    }


# ─────────────────────────────────────────────────────────────────── Main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--student", default=None, help="override cfg.model.name (défaut: valeur du config)")
    ap.add_argument("--teacher", default="dinov3_vitl16_lvd",
                    help="nom d'un backbone frozen (ex. dinov3_vitl16_lvd) OU 'ema_self' "
                         "pour un momentum-teacher = copie EMA du backbone du student "
                         "(pas de modèle externe, cf. docstring module §EMA self-teacher).")
    ap.add_argument("--ema-momentum", type=float, default=0.999,
                    help="momentum EMA, utilisé seulement si --teacher ema_self.")
    ap.add_argument("--lora-rank", type=int, default=None, help="override cfg.lora.r (alpha=2r, convention du dépôt)")
    ap.add_argument("--lora-blocks", default=None, help="override cfg.lora.lora_block_indices, ex. '6,7,8,9,10,11'")
    ap.add_argument("--context-size", type=int, required=True, help="doit correspondre à un dossier context_<size>/ produit par context_crop.py")
    ap.add_argument("--context-dir", required=True,
                    help="chemin du dossier context_<context-size>/ (PAS son parent). Design A : "
                         "besoin du contexte train seulement. Design B : besoin AUSSI de val/test "
                         "dans ce MÊME dossier (relancer context_crop.py sur val.csv/test.csv).")
    ap.add_argument("--design", choices=["A", "B"], default="A")
    ap.add_argument("--lambda-distill", type=float, default=1.0)
    ap.add_argument("--distill-loss", choices=["cosine", "mse"], default="cosine")
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=None, help="override cfg.data.batch_size")
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()

    import torch
    from src.config import load_config
    from src import utils, engine
    from src.losses import build_class_weights
    from src.models import build_frozen_extractor, build_model
    from datacurve_one_run import _extract_backbone_embeddings, _apply_11cls_remap

    cfg = load_config(args.config)
    if args.student:
        cfg.model.name = args.student
    if args.lora_rank is not None:
        cfg.lora.r = args.lora_rank
        cfg.lora.alpha = float(2 * args.lora_rank)  # convention du dépôt (alpha=2r), cf. AGENTS.md
    if args.lora_blocks is not None:
        cfg.lora.lora_block_indices = [int(b) for b in args.lora_blocks.split(",")]
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    utils.set_seed(args.seed, deterministic=cfg.train.deterministic)
    device = utils.get_device()
    if device.type != "cuda":
        print("[WARN] pas de GPU détecté — ce script est prévu pour Narval (A100). "
              "Exécution possible mais très lente sur CPU (sanity-check seulement).")

    # Tag : model_ctxdistill_d{A|B}_t{tag-teacher}_ctx{size}_r{r}a{alpha}_frac100_seed{n}.
    # d/t DOIVENT être dans le tag : R1 (dA_tL_ctx1024) et l'EMA self-teacher
    # (dA_tEMA_ctx1024) partagent SINON le même {model,ctx,r,alpha,frac,seed} → même
    # run_dir → metrics.json écrasé silencieusement (classe de bug déjà vécue sur ce
    # dépôt, cf. datacurve_one_run.py:229, perte d'un run ~5h GPU en 2026-07-27/28).
    tag = (f"{cfg.model.name}_ctxdistill_d{args.design}_{_teacher_tag(args.teacher)}"
           f"_ctx{args.context_size}_r{cfg.lora.r}a{int(cfg.lora.alpha)}_frac100_seed{args.seed}")
    run_dir = _os.path.join(args.out_dir, "runs", tag)
    _os.makedirs(run_dir, exist_ok=True)
    done_path = _os.path.join(run_dir, "done")
    if args.skip_if_done and _os.path.exists(done_path):
        print(f"[context_distill] {tag} déjà complété — skip.")
        return

    ckpt_dir = _os.path.join(args.out_dir, "checkpoints")
    _os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = _os.path.join(ckpt_dir, f"{tag}_best.pth")
    last_ckpt = _os.path.join(ckpt_dir, f"{tag}_last.pth")

    print(f"[context_distill] tag={tag}")
    print(f"[context_distill] student={cfg.model.name} teacher={args.teacher} "
          f"lora r={cfg.lora.r} alpha={cfg.lora.alpha} blocks={cfg.lora.lora_block_indices} "
          f"context_size={args.context_size} lambda={args.lambda_distill} "
          f"focal_gamma={args.focal_gamma} design={args.design}")

    # ── Student : backbone LoRA construit D'ABORD (fournit embed_dim, requis avant
    # de dimensionner proj/scale_mlp ET avant de savoir teacher_dim si EMA) ────────
    classifier, groups = build_model(cfg.model.name, "lora", cfg.model.num_classes, lora=cfg.lora)
    embed_dim = classifier.head.in_features
    if args.design == "B":
        classifier, groups = _replace_head_for_fusion(classifier, groups, embed_dim, cfg.model.num_classes)
    classifier = classifier.to(device)

    # ── Teacher : externe gelé OU copie EMA du student (APRÈS classifier.to(device),
    # cf. docstring build_ema_teacher — évite tout risque lié à Module.to()) ───────
    if args.teacher == "ema_self":
        teacher_dim = embed_dim
        teacher_norm_key = cfg.model.norm
        teacher, ema_pairs = build_ema_teacher(classifier)
        teacher_fwd = classifier._forward_fn
    else:
        teacher, teacher_fwd, teacher_dim, teacher_norm_key = build_frozen_extractor(args.teacher)
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        ema_pairs = None

    # ── Tête de projection (branche distillation), dimensionnée sur teacher_dim ──
    proj, scale_mlp = build_projection_head(embed_dim, teacher_dim)
    proj = proj.to(device)
    scale_mlp = scale_mlp.to(device)
    groups = dict(groups)
    groups["proj"] = list(proj.parameters()) + list(scale_mlp.parameters())

    all_params = list(classifier.parameters()) + list(proj.parameters()) + list(scale_mlp.parameters())
    n_total = sum(p.numel() for p in all_params)
    n_train = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"[context_distill] student params total={n_total:,} entraînables={n_train:,} "
          f"({100 * n_train / max(1, n_total):.1f}%)")

    # ── Data ──────────────────────────────────────────────────────────────
    log_scale_value = math.log(args.context_size / cfg.data.image_size)
    train_loader, train_labels_11 = _make_train_loader(
        cfg, args.context_dir, teacher_norm_key, cfg.data.batch_size, cfg.data.num_workers, args.seed)
    if args.design == "B":
        val_loader, _ = _make_eval_loader_with_context(
            cfg, "val", args.context_dir, cfg.data.batch_size, cfg.data.num_workers)
    else:
        val_loader, _ = _make_eval_loader(cfg, "val", cfg.data.batch_size, cfg.data.num_workers)
    print(f"[context_distill] train={len(train_loader.dataset)} tuiles (log_scale={log_scale_value:.4f})")

    class_weights = build_class_weights(train_labels_11, cfg.model.num_classes)
    criterion_cls = build_focal_loss(class_weights, args.focal_gamma)

    optimizer = engine.build_optimizer(groups, cfg)
    scheduler = engine.build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    grad_clip = float(cfg.train.grad_clip)
    if cfg.train.amp_dtype != "bfloat16":
        raise ValueError("context_distill.py suppose amp_dtype=bfloat16 (convention A100 du dépôt, "
                         "pas de GradScaler implémenté pour float16 dans ce script) — "
                         f"cfg.train.amp_dtype={cfg.train.amp_dtype!r}.")

    # ── Boucle d'entraînement ────────────────────────────────────────────────
    history = {"train_loss_cls": [], "train_loss_distill": [], "val_f1_macro_pres": [],
              "val_balanced_accuracy": [], "val_accuracy": [], "lr": []}
    best_metric, patience_left = -1.0, cfg.train.patience
    metric_name = cfg.train.early_stop_metric  # ex. f1_macro_pres (convention du dépôt)

    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
    for epoch in range(cfg.train.epochs):
        lrs = [g["lr"] for g in optimizer.param_groups]
        loss_cls, loss_dist = train_one_epoch(
            classifier, proj, scale_mlp, teacher, teacher_fwd, train_loader, optimizer, criterion_cls,
            args.lambda_distill, args.distill_loss, log_scale_value, device, scheduler, grad_clip,
            design=args.design, ema_pairs=ema_pairs, ema_momentum=args.ema_momentum)

        if args.design == "B":
            preds, labels = _predict_fused(classifier, val_loader, device)
        else:
            preds, labels = _predict_tile_only(classifier, val_loader, device)
        present = sorted({int(v) for v in labels})
        val_f1_pres = float(f1_score(labels, preds, average="macro", labels=present, zero_division=0))
        val_bacc = float(balanced_accuracy_score(labels, preds))
        val_acc = float(accuracy_score(labels, preds))
        val_metrics = {"f1_macro_pres": val_f1_pres, "balanced_accuracy": val_bacc, "accuracy": val_acc}

        history["train_loss_cls"].append(loss_cls)
        history["train_loss_distill"].append(loss_dist)
        history["val_f1_macro_pres"].append(val_f1_pres)
        history["val_balanced_accuracy"].append(val_bacc)
        history["val_accuracy"].append(val_acc)
        history["lr"].append(lrs)

        cur = val_metrics[metric_name if metric_name in val_metrics else "f1_macro_pres"]
        improved = cur > best_metric
        print(f"[{tag}] epoch {epoch + 1}/{cfg.train.epochs} loss_cls={loss_cls:.4f} "
              f"loss_distill={loss_dist:.4f} valF1pres={val_f1_pres:.4f} "
              f"valBAcc={val_bacc:.4f}" + ("  *best*" if improved else ""), flush=True)

        if improved:
            best_metric = cur
            patience_left = cfg.train.patience
            torch.save({
                "epoch": epoch,
                "tag": tag,
                "model_state_dict": classifier.state_dict(),  # backbone.*/head.* pur, cf. build_projection_head
                "proj_state_dict": proj.state_dict(),
                "scale_mlp_state_dict": scale_mlp.state_dict(),
                "best_metric": best_metric, "metric_name": metric_name, "history": history,
                "context_size": args.context_size, "design": args.design, "teacher": args.teacher,
            }, best_ckpt)
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"[{tag}] early stopping à l'epoch {epoch + 1} "
                  f"({metric_name} stagnant depuis {cfg.train.patience} epochs)")
            break

    torch.save({"tag": tag, "model_state_dict": classifier.state_dict(),
               "proj_state_dict": proj.state_dict(),
               "scale_mlp_state_dict": scale_mlp.state_dict(),
               "history": history}, last_ckpt)
    if not _os.path.exists(best_ckpt):
        best_ckpt = last_ckpt

    # ── Extraction + sonde canonique ─────────────────────────────────────────
    if args.design == "B":
        print("\n[context_distill] extraction + sonde canonique (Design B : tuile+contexte fusionnés)")
        feats_all = _extract_fused_embeddings(cfg.model.name, best_ckpt, cfg, args.context_dir,
                                              splits=("train", "val", "test"))
        feats_vt = {"val": feats_all["val"], "test": feats_all["test"]}
        feats_tr = feats_all["train"]
    else:
        print("\n[context_distill] extraction + sonde canonique (Design A : tuile 224 seule)")
        feats_vt_12 = _extract_backbone_embeddings(cfg.model.name, best_ckpt, cfg,
                                                   pretrain_checkpoint=None, splits=("val", "test"))
        feats_vt = {s: _apply_11cls_remap(E, L) for s, (E, L) in feats_vt_12.items()}
        feats_tr_12 = _extract_backbone_embeddings(cfg.model.name, best_ckpt, cfg,
                                                   pretrain_checkpoint=None, splits=("train",))
        feats_tr = _apply_11cls_remap(*feats_tr_12["train"])

    metrics = _run_probe_with_balanced_acc(feats_vt, feats_tr, list(cfg.probe.C_grid), cfg.probe.max_iter)
    metrics.update({
        "tag": tag, "student": cfg.model.name, "teacher": args.teacher,
        "ema_momentum": args.ema_momentum if args.teacher == "ema_self" else None,
        "context_size": args.context_size, "design": args.design,
        "lora_r": cfg.lora.r, "lora_alpha": cfg.lora.alpha,
        "lora_block_indices": cfg.lora.lora_block_indices,
        "lambda_distill": args.lambda_distill, "distill_loss": args.distill_loss,
        "focal_gamma": args.focal_gamma, "seed": args.seed,
        "n_train_tiles": len(feats_tr[1]), "best_epoch_val_metric": best_metric,
        "early_stop_metric": metric_name, "schema": "11cls_no_rhol", "split": "spatial",
    })
    print(f"\n  ✓ f1_macro_pres_test={metrics['f1_macro_pres_test']:.4f}  "
          f"balanced_accuracy_test={metrics['balanced_accuracy_test']:.4f}  best_C={metrics['best_C']}")

    metrics_path = _os.path.join(run_dir, "metrics.json")
    # Garde-fou anti-collision (même logique que datacurve_one_run.py:411-427) :
    # run_dir est scopé par tag (design+teacher inclus depuis ce commit), donc un
    # metrics.json déjà présent ici ne devrait JAMAIS appartenir à un autre run —
    # sécurité si jamais un run_dir est réutilisé manuellement malgré tout.
    if _os.path.exists(metrics_path):
        with open(metrics_path) as f:
            existing_tag = json.load(f).get("tag")
        if existing_tag is not None and existing_tag != tag:
            raise RuntimeError(
                f"[context_distill] COLLISION DE CHEMIN : {metrics_path} contient déjà un "
                f"résultat pour tag={existing_tag!r}, différent du run courant ({tag!r}). "
                "Écriture refusée pour ne pas écraser un résultat existant — vérifier --out-dir.")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(done_path, "w") as f:
        f.write(f"completed {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    print(f"[context_distill] → {metrics_path}")


if __name__ == "__main__":
    main()
