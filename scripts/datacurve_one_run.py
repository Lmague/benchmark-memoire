#!/usr/bin/env python3
"""Data curve — un run (proportion × seed) : train → extract → probe → métriques.

Utilisé directement par slurm_datacurve.sh. Peut aussi être appelé manuellement :

    python scripts/datacurve_one_run.py \
        --config configs/vitb16_fulft_datacurve.yaml \
        --fraction 0.01 --seed 0 \
        --out-dir outputs/datacurve

Sorties (sous --out-dir) :
  runs/frac01_seed0/metrics.json   — chiffres du run
  runs/frac01_seed0/done           — sentinel de complétion (skip si présent)

Embeddings : écrits dans {emb_base_dir}/vitb16_fulft_frac01_seed0/ (val + test + labels)
où emb_base_dir = cfg.paths.emb_dir par défaut ou --emb-dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Classes exclues des deux métriques
# CLASS_NAMES = ["ALDE","ARCA","BIRC","DRYI","LICH","MOSS","PETF","RHOL","RUBC","SEDG","TUSS","WILL"]
# Indices :         0     1     2     3     4     5     6     7     8     9    10    11
# f1_macro_pres (11 cls) : RHOL absente du test → exclue automatiquement par f1_macro_pres
# f1_macro_8cls  (8 cls) : ARCA(1) DRYI(3) RHOL(7) RUBC(8) explicitement exclus
LABELS_8CLS = [0, 2, 4, 5, 6, 9, 10, 11]  # ALDE BIRC LICH MOSS PETF SEDG TUSS WILL


def _frac_tag(fraction: float, seed: int) -> str:
    pct = int(round(fraction * 100))
    return f"frac{pct:03d}_seed{seed}"


def _extract_backbone_embeddings(ckpt_path: str, cfg, splits=("val", "test")) -> dict:
    """Charge le backbone fine-tuné depuis ckpt_path, extrait les features pour splits.

    Retourne {split: (embeddings float16, labels int64)}.
    """
    import torch
    from src.models import _load_finetuned_backbone
    from src.data import build_transforms, ArcticTVCDataset
    from src.utils import get_normalization, get_device
    from torch.utils.data import DataLoader

    device = get_device()
    backbone = _load_finetuned_backbone("vit_base_patch16_224", ckpt_path, 768)
    backbone = backbone.to(device).eval()

    mean, std = get_normalization("imagenet")
    out: dict = {}
    for s in splits:
        tf = build_transforms("eval", mean, std, cfg.data.image_size)
        ds = ArcticTVCDataset(os.path.join(cfg.paths.csv_dir, f"{s}.csv"),
                              cfg.paths.tiles_dir, tf)
        loader = DataLoader(ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)
        embs, lbls = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                f = backbone(x)
                embs.append(f.detach().cpu().to(torch.float16).numpy())
                lbls.append(y.numpy())
        E = np.concatenate(embs, axis=0)
        L = np.concatenate(lbls, axis=0)
        out[s] = (E, L)
        print(f"  [extract] {s}: {E.shape}", flush=True)
    return out


def _run_probe_and_metrics(feats_val_test: dict, feats_train: tuple,
                           C_grid=(0.01, 0.1, 1.0, 10.0),
                           max_iter: int = 2000) -> dict:
    """Sonde linéaire lbfgs sur features fine-tunées ; retourne métriques sur test.

    feats_val_test = {"val": (E, L), "test": (E, L)} (déjà extraits).
    feats_train    = (E_train, L_train) — sous-ensemble effectif d'entraînement.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    from src.utils import make_canonical_lr, CLASS_NAMES

    E_tr, L_tr = feats_train
    E_va, L_va = feats_val_test["val"]
    E_te, L_te = feats_val_test["test"]

    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr.astype(np.float32))
    X_va = sc.transform(E_va.astype(np.float32))
    X_te = sc.transform(E_te.astype(np.float32))

    # Sélection du C sur val (f1_macro_all sur les 12 classes, convention benchmark)
    best_c, best_f1v = C_grid[0], -1.0
    for c in C_grid:
        clf = make_canonical_lr(C=c, max_iter=max_iter)
        clf.fit(X_tr, L_tr)
        f1v = f1_score(L_va, clf.predict(X_va), average="macro",
                       labels=list(range(12)), zero_division=0)
        if f1v > best_f1v:
            best_c, best_f1v = c, f1v

    # Fit final avec best_c
    clf_final = make_canonical_lr(C=best_c, max_iter=max_iter)
    clf_final.fit(X_tr, L_tr)
    preds_va = clf_final.predict(X_va)
    preds_te = clf_final.predict(X_te)

    # Métriques val
    present_va = sorted({int(v) for v in L_va})
    f1_pres_va = float(f1_score(L_va, preds_va, average="macro",
                                labels=present_va, zero_division=0))

    # Métriques test primaire : f1_macro_pres (classes présentes dans y_true test)
    present_te = sorted({int(v) for v in L_te})
    f1_pres_te = float(f1_score(L_te, preds_te, average="macro",
                                labels=present_te, zero_division=0))

    # Métriques test secondaire : f1_macro_8cls
    f1_8cls_te = float(f1_score(L_te, preds_te, average="macro",
                                labels=LABELS_8CLS, zero_division=0))

    acc_te = float(accuracy_score(L_te, preds_te))

    # F1 par classe sur test (pour per_class_curve Tier 2)
    f1_all = f1_score(L_te, preds_te, average=None, zero_division=0,
                      labels=list(range(12)))
    f1_per_class = {CLASS_NAMES[i]: float(f1_all[i]) for i in range(12)}

    return {
        "best_C": best_c,
        "f1_macro_pres_val": f1_pres_va,
        "f1_macro_pres_test": f1_pres_te,
        "f1_macro_8cls_test": f1_8cls_te,
        "accuracy_test": acc_te,
        "f1_per_class_test": f1_per_class,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vitb16_fulft_datacurve.yaml")
    ap.add_argument("--fraction", type=float, required=True,
                    help="fraction du train set (ex. 0.01)")
    ap.add_argument("--seed", type=int, required=True,
                    help="seed du sous-échantillonnage (0, 1, 2)")
    ap.add_argument("--out-dir", default="outputs/datacurve",
                    help="répertoire de sortie (métriques + checkpoints locaux)")
    ap.add_argument("--emb-dir", default=None,
                    help="répertoire de base pour les embeddings (défaut: cfg.paths.emb_dir)")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="saute le run si le fichier 'done' existe déjà")
    args = ap.parse_args()

    # Résolution des chemins
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, code_dir)

    from src.config import load_config
    from src import utils

    cfg = load_config(os.path.join(code_dir, args.config))

    tag = _frac_tag(args.fraction, args.seed)
    run_dir = os.path.join(args.out_dir, "runs", tag)
    os.makedirs(run_dir, exist_ok=True)
    done_path = os.path.join(run_dir, "done")

    if args.skip_if_done and os.path.exists(done_path):
        print(f"[datacurve] {tag} déjà complété — skip.", flush=True)
        return

    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"[datacurve] START  fraction={args.fraction:.2%}  seed={args.seed}  tag={tag}",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    # Chemins checkpoints (sous out-dir pour ne pas polluer le ckpt_dir principal)
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    pct = int(round(args.fraction * 100))
    ckpt_tag = f"vitb16_full_frac{pct:03d}_seed{args.seed}"
    best_ckpt = os.path.join(ckpt_dir, f"{ckpt_tag}_best.pth")

    # ── 1. ENTRAÎNEMENT ─────────────────────────────────────────────────────────
    print("[datacurve] 1/3 — Entraînement", flush=True)
    # On délègue à train.py pour réutiliser exactement la même recette.
    # Les chemins de checkpoint sont redirigés via cfg.paths.ckpt_dir.
    # On patche temporairement le ckpt_dir dans la config.
    cfg.paths.ckpt_dir = ckpt_dir

    utils.set_seed(cfg.train.seed)
    device = utils.get_device()

    from train import stratified_subsample
    from src.data import make_loaders
    from src.losses import build_class_weights, build_criterion
    from src.models import build_model
    from src import engine

    # Sous-échantillonnage stratifié
    _, all_train_labels = utils.read_split_csv(
        os.path.join(cfg.paths.csv_dir, "train.csv"))
    train_indices = stratified_subsample(all_train_labels, args.fraction, args.seed)
    sub_labels = all_train_labels[train_indices]
    n_train = len(train_indices)

    print(f"  sous-échantillon : {n_train} tuiles ({args.fraction:.2%})", flush=True)
    counts_str = {utils.CLASS_NAMES[c]: int((sub_labels == c).sum())
                  for c in range(cfg.model.num_classes)}
    print(f"  effectifs : {counts_str}", flush=True)

    # Alerte classes rares à 1%
    EXCL = {utils.CLASS_NAMES[c]: int((sub_labels == c).sum())
            for c in [1, 3, 7, 8]}  # ARCA DRYI RHOL RUBC
    print(f"  [Tier3] classes rares dans le sous-échantillon : {EXCL}", flush=True)

    model, groups = build_model(cfg.model.name, cfg.regime, cfg.model.num_classes)
    model = model.to(device)
    criterion = build_criterion(sub_labels, cfg.model.num_classes, device)
    loaders = make_loaders(cfg, train_aug=True, train_indices=train_indices)

    results = engine.fit(cfg, model, groups, loaders, criterion, device,
                         tag_override=ckpt_tag)

    best_epoch = int(np.argmax(results["history"]["val_f1_macro"])) + 1
    best_val_f1 = float(max(results["history"]["val_f1_macro"]))
    print(f"  best_epoch={best_epoch}  val_f1_macro={best_val_f1:.4f}", flush=True)

    if not os.path.exists(best_ckpt):
        print(f"  [WARN] checkpoint best non trouvé : {best_ckpt}", flush=True)
        # Utiliser last si best absent (run très court)
        last_ckpt = os.path.join(ckpt_dir, f"{ckpt_tag}_last.pth")
        best_ckpt = last_ckpt

    # ── 2. EXTRACTION DES EMBEDDINGS ────────────────────────────────────────────
    print("\n[datacurve] 2/3 — Extraction des embeddings", flush=True)

    emb_base = args.emb_dir or cfg.paths.emb_dir
    emb_key = f"vitb16_fulft_frac{pct:03d}_seed{args.seed}"
    emb_dir_run = os.path.join(emb_base, emb_key)
    os.makedirs(emb_dir_run, exist_ok=True)

    val_emb_path = os.path.join(emb_dir_run, "val.npy")
    val_lbl_path = os.path.join(emb_dir_run, "val_labels.npy")
    test_emb_path = os.path.join(emb_dir_run, "test.npy")
    test_lbl_path = os.path.join(emb_dir_run, "test_labels.npy")

    feats_vt = _extract_backbone_embeddings(best_ckpt, cfg, splits=("val", "test"))
    np.save(val_emb_path, feats_vt["val"][0])
    np.save(val_lbl_path, feats_vt["val"][1])
    np.save(test_emb_path, feats_vt["test"][0])
    np.save(test_lbl_path, feats_vt["test"][1])
    print(f"  embeddings sauvés → {emb_dir_run}", flush=True)

    # Embeddings d'entraînement (sous-ensemble) — nécessaires pour le fit de la sonde
    print("  extraction train (sous-ensemble)...", flush=True)
    feats_tr = _extract_backbone_embeddings(best_ckpt, cfg, splits=("train",))
    E_tr_full, L_tr_full = feats_tr["train"]
    # On ne garde que le sous-ensemble utilisé pour l'entraînement
    E_tr = E_tr_full[train_indices]
    L_tr = L_tr_full[train_indices]

    # ── 3. SONDE LINÉAIRE + MÉTRIQUES ───────────────────────────────────────────
    print("\n[datacurve] 3/3 — Sonde linéaire (lbfgs)", flush=True)
    metrics = _run_probe_and_metrics(
        feats_val_test=feats_vt,
        feats_train=(E_tr, L_tr),
        C_grid=list(cfg.probe.C_grid),
        max_iter=cfg.probe.max_iter,
    )
    metrics["best_epoch"] = best_epoch
    metrics["n_train_tiles"] = n_train
    metrics["fraction"] = args.fraction
    metrics["seed"] = args.seed
    metrics["ckpt_tag"] = ckpt_tag
    metrics["emb_dir"] = emb_dir_run

    print(f"\n  ✓ f1_macro_pres_test = {metrics['f1_macro_pres_test']:.4f}"
          f"  f1_macro_8cls_test = {metrics['f1_macro_8cls_test']:.4f}"
          f"  best_C = {metrics['best_C']}", flush=True)

    # Sauvegarde
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path}", flush=True)

    # Sentinel de complétion
    with open(done_path, "w") as f:
        f.write(f"completed {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    elapsed = time.time() - t0
    print(f"\n[datacurve] DONE {tag}  ({elapsed/60:.1f} min)\n", flush=True)


if __name__ == "__main__":
    main()
