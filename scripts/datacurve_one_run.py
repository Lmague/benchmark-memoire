#!/usr/bin/env python3
"""Data curve — un run (proportion × seed) : train → extract → probe → métriques.

Pipeline Q5 : schéma 11 classes (RHOL exclu du train, labels remappés 0-10).
Utilisé directement par slurm_datacurve.sh. Peut aussi être appelé manuellement :

    python scripts/datacurve_one_run.py \\
        --config configs/vitb16_fulft_datacurve.yaml \\
        --fraction 0.01 --seed 0 \\
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
# Schéma 11 classes Q5 (RHOL exclue)
# 12-class : ALDE(0) ARCA(1) BIRC(2) DRYI(3) LICH(4) MOSS(5) PETF(6) RHOL(7) RUBC(8) SEDG(9) TUSS(10) WILL(11)
# 11-class : ALDE(0) ARCA(1) BIRC(2) DRYI(3) LICH(4) MOSS(5) PETF(6)         RUBC(7) SEDG(8) TUSS(9)  WILL(10)
_RHOL_IDX = 7
LABEL_REMAP_12TO11: dict[int, int] = {
    **{i: i for i in range(_RHOL_IDX)},
    **{i: i - 1 for i in range(_RHOL_IDX + 1, 12)},
}
CLASS_NAMES_11 = ["ALDE", "ARCA", "BIRC", "DRYI", "LICH", "MOSS", "PETF",
                   "RUBC", "SEDG", "TUSS", "WILL"]

# 8-class diagnostic (hors ARCA=1, DRYI=3, RUBC=7 en 11-class) — SEDG=8, TUSS=9, WILL=10
LABELS_8CLS = [0, 2, 4, 5, 6, 8, 9, 10]   # ALDE BIRC LICH MOSS PETF SEDG TUSS WILL (11-class)

# Tag court par régime → injecté dans ckpt_tag / emb_key pour éviter toute collision entre
# régimes sous un même out-dir (sota_screening : full/mhsa/explora/scratch). 'full' inchangé
# (rétro-compat datacurve Q5). Régime inconnu → son nom brut.
_REGIME_TAG = {"full": "full", "mhsa": "mhsa", "explora_like": "explora", "scratch": "scratch"}


def _apply_11cls_remap(E: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filtre RHOL (label 7) et remappe les labels restants vers le schéma 11-classes."""
    mask = L != _RHOL_IDX
    E_out = E[mask]
    L_raw = L[mask]
    L_out = np.array([LABEL_REMAP_12TO11[int(l)] for l in L_raw], dtype=np.int64)
    return E_out, L_out


def _frac_tag(fraction: float, seed: int) -> str:
    pct = int(round(fraction * 100))
    return f"frac{pct:03d}_seed{seed}"


def _extract_backbone_embeddings(ckpt_path: str, cfg, splits=("val", "test")) -> dict:
    """Charge le backbone fine-tuné depuis ckpt_path, extrait les features pour splits.

    Retourne {split: (embeddings float16, labels int64)} — labels encore en 12-class ici.
    Le remapping 11-class est appliqué en aval par _apply_11cls_remap.
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

    feats_val_test = {"val": (E, L), "test": (E, L)} — labels déjà en 11-class.
    feats_train    = (E_train, L_train) — sous-ensemble effectif, labels 11-class.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score
    from src.utils import make_canonical_lr

    E_tr, L_tr = feats_train
    E_va, L_va = feats_val_test["val"]
    E_te, L_te = feats_val_test["test"]

    sc = StandardScaler()
    X_tr = sc.fit_transform(E_tr.astype(np.float32))
    X_va = sc.transform(E_va.astype(np.float32))
    X_te = sc.transform(E_te.astype(np.float32))

    # Sélection du C sur val (f1_macro_all sur les 11 classes Q5)
    best_c, best_f1v = C_grid[0], -1.0
    for c in C_grid:
        clf = make_canonical_lr(C=c, max_iter=max_iter)
        clf.fit(X_tr, L_tr)
        f1v = f1_score(L_va, clf.predict(X_va), average="macro",
                       labels=list(range(11)), zero_division=0)
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

    # Métriques test secondaire : f1_macro_8cls (8 classes fiables en 11-class)
    f1_8cls_te = float(f1_score(L_te, preds_te, average="macro",
                                labels=LABELS_8CLS, zero_division=0))

    acc_te = float(accuracy_score(L_te, preds_te))

    # F1 par classe sur test (pour per_class_curve Tier 2)
    f1_all = f1_score(L_te, preds_te, average=None, zero_division=0,
                      labels=list(range(11)))
    f1_per_class = {CLASS_NAMES_11[i]: float(f1_all[i]) for i in range(11)}

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
    regime_tag = _REGIME_TAG.get(cfg.regime, cfg.regime)
    ckpt_tag = f"vitb16_{regime_tag}_frac{pct:03d}_seed{args.seed}"
    best_ckpt = os.path.join(ckpt_dir, f"{ckpt_tag}_best.pth")

    # ── 1. ENTRAÎNEMENT ─────────────────────────────────────────────────────────
    print("[datacurve] 1/3 — Entraînement (11 classes, RHOL exclue)", flush=True)
    cfg.paths.ckpt_dir = ckpt_dir

    utils.set_seed(args.seed, deterministic=cfg.train.deterministic)
    device = utils.get_device()

    from src.utils import stratified_subsample, read_split_csv
    from src.data import make_loaders
    from src.losses import build_criterion
    from src.models import build_model
    from src import engine

    # ── Filtrage RHOL et remapping labels 12→11 ─────────────────────────────────
    _, all_train_labels_12 = read_split_csv(
        os.path.join(cfg.paths.csv_dir, "train.csv"))
    non_rhol_orig_idx = np.where(all_train_labels_12 != _RHOL_IDX)[0]
    all_train_labels_11 = np.array(
        [LABEL_REMAP_12TO11[int(l)] for l in all_train_labels_12[non_rhol_orig_idx]],
        dtype=np.int64)

    # Sous-échantillonnage stratifié dans l'espace 11-classes
    subsample_in_11 = stratified_subsample(all_train_labels_11, args.fraction, args.seed)
    sub_labels = all_train_labels_11[subsample_in_11]

    # Indices dans le CSV original (12-class) pour le sous-ensemble effectif
    orig_train_sub_idx = non_rhol_orig_idx[subsample_in_11]

    # Indices non-RHOL pour val et test (filtrage défensif)
    _, val_labels_12 = read_split_csv(os.path.join(cfg.paths.csv_dir, "val.csv"))
    val_nonrhol_idx = np.where(val_labels_12 != _RHOL_IDX)[0]
    _, test_labels_12 = read_split_csv(os.path.join(cfg.paths.csv_dir, "test.csv"))
    test_nonrhol_idx = np.where(test_labels_12 != _RHOL_IDX)[0]

    n_train = len(orig_train_sub_idx)
    print(f"  sous-échantillon : {n_train} tuiles ({args.fraction:.2%}) — 11 classes, RHOL exclue",
          flush=True)
    counts_str = {CLASS_NAMES_11[c]: int((sub_labels == c).sum()) for c in range(11)}
    print(f"  effectifs : {counts_str}", flush=True)

    # Alerte classes rares à faible fraction (ARCA=1, DRYI=3, RUBC=7 en 11-class)
    rare = {CLASS_NAMES_11[c]: int((sub_labels == c).sum()) for c in [1, 3, 7]}
    print(f"  [Tier3] classes rares dans le sous-échantillon : {rare}", flush=True)

    model, groups = build_model(cfg.model.name, cfg.regime, cfg.model.num_classes,
                                lora=cfg.lora)
    model = model.to(device)
    criterion = build_criterion(sub_labels, cfg.model.num_classes, device)

    loaders = make_loaders(cfg, train_aug=True,
                           indices={
                               "train": orig_train_sub_idx,
                               "val":   val_nonrhol_idx,
                               "test":  test_nonrhol_idx,
                           },
                           label_remap=LABEL_REMAP_12TO11)

    results = engine.fit(cfg, model, groups, loaders, criterion, device,
                         tag_override=ckpt_tag)

    # best_epoch aligné sur la métrique de sélection réelle (val_f1_select si présent,
    # sinon val_f1_macro = ancien comportement) — cohérent avec le best ckpt sauvé.
    sel_hist = results["history"].get("val_f1_select") or results["history"]["val_f1_macro"]
    best_epoch = int(np.argmax(sel_hist)) + 1
    best_val_f1 = float(max(sel_hist))
    print(f"  best_epoch={best_epoch}  val_select={best_val_f1:.4f}", flush=True)

    if not os.path.exists(best_ckpt):
        print(f"  [WARN] checkpoint best non trouvé : {best_ckpt}", flush=True)
        last_ckpt = os.path.join(ckpt_dir, f"{ckpt_tag}_last.pth")
        best_ckpt = last_ckpt

    # ── 2. EXTRACTION DES EMBEDDINGS ────────────────────────────────────────────
    print("\n[datacurve] 2/3 — Extraction des embeddings", flush=True)

    emb_base = args.emb_dir or cfg.paths.emb_dir
    emb_key = f"vitb16_{regime_tag}_frac{pct:03d}_seed{args.seed}"
    emb_dir_run = os.path.join(emb_base, emb_key)
    os.makedirs(emb_dir_run, exist_ok=True)

    # Extraction val + test (labels 12-class depuis le CSV, remapping appliqué après)
    feats_vt_12 = _extract_backbone_embeddings(best_ckpt, cfg, splits=("val", "test"))

    # Remapping 11-class pour val et test
    feats_vt: dict = {}
    for s, (E, L) in feats_vt_12.items():
        feats_vt[s] = _apply_11cls_remap(E, L)

    np.save(os.path.join(emb_dir_run, "val.npy"), feats_vt["val"][0])
    np.save(os.path.join(emb_dir_run, "val_labels.npy"), feats_vt["val"][1])
    np.save(os.path.join(emb_dir_run, "test.npy"), feats_vt["test"][0])
    np.save(os.path.join(emb_dir_run, "test_labels.npy"), feats_vt["test"][1])
    print(f"  embeddings sauvés → {emb_dir_run}", flush=True)

    # Extraction train (full) puis sous-sélection + remapping
    print("  extraction train (sous-ensemble)...", flush=True)
    feats_tr_12 = _extract_backbone_embeddings(best_ckpt, cfg, splits=("train",))
    E_tr_full_12, L_tr_full_12 = feats_tr_12["train"]
    # Garder uniquement les indices du sous-ensemble (déjà non-RHOL)
    E_tr_raw = E_tr_full_12[orig_train_sub_idx]
    L_tr_raw = L_tr_full_12[orig_train_sub_idx]
    E_tr, L_tr = _apply_11cls_remap(E_tr_raw, L_tr_raw)
    np.save(os.path.join(emb_dir_run, "train.npy"),        E_tr)
    np.save(os.path.join(emb_dir_run, "train_labels.npy"), L_tr)
    print(f"  embeddings train sauvés → {emb_dir_run}", flush=True)

    # ── 3. SONDE LINÉAIRE + MÉTRIQUES ───────────────────────────────────────────
    print("\n[datacurve] 3/3 — Sonde linéaire (lbfgs, 11 classes)", flush=True)
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
    metrics["schema"] = "11cls_no_rhol"

    print(f"\n  ✓ f1_macro_pres_test = {metrics['f1_macro_pres_test']:.4f}"
          f"  f1_macro_8cls_test = {metrics['f1_macro_8cls_test']:.4f}"
          f"  best_C = {metrics['best_C']}", flush=True)

    # Sauvegarde
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  → {metrics_path}", flush=True)

    with open(done_path, "w") as f:
        f.write(f"completed {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")

    elapsed = time.time() - t0
    print(f"\n[datacurve] DONE {tag}  ({elapsed/60:.1f} min)\n", flush=True)


if __name__ == "__main__":
    main()
