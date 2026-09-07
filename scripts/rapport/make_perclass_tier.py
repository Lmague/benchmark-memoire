#!/usr/bin/env python3
"""Per-class F1 + matrices de confusion du palier compétitif (20 groupes).

Vient de `tier_preds_cache.npz` (prédictions du probe interne des runs, MÊME
convention que `significance_matrix_tier.json`) — une seule provenance pour les
20 groupes du palier, y compris les modèles de contexte (R1/R2/R3) et DINOv3
ViT-S/16, absents de l'ancien `per_class_all_models.csv`.

Sorties (results/rapport_data/) :
  per_class_tier.csv — 11 colonnes de F1 par classe + F1-macro
  confusion_tier.npz + confusion_tier_meta.json — matrices 11x11 normalisées par ligne

    python3 scripts/rapport/make_perclass_tier.py
"""
from __future__ import annotations
import csv, json, os, re, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
from registry import CLASSES_11, OUT, TIER_GROUP_KEY, ensure_out

CACHE = os.path.join(OUT, "tier_preds_cache.npz")
DISPLAY = {
    "dinov3_vitb16_lvd_lora_r8": "DINOv3 ViT-B/16 LoRA r=8",
    "dinov3_vith16plus_lvd": "DINOv3 ViT-H+/16 LVD",
    "dinov3_vitb16_lvd_mhsa": "DINOv3 ViT-B/16 MHSA",
    "dinov3_vitl16_lvd": "DINOv3 ViT-L/16 LVD",
    "dinov3_vitb16_lvd_full": "DINOv3 ViT-B/16 Full",
    "simdinov2_vitl16": "SimDINOv2 ViT-L/16",
    "simdinov2_vitb16": "SimDINOv2 ViT-B/16",
    "dinov3_vitb16_lvd": "DINOv3 ViT-B/16 LVD",
    "vitb16_full_old": "ViT-B/16 IN-Full",
    "simdinov2_vitb16_mhsa": "SimDINOv2 ViT-B/16 MHSA",
    "vitb16_mhsa_old": "ViT-B/16 IN-MHSA",
    "simdinov2_vitb16_full": "SimDINOv2 ViT-B/16 Full",
    "resnet50_fulft_sota": "ResNet-50 IN-FT",
    "vitb16_imagenet": "ViT-B/16 IN-1k",
    "scalemae_vitl16": "ScaleMAE ViT-L/16",
    "satmae_vitl16": "SatMAE ViT-L/16",
    "resnet50_imagenet": "ResNet-50 IN-1k",
    "vitb16_scratch_old": "ViT-B/16 IN-Scratch",
    "simdinov2_vitl16_lora": "SimDINOv2 ViT-L/16 LoRA r=8",
    "dinov3_vitl16_lvd_lora": "DINOv3 ViT-L/16 LoRA r=8",
    "simdinov2_vitb16_lora": "SimDINOv2 ViT-B/16 LoRA r=8",
    "dinov3_vits16_lvd_lora_r8": "DINOv3 ViT-S/16 LoRA r=8",
    "dinov3_vits16_lvd": "DINOv3 ViT-S/16 LVD",
    "ctxdistill_dB_tL": "Contexte R2 (distil. B, fusion)",
    "ctxdistill_dA_tL": "Contexte R1 (distil. A, tuile)",
    "ctxdistill_dA_tEMA": "Contexte R3 (distil. A, EMA)",
    "simdinov2_vitb16_lora_r8_b611": "SimDINOv2-B LoRA r8 (blocs 6-11)",
    "simdinov2_vitb16_lora_r8_b05": "SimDINOv2-B LoRA r8 (blocs 0-5)",
    "simdinov2_vitb16_lora_r8_b911": "SimDINOv2-B LoRA r8 (blocs 9-11)",
    "simdinov2_vitb16_lora_r2": "SimDINOv2-B LoRA r2",
    "simdinov2_vitb16_lora_r4": "SimDINOv2-B LoRA r4",
    "simdinov2_vitb16_lora_r16": "SimDINOv2-B LoRA r16",
    "simdinov2_vitb16_lora_r32": "SimDINOv2-B LoRA r32",
    "simdinov2_vitb16_lora_r8_s2": "SimDINOv2-B LoRA r8 (scaling 2)",
    "simdinov2_vitb16_lora_r16_s2": "SimDINOv2-B LoRA r16 (scaling 2)",
    "simdinov2_vitb16_lora_r8_rslora": "SimDINOv2-B LoRA r8 (rsLoRA)",
    "simdinov2_vitb16_lora_r16_rslora": "SimDINOv2-B LoRA r16 (rsLoRA)",
    "simdinov2_vitb16_lora_r8_qkv": "SimDINOv2-B LoRA r8 (Q+K+V)",
    "simdinov2_vitb16_norm_tuning": "SimDINOv2-B NormTuning",
}
GROUP_DISPLAY = {grp: DISPLAY[k] for grp, k in TIER_GROUP_KEY.items() if k in DISPLAY}


def f1_per_class(yt, yp, k=11):
    cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
    tp = np.diag(cm)
    denom = 2.0 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(k), where=denom > 0)
    return cm, f1


def main():
    ensure_out()
    d = np.load(CACHE, allow_pickle=True)
    groups = {}
    for key in d.files:
        if not key.endswith("|pred"):
            continue
        spe = key[:-len("|pred")]
        name = re.sub(r"\|seed\d+$", "", spe)      # nom de groupe (sans seed)
        true = d.get(spe + "|true")
        if true is None:
            continue
        groups.setdefault(name, []).append((true, d[key]))
    rows, mats, meta = [], {}, {}
    for name, pairs in sorted(groups.items(), key=lambda kv: kv[0]):
        cms, f1s = [], []
        for yt, yp in pairs:
            cm, f1 = f1_per_class(yt, yp)
            cms.append(cm); f1s.append(f1)
        cm = cms[0] if len(cms) == 1 else np.mean(cms, axis=0)
        f1 = np.mean(f1s, axis=0)
        tgt = GROUP_DISPLAY.get(name, name)
        rec = {"model": tgt.replace(" ", "_"), "display": tgt,
               "id": TIER_GROUP_KEY.get(name, name)}
        for j, c in enumerate(CLASSES_11):
            rec[c] = "{:.4f}".format(f1[j])
        rec["f1_macro_pres"] = "{:.4f}".format(np.mean(f1))
        rows.append(rec)
        cmn = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
        mats[name] = cmn
        meta[name] = {"label": tgt, "f1_macro_pres": float(np.mean(f1))}
    rows.sort(key=lambda r: -float(r["f1_macro_pres"]))   # tri par F1-macro décroissant
    cols = ["model", "display", "id"] + list(CLASSES_11) + ["f1_macro_pres"]
    with open(os.path.join(OUT, "per_class_tier.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    np.savez_compressed(os.path.join(OUT, "confusion_tier.npz"), **mats)
    json.dump(meta, open(os.path.join(OUT, "confusion_tier_meta.json"), "w"),
              indent=1, ensure_ascii=False)
    print("[OK] per_class_tier.csv (" + str(len(rows)) + " groupes) + confusion_tier.npz (" + str(len(mats)) + " matrices)")


if __name__ == "__main__":
    main()
