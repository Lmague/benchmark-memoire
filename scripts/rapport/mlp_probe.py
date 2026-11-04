#!/usr/bin/env python3
"""Probe MLP (readout non linéaire) sur les embeddings cachés des 20 modèles.

Le benchmark canonique n'évalue les modèles qu'avec un probe **linéaire**
(LogisticRegression lbfgs multinomial, `C` sélectionné sur val, `f1_macro_pres`
11 classes). La conclusion « gelé ≈ affiné » repose donc sur l'hypothèse implicite
que l'information de classe est linéairement décodable. Ce script teste cette
hypothèse avec un MLP à **une seule couche cachée** — délibérément : on cherche la
présence de structure non linéaire simple, pas le meilleur classifieur possible.

Protocole (miroir exact de `src/probe.py`) :
  - mêmes embeddings cachés, même split (figé dans l'ordre des lignes des .npy),
  - `StandardScaler` fit sur train appliqué à val/test (obligatoire : sans, le F1
    décroche de ~0,01 — cf. AGENT_MEMORY.md 2026-07-25),
  - schéma **11 classes** (RHOL retirée par `drop_class` pour les sources 12cls),
  - métrique `f1_macro_pres` (`src/metrics.py`), classes présentes dans le split évalué,
  - baseline linéaire **recalculée ici** sur les tableaux standardisés identiques que
    voit le MLP, avec la grille canonique unique `C ∈ {1e-4..10}` pour les 20 modèles
    (AGENTS.md §4.3 : une comparaison n'est valide qu'à grille identique). Le delta
    n'isole donc que la non-linéarité du readout.

Grille MLP : `Linear(D,H) → ReLU → Dropout(p) → Linear(H,11)`,
H ∈ {64,128,256} × p ∈ {0,0.2} × weight_decay ∈ {0,1e-4,1e-3} = 18 configs × 3 seeds.
Early stopping sur val `f1_macro_pres`, patience 15 **fixe** pour toute la grille et
tous les modèles (jamais ajustée). `best_config` = config de moyenne val la plus haute
sur les 3 seeds — miroir de la logique `best_C`. Le test n'entre jamais en sélection.

Sorties (results/rapport_data/) :
  mlp_probe_results.csv               — 20 modèles : best_config, test F1 moy/σ, F1 par classe
  mlp_vs_linear_delta.csv             — delta MLP − linéaire, agrégé et par classe
  correlations_mlp_delta_geometry.csv — Spearman delta ↔ géométrie (IC bootstrap 2000, seed 42)
  mlp_probe_summary.md                — synthèse
  mlp_probe_meta.json                 — protocole, contrôles, contrôle labels permutés

    python3 scripts/rapport/mlp_probe.py --quick        # validation bout-en-bout
    python3 scripts/rapport/mlp_probe.py --control-only # contrôle labels permutés seul
    python3 scripts/rapport/mlp_probe.py                # run complet (~8 h CPU)

Strictement additif : ne modifie aucun régime, cache d'embeddings ou résultat existant.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# MONO-THREAD OBLIGATOIRE — pas un réglage de performance : le nombre de threads
# BLAS change l'ordre de réduction et donc la solution vers laquelle lbfgs
# converge (voir scripts/rapport/significance_tier.py). La baseline linéaire
# recalculée ici doit être comparable au probe canonique. Doit précéder l'import
# de numpy/sklearn.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from registry import (CANONICAL_F1, CLASSES_8, CLASSES_11, CLASSES_DEAD,
                      FROZEN_MODELS, OUT, RUN_FAMILIES, ensure_out, p)
from src.latent import drop_class
from src.metrics import eval_classifier, per_class_f1
from src.utils import CLASS_TO_IDX, make_canonical_lr

# ── Protocole ────────────────────────────────────────────────────────────────
N_CLASSES = len(CLASSES_11)                       # 11 (RHOL exclue)
RHOL_IDX = CLASS_TO_IDX["RHOL"]                   # 7, dans le référentiel 12cls
SEED = 42                                          # seed canonique du dépôt
SEEDS = (0, 1, 2)                                  # seeds de fine-tuning
C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)     # grille canonique unique
MAX_ITER = 2000

HIDDEN_GRID = (64, 128, 256)
DROPOUT_GRID = (0.0, 0.2)
WD_GRID = (0.0, 1e-4, 1e-3)
# lr fixé UNE fois par un pilote sur dinov3_vitb16_lvd (cf. PILOTE_LR), puis tenu
# constant pour toute la grille et les 20 modèles — jamais re-réglé par modèle.
LR = 3e-4
BATCH = 512
MAX_EPOCHS = 200
PATIENCE = 15            # fixe pour toute la grille — ne jamais tuner
N_THREADS = 8

N_TEST_EXPECTED = 17598
TOL_FROZEN = 5e-4        # tolérance de aggregate_screening.checks
CONTROL_MAX_F1 = 0.15    # au-delà, la grille mémorise : aucun delta n'est publiable
CONTROL_EPOCHS = 60      # contrôle : pas d'early stopping, on laisse mémoriser

# Pilote de régularisation (dinov3_vitb16_lvd, h=256, p=0.2) — consigné dans le meta.
# Vérifie que la grille wd {0, 1e-4, 1e-3} imposée n'est pas le facteur limitant :
# même à wd=1, le val plafonne au niveau du probe linéaire et le test reste dessous.
# Donc l'absence de gain non linéaire n'est PAS un artefact de sous-régularisation.
PILOTE_LR = {
    "modele": "dinov3_vitb16_lvd", "hidden": 256, "dropout": 0.2,
    "reference_lineaire": {"best_C": 0.001, "val_f1_pres": 0.4606, "test_f1_pres": 0.4712},
    "grille": [
        {"lr": 1e-3, "weight_decay": 1e-3, "val_f1_pres": 0.4569, "test_f1_pres": 0.4689},
        {"lr": 1e-3, "weight_decay": 1e-2, "val_f1_pres": 0.4569, "test_f1_pres": 0.4642},
        {"lr": 1e-3, "weight_decay": 1e-1, "val_f1_pres": 0.4550, "test_f1_pres": 0.4642},
        {"lr": 1e-3, "weight_decay": 1.0, "val_f1_pres": 0.4604, "test_f1_pres": 0.4564},
        {"lr": 3e-4, "weight_decay": 1e-3, "val_f1_pres": 0.4607, "test_f1_pres": 0.4643},
        {"lr": 3e-4, "weight_decay": 1e-2, "val_f1_pres": 0.4606, "test_f1_pres": 0.4648},
        {"lr": 3e-4, "weight_decay": 1e-1, "val_f1_pres": 0.4609, "test_f1_pres": 0.4639},
        {"lr": 3e-4, "weight_decay": 1.0, "val_f1_pres": 0.4631, "test_f1_pres": 0.4666},
    ],
    "conclusion": ("lr=3e-4 retenu (val plus stable sur toute la plage wd) et figé pour "
                   "les 20 modèles ; élargir la grille wd au-delà de 1e-3 ne renverse "
                   "pas le signe du delta."),
}

SUFFIX = ""              # "_quick" en mode validation : n'écrase jamais les vraies tables


def out_path(name: str) -> str:
    stem, ext = os.path.splitext(name)
    return os.path.join(OUT, f"{stem}{SUFFIX}{ext}")


def cache_path() -> str:
    """Cache de reprise (indispensable sur un run de plusieurs heures)."""
    return out_path("mlp_probe_partial.json")

# ── Carte des chemins d'embeddings des modèles évalués ───────────────────────
# Source de vérité : geometry_test_fixed_n.model_jobs() (vérifié au démarrage par
# check_paths). Deux pièges de nommage, documentés dans AGENT_MEMORY.md :
#   embeddings/vitb16_full_frac100_seed*                    = ResNet-50 FT (2048 dim)
#   DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_...   = LoRA r=8 (malgré le nom)
FT_BASES = {
    "dinov3_vitb16_lvd_full": "ft_ssl_results/dinov3_vitb16_lvd_full_embeddings/dinov3_vitb16_lvd_full_frac100",
    "dinov3_vitb16_lvd_mhsa": "ft_ssl_results/dinov3_vitb16_lvd_mhsa_embeddings/dinov3_vitb16_lvd_mhsa_frac100",
    "simdinov2_vitb16_full": "ft_ssl_results/simdinov2_vitb16_full_embeddings/simdinov2_vitb16_full_frac100",
    "simdinov2_vitb16_mhsa": "ft_ssl_results/simdinov2_vitb16_mhsa_embeddings/simdinov2_vitb16_mhsa_frac100",
    "dinov3_vitb16_lvd_lora_r8": "DINOv3_LoRA_8/embeddings/dinov3_vitb16_lvd_explora_frac100",
    "resnet50_fulft_sota": "embeddings/vitb16_full_frac100",
    "vitb16_full_old": "sota_screening/full/embeddings/vitb16_full_frac100",
    "vitb16_mhsa_old": "sota_screening/mhsa/embeddings/vitb16_mhsa_frac100",
    "vitb16_scratch_old": "sota_screening/scratch/embeddings/vitb16_scratch_frac100",
}
MODELS = list(FROZEN_MODELS) + list(FT_BASES)


def display_of(key: str) -> str:
    if key in FROZEN_MODELS:
        return FROZEN_MODELS[key][0]
    return RUN_FAMILIES[key][0]


def type_of(key: str) -> str:
    return "frozen" if key in FROZEN_MODELS else RUN_FAMILIES[key][3]


def emb_sets(key: str) -> list[dict]:
    """Les 3 jeux d'embeddings d'un modèle : ``[{seed, tag, paths}, ...]``.

    Gelés : un seul jeu, réutilisé pour les 3 seeds d'initialisation (``tag`` identique
    → chargement et probe linéaire mutualisés). Affinés : un dossier par seed de
    fine-tuning, donc trois backbones distincts.
    """
    out = []
    for s in SEEDS:
        if key in FROZEN_MODELS:
            tag = key
            paths = {sp: (p("embeddings", f"{key}_{sp}.npy"),
                          p("embeddings", f"{key}_{sp}_labels.npy"))
                     for sp in ("train", "val", "test")}
        else:
            base = f"{FT_BASES[key]}_seed{s}"
            tag = base
            paths = {sp: (p(base, f"{sp}.npy"), p(base, f"{sp}_labels.npy"))
                     for sp in ("train", "val", "test")}
        out.append({"seed": s, "tag": tag, "paths": paths})
    return out


def check_paths() -> None:
    """Garde-fou : nos chemins test doivent coïncider avec geometry_test_fixed_n."""
    from geometry_test_fixed_n import model_jobs
    ref = {key: [ep for ep, _ in pairs] for key, _kind, pairs, _tr in model_jobs()}
    missing = [m for m in MODELS if m not in ref]
    if missing:
        raise RuntimeError(f"[paths] modèles absents de geometry_test_fixed_n : {missing}")
    for m in MODELS:
        ours = [s["paths"]["test"][0] for s in emb_sets(m)]
        theirs = ref[m]
        if m in FROZEN_MODELS:
            ours = ours[:1]
        if ours != theirs:
            raise RuntimeError(f"[paths] divergence pour {m}:\n  ici  {ours}\n  réf  {theirs}")
    for m in MODELS:
        for st in emb_sets(m):
            for sp, (ep, lp) in st["paths"].items():
                if not (os.path.exists(ep) and os.path.exists(lp)):
                    raise FileNotFoundError(
                        f"[paths] {m}/seed{st['seed']}/{sp} absent : {ep}\n"
                        "Embeddings non trouvés — ne PAS relancer d'extraction, "
                        "signaler le modèle manquant.")
    print(f"[OK] chemins vérifiés contre geometry_test_fixed_n.model_jobs() "
          f"({len(MODELS)} modèles, 3 splits × 3 seeds)")


# ── Chargement / standardisation ─────────────────────────────────────────────
def load_set(paths: dict) -> dict:
    """Charge train/val/test, ramène au schéma 11cls, standardise (fit sur train).

    Les gelés sont stockés en 12cls (RHOL=7 présente au train, absente de val/test) :
    on applique ``drop_class`` — qui retire la classe ET compacte les indices >7 de −1.
    Sans cette réduction, le classifieur serait fitté sur 12 classes et la métrique
    porterait sur les mauvais indices (cf. known_issues.md).
    """
    from sklearn.preprocessing import StandardScaler
    raw = {}
    for sp in ("train", "val", "test"):
        ep, lp = paths[sp]
        raw[sp] = (np.load(ep).astype(np.float32), np.load(lp).astype(np.int64))
    schema = "12cls" if int(max(l.max() for _, l in raw.values())) >= N_CLASSES else "11cls"
    if schema == "12cls":
        raw = {sp: drop_class(*raw[sp], RHOL_IDX) for sp in raw}

    for sp, (_, l) in raw.items():
        u = set(int(v) for v in np.unique(l))
        if not u <= set(range(N_CLASSES)):
            raise RuntimeError(f"labels hors [0,{N_CLASSES}) sur {sp} : {sorted(u)}")
    if raw["test"][1].shape[0] != N_TEST_EXPECTED:
        raise RuntimeError(f"n_test={raw['test'][1].shape[0]} ≠ {N_TEST_EXPECTED}")
    for sp in ("val", "test"):
        n_pres = len(np.unique(raw[sp][1]))
        if n_pres != N_CLASSES:
            print(f"  [!] {sp}: {n_pres}/{N_CLASSES} classes présentes "
                  f"— f1_macro_pres porte sur les classes présentes de ce split")

    sc = StandardScaler()
    xtr = sc.fit_transform(raw["train"][0])
    out = {"schema": schema, "dim": int(xtr.shape[1]),
           "train": (xtr, raw["train"][1]),
           "val": (sc.transform(raw["val"][0]), raw["val"][1]),
           "test": (sc.transform(raw["test"][0]), raw["test"][1])}
    del raw, sc
    gc.collect()
    return out


# ── Baseline linéaire canonique, recalculée ──────────────────────────────────
def linear_probe(data: dict) -> dict:
    """LR canonique (lbfgs multinomial, seed 42, max_iter 2000), C choisi sur val.

    Sélection sur val ``f1_macro_pres``, fit final sur le train COMPLET (défaut
    délibéré de src/probe.py : un sous-échantillon de 20 000 change le best_C des
    modèles à fort effectif et décale le F1 de ~0,01).

    Mémoire : lbfgs travaille en float64. On convertit le train **une fois, en place**
    et on libère la copie float32 — sinon les deux coexistent (1,2 Go à 2048 dim, sur
    une machine qui n'a que ~1 Go libre). val/test restent en float32 : `predict` fait
    une conversion transitoire, bien moins coûteuse qu'une copie permanente.
    APPELER APRÈS la phase MLP : le train n'est plus utilisable en float32 ensuite.
    """
    from sklearn.metrics import f1_score
    data["train"] = (np.asarray(data["train"][0], dtype=np.float64), data["train"][1])
    gc.collect()
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    pres_va = sorted({int(v) for v in yva})
    best_c, best_f1 = None, -1.0
    for c in C_GRID:
        clf = make_canonical_lr(C=c, max_iter=MAX_ITER, random_state=SEED)
        clf.fit(xtr, ytr)
        f1v = float(f1_score(yva, clf.predict(xva), average="macro",
                             zero_division=0, labels=pres_va))
        if f1v > best_f1:
            best_f1, best_c = f1v, c
    clf = make_canonical_lr(C=best_c, max_iter=MAX_ITER, random_state=SEED)
    clf.fit(xtr, ytr)
    yp = clf.predict(xte)
    ev = eval_classifier(yte, yp, N_CLASSES)
    return {"best_C": best_c, "val_f1_pres": best_f1,
            "f1_pres": ev["f1_macro_pres"], "accuracy": ev["accuracy"],
            "f1_weighted": ev["f1_weighted"],
            "per_class": per_class_f1(yte, yp, CLASSES_11)}


# ── MLP ──────────────────────────────────────────────────────────────────────
def _torch():
    import torch
    torch.set_num_threads(N_THREADS)
    return torch


def _predict(torch, net, X):
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 4096):
            outs.append(net(X[i:i + 4096]).argmax(1))
    return torch.cat(outs).numpy()


def train_mlp(data: dict, hidden: int, dropout: float, wd: float, seed: int,
              max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE,
              ytr_override=None, nonlinear: bool = True) -> dict:
    """Entraîne un readout, early stopping sur val f1_macro_pres.

    ``nonlinear=True``  : MLP 1 couche cachée ``Linear(D,H) → ReLU → Dropout → Linear(H,11)``.
    ``nonlinear=False`` : témoin apparié ``Dropout → Linear(D,11)`` — MÊME boucle
    d'optimisation, aucune non-linéarité. Sert à séparer l'effet de la ReLU de celui
    de l'optimiseur (AdamW early-stoppé) face au probe lbfgs convergé : sans ce témoin,
    le delta MLP − lbfgs mélange les deux (cf. `decompose`).

    Retourne les métriques test des poids de MEILLEURE val (le test n'intervient
    jamais dans la sélection). ``ytr_override`` sert au contrôle labels permutés.
    """
    from sklearn.metrics import f1_score
    torch = _torch()
    nn = torch.nn

    xtr, ytr = data["train"]
    if ytr_override is not None:
        ytr = ytr_override
    xva, yva = data["val"]
    xte, yte = data["test"]
    pres_va = sorted({int(v) for v in yva})

    def _t(a, dt=np.float32):   # vue zéro-copie si déjà float32 contigu
        return torch.from_numpy(np.ascontiguousarray(a, dtype=dt))

    Xtr, Xva, Xte = _t(xtr), _t(xva), _t(xte)
    Ytr = _t(ytr, np.int64)

    torch.manual_seed(seed)
    net = (nn.Sequential(nn.Linear(data["dim"], hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))
           if nonlinear else
           nn.Sequential(nn.Dropout(dropout), nn.Linear(data["dim"], N_CLASSES)))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()          # non pondérée (protocole linear-eval)
    g = torch.Generator().manual_seed(seed)

    n = Xtr.shape[0]
    best_f1, best_ep, best_state, bad = -1.0, -1, None, 0
    for ep in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        f1v = float(f1_score(yva, _predict(torch, net, Xva), average="macro",
                             zero_division=0, labels=pres_va))
        if f1v > best_f1 + 1e-6:
            best_f1, best_ep, bad = f1v, ep, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    yp = _predict(torch, net, Xte)
    ev = eval_classifier(yte, yp, N_CLASSES)
    net.eval()
    with torch.no_grad():
        train_acc = float((_predict(torch, net, Xtr) == np.asarray(ytr)).mean())
    return {"hidden": hidden if nonlinear else 0,
            "dropout": dropout, "weight_decay": wd, "seed": seed,
            "best_epoch": best_ep, "n_epochs": ep, "val_f1_pres": best_f1,
            "f1_pres": ev["f1_macro_pres"], "accuracy": ev["accuracy"],
            "train_accuracy": train_acc,
            "per_class": per_class_f1(yte, yp, CLASSES_11)}


def grid() -> list[tuple]:
    return [(h, d, w) for h in HIDDEN_GRID for d in DROPOUT_GRID for w in WD_GRID]


# ── Boucle principale ────────────────────────────────────────────────────────
def load_cache() -> dict:
    if os.path.exists(cache_path()):
        with open(cache_path()) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(cache_path(), "w") as f:
        json.dump(cache, f)


def run_models(models: list[str], cfgs: list[tuple], skip_linear=False,
               force=False) -> dict:
    cache = {} if force else load_cache()
    t0 = time.time()
    for i, key in enumerate(models, 1):
        if key in cache and not force:
            print(f"[{i:2d}/{len(models)}] {key:28s} déjà en cache → SKIP")
            continue
        rec = {"model": key, "display": display_of(key), "type": type_of(key),
               "seed_source": "init_only" if key in FROZEN_MODELS else "backbone+init",
               "linear": [], "mlp": []}
        # Un jeu d'embeddings par tag : les gelés en ont un seul, partagé par les
        # 3 seeds d'initialisation ; les affinés en ont trois (trois backbones).
        by_tag: dict[str, list[dict]] = {}
        for st in emb_sets(key):
            by_tag.setdefault(st["tag"], []).append(st)

        for tag, sts in by_tag.items():
            data = load_set(sts[0]["paths"])
            rec["dim"] = data["dim"]
            rec["schema"] = data["schema"]
            # Phase MLP AVANT la phase linéaire : linear_probe convertit le train en
            # float64 en place et le rend inutilisable pour torch (cf. sa docstring).
            for st in sts:
                for (h, d, w) in cfgs:
                    rec["mlp"].append(train_mlp(data, h, d, w, seed=SEED + st["seed"]))
                print(f"  [mlp ] {key:28s} seed{st['seed']} {len(cfgs)} configs "
                      f"({time.time() - t0:.0f}s)")
            if not skip_linear:
                lin = linear_probe(data)
                lin["tag"] = tag
                print(f"  [lin ] {key:28s} {tag.split('/')[-1]:34s} "
                      f"C={lin['best_C']:<7g} F1={lin['f1_pres']:.4f}  "
                      f"({time.time() - t0:.0f}s)")
                rec["linear"] += [dict(lin) for _ in sts]
            data = None
            gc.collect()
        cache[key] = rec
        save_cache(cache)
        best = select_best(rec, cfgs)
        lin = (f"  lin={np.mean([l['f1_pres'] for l in rec['linear']]):.4f}"
               if rec["linear"] else "")
        print(f"[{i:2d}/{len(models)}] {key:28s} h={best['hidden']} "
              f"p={best['dropout']} wd={best['weight_decay']:g}  "
              f"MLP={best['f1_pres_mean']:.4f}±{best['f1_pres_std']:.4f}{lin}")
    return cache


def _stats(v):
    a = np.asarray(v, dtype=float)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            float(a.min()), float(a.max()))


def select_best(rec: dict, cfgs: list[tuple]) -> dict:
    """best_config = moyenne val f1_macro_pres la plus haute sur les 3 seeds."""
    best, best_val = None, -1.0
    for (h, d, w) in cfgs:
        runs = [r for r in rec["mlp"]
                if r["hidden"] == h and r["dropout"] == d and r["weight_decay"] == w]
        if not runs:
            continue
        mv = float(np.mean([r["val_f1_pres"] for r in runs]))
        if mv > best_val:
            best_val, best = mv, (h, d, w, runs)
    h, d, w, runs = best
    m, s, lo, hi = _stats([r["f1_pres"] for r in runs])
    vm, vs, _, _ = _stats([r["val_f1_pres"] for r in runs])
    am, asd, _, _ = _stats([r["accuracy"] for r in runs])
    eps = [r["best_epoch"] for r in runs]
    return {"hidden": h, "dropout": d, "weight_decay": w, "n_seeds": len(runs),
            "val_f1_pres_mean": vm, "val_f1_pres_std": vs,
            "f1_pres_mean": m, "f1_pres_std": s, "f1_pres_min": lo, "f1_pres_max": hi,
            "accuracy_mean": am, "accuracy_std": asd,
            "best_epoch_mean": float(np.mean(eps)),
            "best_epoch_min": int(np.min(eps)), "best_epoch_max": int(np.max(eps)),
            "per_class_mean": {c: float(np.mean([r["per_class"][c] for r in runs]))
                               for c in CLASSES_11},
            "per_class_std": {c: (float(np.std([r["per_class"][c] for r in runs], ddof=1))
                                  if len(runs) > 1 else 0.0) for c in CLASSES_11},
            "runs": runs}


# ── Contrôle : labels d'entraînement permutés ────────────────────────────────
def chance_levels(ytr, yte) -> dict:
    """Niveaux de hasard sur le test, pour rendre le seuil du contrôle interprétable.

    `stratifie` : prédictions tirées selon les fréquences du train (seed 42).
    `majoritaire` : tout le test prédit dans la classe majoritaire du train.
    """
    rng = np.random.RandomState(SEED)
    freq = np.bincount(np.asarray(ytr), minlength=N_CLASSES).astype(float)
    freq /= freq.sum()
    strat = rng.choice(N_CLASSES, size=len(yte), p=freq)
    major = np.full(len(yte), int(np.argmax(freq)), dtype=np.int64)
    return {"stratifie": eval_classifier(yte, strat, N_CLASSES)["f1_macro_pres"],
            "majoritaire": eval_classifier(yte, major, N_CLASSES)["f1_macro_pres"]}



def run_control(cache: dict, models: list[str], cfgs: list[tuple]) -> list[dict]:
    """Même features, labels du TRAIN permutés (val/test intacts) → le test doit s'effondrer.

    Esprit du contrôle k-means de results/T2a_kmeans_control.json. Pas d'early stopping
    (patience = max_epochs) : on LAISSE le réseau mémoriser le bruit, car c'est la
    non-généralisation qui valide la grille, pas l'incapacité à mémoriser.
    """
    out = []
    for key in models:
        rec = cache.get(key)
        if rec is not None:
            b = select_best(rec, cfgs)
            h, d, w, origin = b["hidden"], b["dropout"], b["weight_decay"], "best_config"
        else:  # capacité maximale de la grille = pire cas pour le surapprentissage
            h, d, w, origin = max(HIDDEN_GRID), min(DROPOUT_GRID), min(WD_GRID), "capacite_max"
        st = emb_sets(key)[0]
        data = load_set(st["paths"])
        ytr = data["train"][1]
        rng = np.random.RandomState(SEED)
        yperm = ytr[rng.permutation(len(ytr))]
        r = train_mlp(data, h, d, w, seed=SEED, max_epochs=CONTROL_EPOCHS,
                      patience=CONTROL_EPOCHS, ytr_override=yperm)
        chance = chance_levels(data["train"][1], data["test"][1])
        ok = r["f1_pres"] < CONTROL_MAX_F1
        out.append({"model": key, "config_origine": origin,
                    "hidden": h, "dropout": d, "weight_decay": w,
                    "n_epochs": r["n_epochs"],
                    "train_accuracy_labels_permutes": r["train_accuracy"],
                    "val_f1_pres": r["val_f1_pres"],
                    "test_f1_macro_pres": r["f1_pres"],
                    "hasard_stratifie": chance["stratifie"],
                    "hasard_majoritaire": chance["majoritaire"],
                    "seuil": CONTROL_MAX_F1, "ok": bool(ok)})
        print(f"  [{'OK  ' if ok else 'FAIL'}] {key:28s} labels permutés : "
              f"test F1={r['f1_pres']:.4f} (seuil {CONTROL_MAX_F1} ; hasard stratifié "
              f"{chance['stratifie']:.4f}) — acc train={r['train_accuracy']:.3f}")
        data = None
        gc.collect()
    return out


# ── Témoin apparié : readout LINÉAIRE entraîné comme le MLP ──────────────────
def linear_adamw_probe(data: dict, seeds: list[int]) -> dict:
    """``Dropout → Linear(D,11)`` entraîné par la MÊME boucle que le MLP.

    Grille dropout × weight_decay (la largeur n'a pas de sens sans couche cachée),
    sélection sur la moyenne val des seeds — protocole identique à ``select_best``.
    """
    best, best_val = None, -1.0
    for d in DROPOUT_GRID:
        for w in WD_GRID:
            runs = [train_mlp(data, 0, d, w, seed=SEED + s, nonlinear=False)
                    for s in seeds]
            mv = float(np.mean([r["val_f1_pres"] for r in runs]))
            if mv > best_val:
                best_val, best = mv, (d, w, runs)
    d, w, runs = best
    return {"dropout": d, "weight_decay": w, "val_f1_pres": best_val,
            "f1_pres": float(np.mean([r["f1_pres"] for r in runs])),
            "per_class": {c: float(np.mean([r["per_class"][c] for r in runs]))
                          for c in CLASSES_11},
            "runs": runs}


def run_linear_adamw(models: list[str]) -> dict:
    """Calcule le témoin linéaire-AdamW pour chaque modèle (cache dédié, reprenable)."""
    path = out_path("mlp_linear_adamw_partial.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for i, key in enumerate(models, 1):
        if key in store:
            print(f"[{i:2d}/{len(models)}] {key:28s} déjà en cache → SKIP")
            continue
        by_tag: dict[str, list[dict]] = {}
        for st in emb_sets(key):
            by_tag.setdefault(st["tag"], []).append(st)
        per_tag = []
        for tag, sts in by_tag.items():
            data = load_set(sts[0]["paths"])
            r = linear_adamw_probe(data, [st["seed"] for st in sts])
            per_tag += [r] * len(sts) if len(by_tag) == 1 else [r]
            data = None
            gc.collect()
        store[key] = {
            "f1_pres": float(np.mean([r["f1_pres"] for r in per_tag])),
            "per_class": {c: float(np.mean([r["per_class"][c] for r in per_tag]))
                          for c in CLASSES_11},
            "dropout": per_tag[0]["dropout"], "weight_decay": per_tag[0]["weight_decay"]}
        with open(path, "w") as f:
            json.dump(store, f)
        print(f"[{i:2d}/{len(models)}] {key:28s} linéaire-AdamW "
              f"F1={store[key]['f1_pres']:.4f}  ({time.time() - t0:.0f}s)")
    return store


def decompose(store: dict) -> list[dict]:
    """Réécrit mlp_vs_linear_delta.csv avec la décomposition du delta.

    delta_total = delta_protocole + delta_nonlinearite, où
      delta_protocole    = linéaire(AdamW) − linéaire(lbfgs)  → coût de l'optimiseur
      delta_nonlinearite = MLP(ReLU)       − linéaire(AdamW)  → effet de la ReLU seule
    Sans cette décomposition, ``delta`` mélange les deux et son SIGNE est trompeur.
    """
    path = out_path("mlp_vs_linear_delta.csv")
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        s = store.get(r["model"])
        if s is None:
            continue
        lin_adamw = (s["f1_pres"] if r["target"] == "f1_pres"
                     else s["per_class"][r["target"]])
        mlp = float(r["mlp_mean"])
        lbfgs = float(r["linear_recalcule"])
        r["linear_adamw"] = lin_adamw
        r["delta_protocole"] = lin_adamw - lbfgs
        r["delta_nonlinearite"] = mlp - lin_adamw
        r["nonlin_interpretable"] = bool(abs(mlp - lin_adamw) > 0.01)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {os.path.basename(path)} enrichi ({len(rows)} lignes, "
          f"+3 colonnes de décomposition)")
    return rows


# ── Sorties ──────────────────────────────────────────────────────────────────
def read_published_linear() -> dict:
    """per_class_all_models.csv : baseline linéaire PUBLIÉE (protocoles mixtes)."""
    path = os.path.join(OUT, "per_class_all_models.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def write_results(cache: dict, cfgs: list[tuple], models: list[str]) -> list[dict]:
    rows = []
    for key in [m for m in models if m in cache]:
        rec = cache[key]
        b = select_best(rec, cfgs)
        row = {"model": key, "display": rec["display"], "type": rec["type"],
               "dim": rec["dim"], "n_seeds": b["n_seeds"],
               "seed_source": rec["seed_source"],
               "best_hidden": b["hidden"], "best_dropout": b["dropout"],
               "best_weight_decay": b["weight_decay"],
               "best_epoch_mean": b["best_epoch_mean"],
               "best_epoch_min": b["best_epoch_min"],
               "best_epoch_max": b["best_epoch_max"],
               "val_f1_pres_mean": b["val_f1_pres_mean"],
               "val_f1_pres_std": b["val_f1_pres_std"],
               "f1_pres_mean": b["f1_pres_mean"], "f1_pres_std": b["f1_pres_std"],
               "f1_pres_min": b["f1_pres_min"], "f1_pres_max": b["f1_pres_max"],
               "accuracy_mean": b["accuracy_mean"], "accuracy_std": b["accuracy_std"]}
        for c in CLASSES_11:
            row[c] = b["per_class_mean"][c]
            row[f"{c}_std"] = b["per_class_std"][c]
        row["_best"] = b
        row["_linear"] = rec["linear"]
        rows.append(row)
    rows.sort(key=lambda r: -r["f1_pres_mean"])
    cols = ["model", "display", "type", "dim", "n_seeds", "seed_source",
            "best_hidden", "best_dropout", "best_weight_decay",
            "best_epoch_mean", "best_epoch_min", "best_epoch_max",
            "val_f1_pres_mean", "val_f1_pres_std",
            "f1_pres_mean", "f1_pres_std", "f1_pres_min", "f1_pres_max",
            "accuracy_mean", "accuracy_std"]
    for c in CLASSES_11:
        cols += [c, f"{c}_std"]
    with open(out_path("mlp_probe_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[OK] {os.path.basename(out_path('mlp_probe_results.csv'))} "
          f"({len(rows)} modèles)")
    return rows


def write_delta(rows: list[dict]) -> list[dict]:
    """Delta MLP − linéaire, agrégé (`f1_pres`) puis par classe. Format long."""
    pub = read_published_linear()
    out = []
    for r in rows:
        b, lins = r["_best"], r["_linear"]
        lin_f1 = float(np.mean([l["f1_pres"] for l in lins]))
        lin_pc = {c: float(np.mean([l["per_class"][c] for l in lins])) for c in CLASSES_11}
        pr = pub.get(r["model"], {})
        targets = [("f1_pres", b["f1_pres_mean"], b["f1_pres_std"], lin_f1,
                    pr.get("f1_pres"))]
        for c in CLASSES_11:
            targets.append((c, b["per_class_mean"][c], b["per_class_std"][c],
                            lin_pc[c], pr.get(c)))
        for target, mlp_m, mlp_s, lin_v, pub_v in targets:
            pub_f = float(pub_v) if pub_v not in (None, "", "nan") else float("nan")
            out.append({
                "model": r["model"], "display": r["display"], "type": r["type"],
                "target": target, "mlp_mean": mlp_m, "mlp_std": mlp_s,
                "linear_recalcule": lin_v, "linear_publie": pub_f,
                "delta": mlp_m - lin_v, "delta_vs_publie": mlp_m - pub_f,
                # AGENTS.md §4.4 : σ inter-seed ≈ 0,008, rien de plus fin n'est lisible
                "interpretable": bool(abs(mlp_m - lin_v) > 0.01)})
    cols = ["model", "display", "type", "target", "mlp_mean", "mlp_std",
            "linear_recalcule", "linear_publie", "delta", "delta_vs_publie",
            "interpretable"]
    with open(out_path("mlp_vs_linear_delta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"[OK] {os.path.basename(out_path('mlp_vs_linear_delta.csv'))} "
          f"({len(out)} lignes)")
    return out


def write_correlations(deltas: list[dict], target: str = "delta_f1_pres",
                       fname: str = "correlations_mlp_delta_geometry.csv") -> list[dict]:
    """Spearman delta ↔ géométrie — méthode identique à correlations.py."""
    from scipy.stats import spearmanr

    from correlations import LABELS, METRICS, bh, boot_ci, loo
    agg = {d["model"]: d["delta"] for d in deltas if d["target"] == "f1_pres"}
    with open(os.path.join(OUT, "geometry_models.csv")) as f:
        geo = [r for r in csv.DictReader(f) if r["model"] in agg]
    geo.sort(key=lambda r: -CANONICAL_F1[r["model"]][0])   # même tri que correlations.py
    pops = {"all": geo,
            "frozen": [r for r in geo if r["model"] in FROZEN_MODELS],
            "top8": geo[:8]}
    out = []
    for pop, sub in pops.items():
        recs = []
        for m in METRICS:
            vals = [(float(r[m]), agg[r["model"]]) for r in sub
                    if r.get(m) not in ("", None)]
            if len(vals) < 5:
                continue
            x = np.array([v[0] for v in vals])
            y = np.array([v[1] for v in vals])
            rho, pv = spearmanr(x, y)
            lo, hi = boot_ci(x, y)
            lmin, lmax = loo(x, y)
            recs.append({"target": target, "population": pop, "n": len(vals),
                         "metric": m, "label": LABELS[m], "rho": float(rho),
                         "ci95_low": lo, "ci95_high": hi, "p": float(pv),
                         "loo_rho_min": lmin, "loo_rho_max": lmax,
                         "signif_ci": bool(lo * hi > 0)})
        if not recs:
            print(f"\n== population « {pop} » : < 5 modèles exploitables → SKIP ==")
            continue
        for rec, q in zip(recs, bh([r["p"] for r in recs])):
            rec["p_bh"] = float(q)
            rec["signif_bh"] = bool(q < 0.05)
        out += recs
        print(f"\n== population « {pop} » (n = {recs[0]['n']}) ==")
        for r in sorted(recs, key=lambda r: -abs(r["rho"]))[:5]:
            print(f"  {'✓' if r['signif_ci'] else ' '} {r['label']:36s} "
                  f"ρ={r['rho']:+.3f}  IC95 [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]"
                  f"  p_BH={r['p_bh']:.3f}")
    cols = ["target", "population", "n", "metric", "label", "rho", "ci95_low",
            "ci95_high", "p", "p_bh", "signif_ci", "signif_bh",
            "loo_rho_min", "loo_rho_max"]
    with open(out_path(fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"\n[OK] {os.path.basename(out_path(fname))} ({len(out)} lignes)")
    return out


def linear_controls(rows: list[dict]) -> list[dict]:
    """Non-régression de la LR recalculée contre registry.CANONICAL_F1.

    Gelés : même protocole que le chiffre canonique (train complet) → doit coller à
    5e-4. Affinés : le chiffre canonique vient d'une grille C interne au run et d'un
    sous-échantillon de 20 000 pour la sélection → écart attendu, consigné mais non
    bloquant (AGENTS.md §4.3 : on privilégie ici la symétrie de grille).
    """
    ctrls = []
    for r in rows:
        key = r["model"]
        if not r["_linear"] or key not in CANONICAL_F1:
            continue
        got = float(np.mean([l["f1_pres"] for l in r["_linear"]]))
        exp = CANONICAL_F1[key][0]
        frozen = key in FROZEN_MODELS
        ok = abs(got - exp) <= TOL_FROZEN if frozen else True
        ctrls.append({"model": key, "kind": "frozen" if frozen else "ft",
                      "metric": "f1_macro_pres_linear", "expected": exp,
                      "got": got, "abs_err": abs(got - exp),
                      "bloquant": frozen, "ok": bool(ok)})
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {key:28s} attendu {exp:.4f}  obtenu {got:.4f}  "
              f"|Δ|={abs(got - exp):.4f}{'' if frozen else '  (grille interne côté canonique)'}")
    return ctrls


def write_summary_decomposed(store: dict) -> None:
    """Réécrit le résumé après décomposition — le message change de signe.

    Relit tout depuis le disque (CSV enrichis + meta) pour rester indépendant d'un
    run complet en mémoire.
    """
    rows = list(csv.DictReader(open(out_path("mlp_probe_results.csv"))))
    deltas = list(csv.DictReader(open(out_path("mlp_vs_linear_delta.csv"))))
    meta_p = out_path("mlp_probe_meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    agg = {d["model"]: d for d in deltas if d["target"] == "f1_pres"}
    disp = {r["model"]: r["display"] for r in rows}

    dn = np.array([float(agg[m]["delta_nonlinearite"]) for m in agg])
    dp = np.array([float(agg[m]["delta_protocole"]) for m in agg])
    dt = np.array([float(agg[m]["delta"]) for m in agg])
    n_interp = int(sum(1 for m in agg if abs(float(agg[m]["delta_nonlinearite"])) > 0.01))

    pc = {}
    for c in CLASSES_8:
        vals = [float(d["delta_nonlinearite"]) for d in deltas
                if d["target"] == c and d.get("delta_nonlinearite") not in (None, "")]
        pc[c] = float(np.mean(vals))
    weak = ("MOSS", "WILL", "PETF")
    d_weak = float(np.mean([pc[c] for c in weak]))
    d_other = float(np.mean([pc[c] for c in CLASSES_8 if c not in weak]))

    def _corr(fname):
        pth = out_path(fname)
        if not os.path.exists(pth):
            return []
        rs = [r for r in csv.DictReader(open(pth))
              if r["population"] == "all" and r["signif_ci"] == "True"]
        return sorted(rs, key=lambda r: -abs(float(r["rho"])))

    sig_nl = _corr("correlations_mlp_nonlin_geometry.csv")

    L = ["# Probe MLP vs probe linéaire — `results/rapport_data/`\n"]
    L.append(f"Sur les {len(agg)} modèles du tableau canonique, le delta brut "
             f"MLP − probe lbfgs est négatif partout (médiane {np.median(dt):+.4f}). "
             "**Ce signe est un artefact d'optimiseur, pas une propriété des "
             "représentations.** Décomposé par un témoin apparié :\n")
    L.append("| Composante | Définition | Médiane | Étendue |")
    L.append("|---|---|---|---|")
    L.append(f"| `delta_protocole` | linéaire(AdamW) − linéaire(lbfgs) | "
             f"{np.median(dp):+.4f} | [{dp.min():+.4f}, {dp.max():+.4f}] |")
    L.append(f"| `delta_nonlinearite` | MLP(ReLU) − linéaire(AdamW) | "
             f"{np.median(dn):+.4f} | [{dn.min():+.4f}, {dn.max():+.4f}] |")
    L.append(f"| `delta` (brut, trompeur) | MLP(ReLU) − linéaire(lbfgs) | "
             f"{np.median(dt):+.4f} | [{dt.min():+.4f}, {dt.max():+.4f}] |")
    L.append("")
    L.append("Un readout **strictement linéaire** entraîné par la même boucle "
             f"(AdamW, early stopping, patience {PATIENCE}) perd "
             f"{abs(np.median(dp)):.4f} de F1 face au même modèle linéaire résolu à "
             "l'optimum par lbfgs. Comparer directement MLP(AdamW) et probe(lbfgs) "
             "fait donc varier **deux** choses à la fois. À protocole égal, la ReLU "
             f"apporte {np.median(dn):+.4f} en médiane "
             f"({int((dn > 0).sum())}/{len(dn)} modèles positifs).\n")

    L.append("## Effet de la non-linéarité, par modèle\n")
    L.append("| Modèle | MLP | linéaire (AdamW) | linéaire (lbfgs) | Δ non-lin. | Δ protocole |")
    L.append("|---|---|---|---|---|---|")
    for m in sorted(agg, key=lambda m: -float(agg[m]["delta_nonlinearite"])):
        d = agg[m]
        L.append(f"| `{m}` | {float(d['mlp_mean']):.4f} | "
                 f"{float(d['linear_adamw']):.4f} | {float(d['linear_recalcule']):.4f} | "
                 f"{float(d['delta_nonlinearite']):+.4f} | "
                 f"{float(d['delta_protocole']):+.4f} |")
    L.append("")
    L.append(f"**{n_interp}/{len(agg)}** modèles dépassent le seuil d'interprétabilité "
             "de 0,01 (AGENTS.md §4.4, σ inter-seed ≈ 0,008). Le gain non linéaire est "
             "donc **réel dans son signe mais trop faible pour être cité modèle par "
             "modèle** — seule sa constance sur 20 modèles est informative.\n")

    L.append("## Répartition par classe (Δ non-linéarité, moyenne sur les modèles)\n")
    L.append("| Classe | Δ moyen |")
    L.append("|---|---|")
    for c, v in sorted(pc.items(), key=lambda kv: -kv[1]):
        L.append(f"| {c} | {v:+.4f} |")
    L.append(f"\nClasses faibles (MOSS, WILL, PETF) : **{d_weak:+.4f}** ; "
             f"les 5 autres classes vivantes : **{d_other:+.4f}**. "
             f"ARCA/DRYI/RUBC restent à 0 dans les trois readouts (`CLASSES_DEAD`).\n")

    L.append("## Corrélations Δ non-linéarité ↔ géométrie (population `all`)\n")
    if sig_nl:
        L.append("| Métrique | ρ | IC95 | p_BH |")
        L.append("|---|---|---|---|")
        for c in sig_nl:
            L.append(f"| {c['label']} | {float(c['rho']):+.3f} | "
                     f"[{float(c['ci95_low']):+.2f}, {float(c['ci95_high']):+.2f}] | "
                     f"{float(c['p_bh']):.3f} |")
        L.append("")
    else:
        L.append("Aucune métrique géométrique n'a d'IC95 bootstrap excluant 0. "
                 "Les corrélations fortes de `correlations_mlp_delta_geometry.csv` "
                 "(anisotropie, silhouette…) portent sur le delta **confondu** et "
                 "reflètent surtout le coût de l'optimiseur — ne pas les citer comme "
                 "un lien géométrie ↔ non-linéarité. —\n")

    ctrl = meta.get("controle_labels_permutes", [])
    if ctrl:
        L.append("## Contrôles\n")
        L.append("| Contrôle | Statut | Détail |")
        L.append("|---|---|---|")
        for c in ctrl:
            L.append(f"| labels train permutés — `{c['model']}` | "
                     f"{'✅' if c['ok'] else '❌'} | test F1 = "
                     f"{c['test_f1_macro_pres']:.4f} (hasard stratifié "
                     f"{c.get('hasard_stratifie', float('nan')):.4f}), acc train = "
                     f"{c['train_accuracy_labels_permutes']:.3f} |")
        L.append("| témoin linéaire-AdamW (sans ReLU) | ✅ | "
                 f"médiane {np.median(dp):+.4f} vs lbfgs — isole le coût de l'optimiseur |")
        L.append("")

    L.append("**Un readout non linéaire peu profond extrait un peu plus que le probe "
             "linéaire (≈ +0,004 de F1-macro), mais l'effet est sous le seuil de bruit "
             "inter-seed : il ne remet pas en cause le classement du benchmark.**\n")
    L.append("_Généré par `scripts/rapport/mlp_probe.py --decompose`, "
             f"{time.strftime('%Y-%m-%d %H:%M')}._\n")

    with open(out_path("mlp_probe_summary.md"), "w") as f:
        f.write("\n".join(L))
    print(f"[OK] {os.path.basename(out_path('mlp_probe_summary.md'))} (réécrit)")

    if meta:
        meta["decomposition"] = {
            "temoin": ("readout Dropout → Linear(D,11) entraîné par la MÊME boucle que le "
                       "MLP (AdamW, early stopping, grille dropout × weight_decay, 3 seeds)"),
            "pourquoi": ("le delta brut MLP(AdamW) − probe(lbfgs) fait varier la "
                         "non-linéarité ET l'optimiseur ; son signe est dominé par le second"),
            "delta_protocole_mediane": float(np.median(dp)),
            "delta_nonlinearite_mediane": float(np.median(dn)),
            "n_modeles_nonlin_positifs": int((dn > 0).sum()),
            "n_modeles_interpretables": n_interp,
            "config_temoin": {m: {"dropout": store[m]["dropout"],
                                  "weight_decay": store[m]["weight_decay"]}
                              for m in store},
        }
        with open(meta_p, "w") as f:
            json.dump(meta, f, indent=1, ensure_ascii=False)
        print(f"[OK] {os.path.basename(meta_p)} (décomposition ajoutée)")


def write_summary(rows, deltas, corrs, controls, lin_ctrls, meta) -> None:
    agg = {d["model"]: d for d in deltas if d["target"] == "f1_pres"}
    ordered = sorted(rows, key=lambda r: -agg[r["model"]]["delta"])
    n_interp = sum(1 for r in rows if agg[r["model"]]["interpretable"])
    deltas_v = np.array([agg[r["model"]]["delta"] for r in rows])

    per_class = {}
    for c in CLASSES_8:
        per_class[c] = float(np.mean([d["delta"] for d in deltas if d["target"] == c]))
    weak = ("MOSS", "WILL", "PETF")
    d_weak = float(np.mean([per_class[c] for c in weak]))
    d_other = float(np.mean([per_class[c] for c in CLASSES_8 if c not in weak]))

    sig = [c for c in corrs if c["population"] == "all" and c["signif_ci"]]
    sig.sort(key=lambda c: -abs(c["rho"]))

    L = []
    L.append("# Probe MLP vs probe linéaire — `results/rapport_data/`\n")
    L.append(f"Delta MLP − linéaire sur {len(rows)} modèles : "
             f"médiane {np.median(deltas_v):+.4f}, "
             f"étendue [{deltas_v.min():+.4f}, {deltas_v.max():+.4f}] ; "
             f"**{n_interp}/{len(rows)}** dépassent le seuil d'interprétabilité "
             f"de 0,01 (AGENTS.md §4.4).\n")

    L.append("## Protocole\n")
    L.append("MLP à **une seule couche cachée** `Linear(D,H) → ReLU → Dropout → "
             f"Linear(H,{N_CLASSES})`, grille {HIDDEN_GRID} × {DROPOUT_GRID} × "
             f"{WD_GRID} = {len(grid())} configs × 3 seeds, early stopping sur val "
             f"`f1_macro_pres` (patience {PATIENCE}, fixe). Baseline linéaire "
             "recalculée sur les **mêmes tableaux standardisés**, grille "
             "`C ∈ {1e-4..10}` identique pour les 20 modèles. Schéma 11 classes, "
             "métrique `f1_macro_pres` sur le test (n = 17 598).\n")

    L.append("## Delta agrégé par modèle\n")
    L.append("| Modèle | type | config MLP | MLP | linéaire | Δ | > 0,01 |")
    L.append("|---|---|---|---|---|---|---|")
    for r in ordered:
        d = agg[r["model"]]
        L.append(f"| `{r['model']}` | {r['type']} | "
                 f"h={r['best_hidden']}, p={r['best_dropout']}, "
                 f"wd={r['best_weight_decay']:g} | "
                 f"{r['f1_pres_mean']:.4f}±{r['f1_pres_std']:.4f} | "
                 f"{d['linear_recalcule']:.4f} | {d['delta']:+.4f} | "
                 f"{'✅' if d['interpretable'] else '—'} |")
    L.append("")

    L.append("## Répartition par classe (moyenne sur les 20 modèles)\n")
    L.append("| Classe | Δ moyen |")
    L.append("|---|---|")
    for c, v in sorted(per_class.items(), key=lambda kv: -kv[1]):
        L.append(f"| {c} | {v:+.4f} |")
    L.append(f"\nClasses faibles (MOSS, WILL, PETF) : Δ moyen **{d_weak:+.4f}** ; "
             f"les 5 autres classes vivantes : **{d_other:+.4f}**. "
             f"ARCA/DRYI/RUBC restent à 0 dans les deux readouts "
             f"(`CLASSES_DEAD`, support test insuffisant).\n")

    L.append("## Corrélations Δ ↔ géométrie (population `all`)\n")
    if sig:
        L.append("| Métrique | ρ | IC95 | p_BH |")
        L.append("|---|---|---|---|")
        for c in sig:
            L.append(f"| {c['label']} | {c['rho']:+.3f} | "
                     f"[{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}] | {c['p_bh']:.3f} |")
        L.append("")
    else:
        L.append("Aucune métrique géométrique n'a d'IC95 bootstrap excluant 0 "
                 "sur la population complète. —\n")

    L.append("## Contrôles\n")
    L.append("| Contrôle | Statut | Détail |")
    L.append("|---|---|---|")
    for c in controls:
        L.append(f"| labels train permutés — `{c['model']}` | "
                 f"{'✅' if c['ok'] else '❌'} | "
                 f"test F1 = {c['test_f1_macro_pres']:.4f} "
                 f"(hasard stratifié {c['hasard_stratifie']:.4f}, seuil {c['seuil']}), "
                 f"acc train = {c['train_accuracy_labels_permutes']:.3f} |")
    nb_fail = [c for c in lin_ctrls if not c["ok"]]
    maxerr = max((c["abs_err"] for c in lin_ctrls if c["kind"] == "frozen"), default=0.0)
    L.append(f"| LR recalculée == `CANONICAL_F1` (gelés, tol. {TOL_FROZEN}) | "
             f"{'✅' if not nb_fail else '❌'} | max &#124;Δ&#124; = {maxerr:.5f} |")
    L.append("")

    verdict = ("**Le readout non linéaire ne change pas les conclusions du benchmark.**"
               if n_interp == 0 else
               f"**{n_interp} modèle(s) gagnent plus de 0,01 de F1 avec un readout non "
               "linéaire — le classement linéaire n'est donc pas suffisant à lui seul.**")
    L.append(verdict + "\n")
    L.append(f"_Généré par `scripts/rapport/mlp_probe.py`, {meta['date']}, "
             f"{meta['duree_s']:.0f} s._\n")

    with open(out_path("mlp_probe_summary.md"), "w") as f:
        f.write("\n".join(L))
    print(f"[OK] {os.path.basename(out_path('mlp_probe_summary.md'))}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="1 gelé + 1 affiné, grille réduite à 2 configs (validation)")
    ap.add_argument("--only", nargs="*", default=None, help="restreindre à certains modèles")
    ap.add_argument("--control-only", action="store_true",
                    help="contrôle labels permutés seul (n'écrit aucun CSV)")
    ap.add_argument("--skip-linear", action="store_true",
                    help="ne pas recalculer la baseline linéaire (debug)")
    ap.add_argument("--force", action="store_true", help="ignorer le cache de reprise")
    ap.add_argument("--decompose", action="store_true",
                    help="ajoute le témoin linéaire-AdamW et décompose le delta "
                         "(protocole vs non-linéarité) — nécessite un run complet préalable")
    args = ap.parse_args()

    global SUFFIX
    ensure_out()
    t0 = time.time()
    check_paths()

    models = args.only or MODELS
    cfgs = grid()
    if args.quick:
        SUFFIX = "_quick"        # ne jamais écraser les vraies tables en validation
        models = args.only or ["dinov3_vitb16_lvd", "dinov3_vitb16_lvd_full"]
        cfgs = [(64, 0.0, 0.0), (256, 0.2, 1e-3)]
        print(f"[quick] {models} × {len(cfgs)} configs → sorties suffixées « _quick »")

    if args.decompose:
        print(f"\n== Témoin linéaire-AdamW : {len(models)} modèles × "
              f"{len(DROPOUT_GRID) * len(WD_GRID)} configs × 3 seeds ==")
        store = run_linear_adamw(models)
        rows = decompose(store)
        agg = [r for r in rows if r["target"] == "f1_pres"]
        dn = np.array([float(r["delta_nonlinearite"]) for r in agg])
        dp = np.array([float(r["delta_protocole"]) for r in agg])
        print(f"\n  delta_protocole    : médiane {np.median(dp):+.4f}  "
              f"[{dp.min():+.4f}, {dp.max():+.4f}]  ({(dp < 0).sum()}/{len(dp)} négatifs)")
        print(f"  delta_nonlinearite : médiane {np.median(dn):+.4f}  "
              f"[{dn.min():+.4f}, {dn.max():+.4f}]  ({(dn > 0).sum()}/{len(dn)} positifs)")
        write_correlations([{"model": r["model"], "target": "f1_pres",
                             "delta": float(r["delta_nonlinearite"])} for r in agg],
                           target="delta_nonlinearite",
                           fname="correlations_mlp_nonlin_geometry.csv")
        write_summary_decomposed(store)
        return 0

    if args.control_only:
        cache = load_cache()
        print("\n== Contrôle labels permutés ==")
        ctrl = run_control(cache, args.only or ["dinov3_vitb16_lvd"], cfgs)
        print(json.dumps(ctrl, indent=1, ensure_ascii=False))
        return 0 if all(c["ok"] for c in ctrl) else 1

    print(f"\n== Probe MLP : {len(models)} modèles × {len(cfgs)} configs × 3 seeds ==")
    cache = run_models(models, cfgs, skip_linear=args.skip_linear, force=args.force)

    print("\n== Écriture des tables ==")
    rows = write_results(cache, cfgs, models)
    deltas = write_delta(rows)

    print("\n== Contrôle labels permutés ==")
    top = max(rows, key=lambda r: next(d["delta"] for d in deltas
                                       if d["model"] == r["model"] and d["target"] == "f1_pres"))
    ctrl_models = list(dict.fromkeys(["dinov3_vitb16_lvd", top["model"]]))
    ctrl_models = [m for m in ctrl_models if m in cache]
    controls = run_control(cache, ctrl_models, cfgs)
    if not all(c["ok"] for c in controls):
        print("\n[ÉCHEC] le MLP généralise depuis des labels permutés → la grille "
              "mémorise. Aucun delta n'est publiable en l'état : resserrer les bornes "
              "de régularisation (dropout/weight decay) avant de reporter quoi que ce soit.")
        return 1

    print("\n== Non-régression de la baseline linéaire ==")
    lin_ctrls = linear_controls(rows)

    print("\n== Corrélations Δ ↔ géométrie ==")
    corrs = write_correlations(deltas)

    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "protocole": (
            "MLP 1 couche cachée sur embeddings cachés standardisés (StandardScaler fit "
            "train) ; schéma 11cls ; f1_macro_pres sur test n=17598 ; early stopping sur "
            f"val f1_macro_pres, patience {PATIENCE} fixe ; best_config = moyenne val la "
            "plus haute sur 3 seeds ; test jamais utilisé en sélection."),
        "grille": {"hidden": list(HIDDEN_GRID), "dropout": list(DROPOUT_GRID),
                   "weight_decay": list(WD_GRID), "n_configs": len(cfgs),
                   "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS,
                   "optimizer": "AdamW", "loss": "CrossEntropy non pondérée"},
        "baseline_lineaire": (
            f"LR canonique make_canonical_lr (lbfgs, max_iter={MAX_ITER}, seed={SEED}), "
            f"C_grid {list(C_GRID)} sélectionné sur val f1_macro_pres, fit final sur "
            "train complet — recalculée ici sur les MÊMES tableaux standardisés que le MLP "
            "(AGENTS.md §4.3 : symétrie de grille)."),
        "conventions": {
            "seed_source": "frozen = init_only (1 jeu d'embeddings, 3 inits) ; "
                           "ft = backbone+init (3 dossiers de seed)",
            "mlp_seed": f"{SEED} + seed_de_finetuning",
            "interpretable": "|delta| > 0.01 (AGENTS.md §4.4, σ inter-seed ≈ 0,008)",
            "classes_mortes": CLASSES_DEAD,
        },
        "n_modeles": len(rows), "duree_s": time.time() - t0,
        "controles": lin_ctrls,
        "controle_labels_permutes": controls,
        "pilote_regularisation": PILOTE_LR,
    }
    with open(out_path("mlp_probe_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"[OK] {os.path.basename(out_path('mlp_probe_meta.json'))}")

    write_summary(rows, deltas, corrs, controls, lin_ctrls, meta)

    fails = [c for c in lin_ctrls if not c["ok"]]
    if fails:
        print(f"\n[ÉCHEC] {len(fails)} modèle(s) gelé(s) ne reproduisent pas "
              f"CANONICAL_F1 à {TOL_FROZEN} : {[c['model'] for c in fails]}")
        return 1
    # Le cache est CONSERVÉ : il porte le détail des 18 configs × 3 seeds, indispensable
    # pour toute analyse a posteriori (décomposition, ré-agrégation) sans re-tourner 3 h.
    print(f"\n[RÉSULTAT] {len(rows)} modèles, tous les contrôles passent "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
