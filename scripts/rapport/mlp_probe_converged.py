#!/usr/bin/env python3
"""Probe MLP à convergence — isole la NON-LINÉARITÉ du coût de l'optimiseur.

Pourquoi ce second run : dans `mlp_probe.py`, le MLP est entraîné par AdamW avec
early stopping, alors que le probe linéaire canonique est résolu à l'optimum par
lbfgs (`max_iter=2000`). Le témoin sans ReLU montre qu'un readout **strictement
linéaire** entraîné par la même boucle perd déjà 0,006–0,008 de F1 face au lbfgs :
le delta `MLP − probe` mélange donc deux causes, et son signe est porté par
l'optimiseur, pas par la non-linéarité.

Ici les deux readouts partagent le MÊME objectif et le MÊME critère d'arrêt :

    minimiser   CE_moyenne + ‖W‖² / (2·C·n)          (biais non pénalisés)

qui est exactement l'objectif de `LogisticRegression(C=...)` de sklearn
(sklearn minimise ``0.5·wᵀw + C·Σ logloss`` — même optimum à un facteur près).
Optimiseur : L-BFGS full-batch, line search Wolfe forte, jusqu'à stagnation du
gradient. **Ni dropout ni early stopping** : la seule régularisation est la L2,
sous la même forme et la même grille `C` que le probe canonique.

Garde-fou décisif (`--validate`) : le readout SANS couche cachée, entraîné par ce
code, doit reproduire le F1 de `sklearn.LogisticRegression` au même `C`. Si oui, le
protocole est prouvé équivalent en convergence, et l'écart MLP − linéaire mesuré
ensuite ne peut plus être imputé à l'optimisation.

    python3 scripts/rapport/mlp_probe_converged.py --validate      # garde-fou seul
    python3 scripts/rapport/mlp_probe_converged.py                 # run complet

Sorties : results/rapport_data/mlp_converged_{results.csv,meta.json}
Strictement additif — n'écrase aucune sortie de mlp_probe.py.
"""
from __future__ import annotations

import argparse
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
# converge (voir scripts/rapport/significance_tier.py). Doit précéder l'import
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

from registry import CANONICAL_F1, CLASSES_11, FROZEN_MODELS, OUT, ensure_out
from mlp_probe import (C_GRID, HIDDEN_GRID, MODELS, N_CLASSES, SEED, SEEDS,
                       display_of, emb_sets, load_set, out_path, type_of)
from src.metrics import eval_classifier, per_class_f1

# L-BFGS : blocs de LBFGS_ITERS itérations, répétés jusqu'à stagnation RELATIVE de la
# perte ou épuisement du plafond.
#
# Historique (2026-07-26) : la première version utilisait 3 blocs de 100 itérations et
# lisait la valeur de retour de ``opt.step()``. DEUX bugs, corrigés ici :
#   1. ``torch.optim.LBFGS.step`` renvoie ``orig_loss``, la perte AVANT ses itérations
#      internes — le critère portait donc sur une perte retardée d'un bloc.
#   2. ``prev`` était réassigné en fin de boucle, si bien que le ``abs(prev-cur)``
#      évalué APRÈS la boucle valait toujours 0 : le drapeau ``converged`` était
#      vrai par construction.
# Conséquence mesurée : le readout linéaire convergeait (Δ 6e-4 sur le dernier bloc)
# mais le MLP était coupé alors qu'il progressait encore 10× plus vite (Δ 6,5e-3) —
# soit exactement le biais d'optimisation asymétrique que ce script doit éliminer.
# Ici la perte est RE-ÉVALUÉE après chaque bloc, et la non-convergence est bloquante.
LBFGS_ROUNDS = 40        # plafond de blocs (le MLP en demande nettement plus de 3)
LBFGS_ITERS = 50         # granularité plus fine : un fit court ne paie plus 100 iters
TOL_REL = 1e-6           # stagnation relative : |Δperte| < TOL_REL · max(1, |perte|)
VALIDATE_TOL = 2e-3      # écart max toléré entre readout linéaire torch et sklearn

# Ordre de passage : 6 modèles couvrant toute la plage de F1 et les deux régimes
# d'abord (réponse exploitable en ~4 h), puis le reste.
PRIORITY = ["dinov3_vitl16_lvd", "dinov3_vitb16_lvd", "dinov3_vitl16_sat",
            "satmae_vitl16", "dinov3_vitb16_lvd_full", "vitb16_scratch_old"]
ORDERED = PRIORITY + [m for m in MODELS if m not in PRIORITY]


def _torch():
    import torch
    torch.set_num_threads(8)
    return torch


def fit_converged(data: dict, hidden: int | None, C: float, seed: int) -> dict:
    """Ajuste un readout à convergence par L-BFGS full-batch.

    ``hidden=None`` → readout linéaire ``Linear(D,11)`` (doit retrouver sklearn).
    ``hidden=H``    → MLP ``Linear(D,H) → ReLU → Linear(H,11)``.
    Régularisation L2 sur les poids uniquement, coefficient ``1/(2·C·n)`` — même
    convention que ``LogisticRegression(C=C)``.
    """
    torch = _torch()
    nn = torch.nn
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]

    def _t(a, dt=np.float32):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=dt))

    Xtr, Ytr = _t(xtr), _t(ytr, np.int64)
    n = Xtr.shape[0]
    torch.manual_seed(seed)
    net = (nn.Linear(data["dim"], N_CLASSES) if hidden is None else
           nn.Sequential(nn.Linear(data["dim"], hidden), nn.ReLU(),
                         nn.Linear(hidden, N_CLASSES)))
    weights = [p for nm, p in net.named_parameters() if nm.endswith("weight")]
    l2 = 1.0 / (2.0 * C * n)
    lossf = nn.CrossEntropyLoss()
    # Tolérances internes desserrées : à 1e-9/1e-12 les arrêts anticipés de L-BFGS ne
    # se déclenchaient jamais et chaque bloc consommait ses itérations en entier.
    opt = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=LBFGS_ITERS,
                            history_size=10, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = lossf(net(Xtr), Ytr) + l2 * sum((w * w).sum() for w in weights)
        loss.backward()
        return loss

    def objective() -> float:
        """Perte RÉELLE à l'état courant — ne jamais utiliser le retour de step()."""
        with torch.no_grad():
            return float(lossf(net(Xtr), Ytr)
                         + l2 * sum((w * w).sum() for w in weights))

    prev, converged, rounds = objective(), False, 0
    for rounds in range(1, LBFGS_ROUNDS + 1):
        opt.step(closure)
        cur = objective()
        if abs(prev - cur) < TOL_REL * max(1.0, abs(cur)):
            converged = True
            break
        prev = cur

    net.eval()
    with torch.no_grad():
        pva = net(_t(xva)).argmax(1).numpy()
        pte = net(_t(xte)).argmax(1).numpy()
    return {"hidden": 0 if hidden is None else hidden, "C": C, "seed": seed,
            "loss": cur, "rounds": rounds, "n_iter_max": rounds * LBFGS_ITERS,
            "converged": bool(converged),
            "val_f1_pres": eval_classifier(yva, pva, N_CLASSES)["f1_macro_pres"],
            "f1_pres": eval_classifier(yte, pte, N_CLASSES)["f1_macro_pres"],
            "accuracy": eval_classifier(yte, pte, N_CLASSES)["accuracy"],
            "per_class": per_class_f1(yte, pte, CLASSES_11)}


def sklearn_linear(data: dict, C: float) -> float:
    """F1 test du probe canonique sklearn au même C (référence du garde-fou)."""
    from src.utils import make_canonical_lr
    xtr, ytr = data["train"]
    clf = make_canonical_lr(C=C, max_iter=2000, random_state=SEED)
    clf.fit(np.asarray(xtr, dtype=np.float64), ytr)
    yp = clf.predict(np.asarray(data["test"][0], dtype=np.float64))
    return eval_classifier(data["test"][1], yp, N_CLASSES)["f1_macro_pres"]


def validate(models: list[str]) -> list[dict]:
    """Le readout linéaire torch reproduit-il sklearn ? Sans ça, rien n'est comparable."""
    out = []
    for key in models:
        data = load_set(emb_sets(key)[0]["paths"])
        for C in (0.001, 0.01):
            t = fit_converged(data, None, C, SEED)
            s = sklearn_linear(data, C)
            ok = abs(t["f1_pres"] - s) <= VALIDATE_TOL
            out.append({"model": key, "C": C, "torch_lineaire": t["f1_pres"],
                        "sklearn_lbfgs": s, "abs_err": abs(t["f1_pres"] - s),
                        "converged": t["converged"], "ok": bool(ok)})
            print(f"  [{'OK  ' if ok else 'FAIL'}] {key:26s} C={C:<7g} "
                  f"torch={t['f1_pres']:.4f}  sklearn={s:.4f}  "
                  f"|Δ|={abs(t['f1_pres'] - s):.4f}  conv={t['converged']}")
        data = None
        gc.collect()
    return out


def run(models: list[str]) -> dict:
    path = out_path("mlp_converged_partial.json")
    store = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for i, key in enumerate(models, 1):
        if key in store:
            print(f"[{i:2d}/{len(models)}] {key:26s} en cache → SKIP")
            continue
        by_tag: dict[str, list[dict]] = {}
        for st in emb_sets(key):
            by_tag.setdefault(st["tag"], []).append(st)
        lin_runs, mlp_runs = [], []
        for tag, sts in by_tag.items():
            data = load_set(sts[0]["paths"])
            # Linéaire : objectif convexe → indépendant du seed, un fit par C.
            best_lin = max((fit_converged(data, None, C, SEED) for C in C_GRID),
                           key=lambda r: r["val_f1_pres"])
            lin_runs += [best_lin] * len(sts) if len(by_tag) == 1 else [best_lin]
            # MLP : grille hidden × C, 3 seeds d'init, sélection sur moyenne val.
            best, best_val = None, -1.0
            for h in HIDDEN_GRID:
                for C in C_GRID:
                    rs = [fit_converged(data, h, C, SEED + st["seed"]) for st in sts]
                    mv = float(np.mean([r["val_f1_pres"] for r in rs]))
                    if mv > best_val:
                        best_val, best = mv, rs
            mlp_runs += best
            data = None
            gc.collect()
        rec = {
            "model": key, "display": display_of(key), "type": type_of(key),
            "lin_C": lin_runs[0]["C"],
            "lin_f1_pres": float(np.mean([r["f1_pres"] for r in lin_runs])),
            "lin_per_class": {c: float(np.mean([r["per_class"][c] for r in lin_runs]))
                              for c in CLASSES_11},
            "mlp_hidden": mlp_runs[0]["hidden"], "mlp_C": mlp_runs[0]["C"],
            "mlp_f1_pres_mean": float(np.mean([r["f1_pres"] for r in mlp_runs])),
            "mlp_f1_pres_std": (float(np.std([r["f1_pres"] for r in mlp_runs], ddof=1))
                                if len(mlp_runs) > 1 else 0.0),
            "mlp_per_class": {c: float(np.mean([r["per_class"][c] for r in mlp_runs]))
                              for c in CLASSES_11},
            "n_converged": int(sum(r["converged"] for r in lin_runs + mlp_runs)),
            "n_fits": len(lin_runs) + len(mlp_runs),
        }
        rec["delta_nonlinearite"] = rec["mlp_f1_pres_mean"] - rec["lin_f1_pres"]
        store[key] = rec
        with open(path, "w") as f:
            json.dump(store, f)
        print(f"[{i:2d}/{len(models)}] {key:26s} lin={rec['lin_f1_pres']:.4f} "
              f"MLP={rec['mlp_f1_pres_mean']:.4f}±{rec['mlp_f1_pres_std']:.4f}  "
              f"Δ_nonlin={rec['delta_nonlinearite']:+.4f}  "
              f"(h={rec['mlp_hidden']}, conv {rec['n_converged']}/{rec['n_fits']}, "
              f"{time.time() - t0:.0f}s)")
    return store


def write(store: dict, val_ctrl: list[dict], t0: float) -> int:
    """Écrit les tables. Retourne un code non nul si des ajustements n'ont pas convergé.

    Une convergence partielle réintroduit exactement le biais que ce script élimine
    (le MLP, plus dur à optimiser, s'arrête plus loin de son optimum que le linéaire) :
    le résultat est alors inexploitable comme mesure de la non-linéarité, et doit être
    signalé, pas publié en silence.
    """
    rows = sorted(store.values(), key=lambda r: -r["delta_nonlinearite"])
    incomplet = [r for r in rows if r["n_converged"] < r["n_fits"]]
    cols = ["model", "display", "type", "lin_C", "lin_f1_pres", "mlp_hidden", "mlp_C",
            "mlp_f1_pres_mean", "mlp_f1_pres_std", "delta_nonlinearite",
            "n_converged", "n_fits"]
    for c in CLASSES_11:
        cols += [f"lin_{c}", f"mlp_{c}", f"delta_{c}"]
    with open(out_path("mlp_converged_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in cols}
            for c in CLASSES_11:
                row[f"lin_{c}"] = r["lin_per_class"][c]
                row[f"mlp_{c}"] = r["mlp_per_class"][c]
                row[f"delta_{c}"] = r["mlp_per_class"][c] - r["lin_per_class"][c]
            w.writerow(row)
    print(f"[OK] mlp_converged_results.csv ({len(rows)} modèles)")

    d = np.array([r["delta_nonlinearite"] for r in rows])
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "protocole": ("readouts linéaire ET MLP ajustés au MÊME objectif "
                      "(CE + ‖W‖²/(2·C·n), biais non pénalisés) par L-BFGS full-batch, "
                      "ni dropout ni early stopping ; grille C identique au probe "
                      "canonique ; MLP = Linear(D,H) → ReLU → Linear(H,11)."),
        "pourquoi": ("le delta de mlp_probe.py mélange non-linéarité et optimiseur ; "
                     "ici les deux readouts partagent objectif et convergence, donc "
                     "l'écart mesure la non-linéarité seule."),
        "garde_fou_sklearn": val_ctrl,
        "delta_nonlinearite": {
            "mediane": float(np.median(d)), "min": float(d.min()), "max": float(d.max()),
            "n_positifs": int((d > 0).sum()), "n_modeles": len(d),
            "n_interpretables": int((np.abs(d) > 0.01).sum()),
        },
        "n_modeles": len(rows), "duree_s": time.time() - t0,
    }
    meta["convergence"] = {
        "critere": f"|Δperte| < {TOL_REL} · max(1, |perte|), perte ré-évaluée après "
                   f"chaque bloc de {LBFGS_ITERS} itérations (plafond {LBFGS_ROUNDS} blocs)",
        "modeles_incomplets": [r["model"] for r in incomplet],
    }
    with open(out_path("mlp_converged_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("[OK] mlp_converged_meta.json")
    print(f"\n  Δ non-linéarité (protocole apparié) : médiane {np.median(d):+.4f}  "
          f"[{d.min():+.4f}, {d.max():+.4f}]  "
          f"{int((d > 0).sum())}/{len(d)} positifs, "
          f"{int((np.abs(d) > 0.01).sum())} interprétables (>0,01)")
    if incomplet:
        print(f"\n[ÉCHEC] {len(incomplet)} modèle(s) avec des ajustements NON convergés "
              f"({', '.join(r['model'] for r in incomplet)}) : le MLP s'arrête plus loin "
              "de son optimum que le linéaire, donc le delta reste confondu avec "
              f"l'optimisation. Augmenter LBFGS_ROUNDS (actuellement {LBFGS_ROUNDS}). "
              "Ne PAS citer ces chiffres comme mesure de la non-linéarité.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="garde-fou seul : readout linéaire torch vs sklearn")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    ensure_out()
    t0 = time.time()
    models = args.only or ORDERED

    print("== Garde-fou : readout linéaire torch == sklearn lbfgs ? ==")
    val_ctrl = validate(models[:2])
    if not all(c["ok"] for c in val_ctrl):
        print(f"\n[ÉCHEC] le readout linéaire torch ne reproduit pas sklearn à "
              f"{VALIDATE_TOL} : le protocole n'est PAS équivalent en convergence, "
              "l'écart MLP − linéaire resterait confondu avec l'optimisation. "
              "Aucun résultat publié.")
        return 1
    if args.validate:
        return 0

    print(f"\n== Readouts à convergence : {len(models)} modèles "
          f"({len(HIDDEN_GRID)}×{len(C_GRID)} configs MLP × 3 seeds + {len(C_GRID)} linéaires) ==")
    store = run(models)
    return write(store, val_ctrl, t0)


if __name__ == "__main__":
    raise SystemExit(main())
