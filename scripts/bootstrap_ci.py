"""Intervalles de confiance bootstrap sur les prédictions du linear probe.

Les résultats probe sauvegardés (``results/<pass>/probe_knn.json``) ne contiennent que
des scalaires (best_C, métriques) — pas les prédictions brutes. On re-fitte donc le
probe UNE fois par modèle avec le ``best_C`` déjà sélectionné (pas de re-grille C), on
récupère ``(y_pred, y_true)`` sur le test, puis on bootstrappe le test set :

  - N tirages avec remise de ``len(y_true)`` indices (seed=42, reproductible) ;
  - recalcul de ``f1_macro_all`` (labels=range(n_classes) de la passe) et
    ``f1_macro_pres`` (classes présentes dans le tirage) à chaque itération ;
  - IC à 95 % par percentiles bootstrap (2.5 / 97.5), pas par formule normale.

PASSE DE CLASSES (``--pass``) : le re-fit doit reproduire EXACTEMENT la réduction de
classes appliquée par ``probe.py`` pour la passe visée. Les embeddings sur disque sont
en schéma SOURCE (12cls pour les canoniques, 11cls pour les runs SOTA) ; les passes
``without_rhol`` (11cls) et ``8cls`` (8cls) sont obtenues en RETIRANT des classes via
``src.latent.drop_class`` (cascade décroissante ``src.utils.pass_drops``), ce qui
COMPACTE les indices restants (ex. 8cls → labels 0..7). Sans cette cascade, un simple
``--n-classes 8`` fitterait un classifieur 11/12 classes et calculerait ``f1_macro_all``
sur ``range(8)`` = les MAUVAISES classes (ALDE,ARCA,BIRC,DRYI,LICH,MOSS,PETF,RHOL en
12cls). ``--pass`` remplace donc ``--n-classes`` (voir known_issues.md).

  # schéma A (défaut, rétro-compatible) : passe with_rhol, 12 classes, aucun drop
  python scripts/bootstrap_ci.py --config configs/frozen_eval.yaml --n-bootstrap 1000
  # schéma 8cls (source /scratch/.../8cls/probe_knn.json)
  python scripts/bootstrap_ci.py --config configs/probe_all.yaml --pass 8cls \\
      --probe-json /scratch/lmague/datacurve/results/8cls/probe_knn.json --n-bootstrap 1000
  python scripts/bootstrap_ci.py --include-finetuned --n-bootstrap 10   # dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Racine du projet sur le path quand le script est lancé directement (python scripts/...).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import _sota_run_dir, load_features
from src.latent import drop_class
from src.probe import _standardize
from src.utils import is_sota_key, make_canonical_lr, pass_drops, probe_passes, source_schema

# Passe par défaut = with_rhol (12 classes, AUCUN drop) — comportement schéma A
# historique. NE PAS changer sans casser les runs existants (with_rhol/probe_knn*.json).
N_CLASSES = 12
DEFAULT_PASS = "with_rhol"
# {tag: liste des noms de classes de la passe} → n_classes = len(names).
_PASS_NAMES: dict[str, list[str]] = {tag: names for tag, names in probe_passes()}
# Paire clé du mémoire : représentation gelée vs backbone fine-tuné.
PAIR_FROZEN = "dinov3_vitl16_lvd"
PAIR_FINETUNED = "vitb16_fulft_arctic"


def _load_best_C(probe_json: str) -> dict[str, float]:
    """Lit ``best_C`` par modèle depuis un ``probe_knn.json`` existant (clé ``probe``)."""
    with open(probe_json) as f:
        data = json.load(f)
    probe = data.get("probe", data)
    return {m: float(d["best_C"]) for m, d in probe.items() if "best_C" in d}


def _refit_predict(cfg, model_key: str, best_C: float, drops=(), seed: int = 42):
    """Re-fitte LogisticRegression (best_C fixé) et renvoie ``(y_true, y_pred)`` sur le test.

    StandardScaler fit sur train (via :func:`src.probe._standardize`), C non re-griddé.
    ``drops`` : cascade DÉCROISSANTE de labels à retirer AVANT le fit (identique à
    ``probe.py`` : réduit le schéma source vers la passe visée et compacte les indices).
    Passer ``()`` (défaut) = aucune réduction = passe with_rhol / schéma A.
    """
    feats = load_features(cfg, model_key)
    for d in drops:  # cascade décroissante (ordre imposé par utils.pass_drops)
        feats = {s: drop_class(*feats[s], d) for s in feats}
    ytr = np.asarray(feats["train"][1])
    yte = np.asarray(feats["test"][1])
    xtr, _xva, xte = _standardize(feats)
    clf = make_canonical_lr(C=best_C, max_iter=cfg.probe.max_iter, random_state=seed)
    clf.fit(xtr, ytr)
    return yte, clf.predict(xte)


def _bootstrap_metrics(y_true, y_pred, n_boot: int, n_classes: int = N_CLASSES,
                       seed: int = 42):
    """Bootstrap (tirage avec remise) du test set : renvoie deux tableaux (N,).

    Colonne 0 = f1_macro_all (labels=range(n_classes)), colonne 1 = f1_macro_pres
    (labels présents dans le tirage). RandomState ré-initialisé au seed pour que la
    séquence de tirages soit identique entre modèles (comparaison appariée si tailles
    égales).
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]
    all_labels = list(range(n_classes))
    rng = np.random.RandomState(seed)
    f1_all = np.empty(n_boot, dtype=np.float64)
    f1_pres = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        present = sorted({int(v) for v in yt})
        f1_all[b] = f1_score(yt, yp, average="macro", zero_division=0, labels=all_labels)
        f1_pres[b] = f1_score(yt, yp, average="macro", zero_division=0, labels=present)
    return f1_all, f1_pres


def _summary(samples: np.ndarray, observed: float | None = None) -> dict:
    """mean / std / IC95 (percentiles 2.5 et 97.5) d'un tableau bootstrap."""
    return {
        "observed": float(observed) if observed is not None else None,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


def _observed_f1(y_true, y_pred, n_classes: int = N_CLASSES) -> tuple[float, float]:
    """f1_macro_all et f1_macro_pres sur le test complet (point estimate)."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    present = sorted({int(v) for v in y_true})
    f1_all = f1_score(y_true, y_pred, average="macro", zero_division=0,
                      labels=list(range(n_classes)))
    f1_pres = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present)
    return float(f1_all), float(f1_pres)


def run(cfg, probe_json: str, n_boot: int, include_finetuned: bool,
        seed: int = 42, force_c: float | None = None,
        pairs: list[tuple[str, str]] | None = None,
        pass_tag: str = DEFAULT_PASS) -> dict:
    """Bootstrappe tous les modèles disponibles et construit le dict de résultats.

    ``force_c``  : si fourni, ce C est utilisé pour TOUS les modèles (bypass de la
    lecture des best_C dans ``probe_json`` et du skip "aucun best_C").
    ``pairs``    : liste de (model_a, model_b) à comparer via ``_compare_pair_generic``.
    ``pass_tag`` : passe de classes (``with_rhol`` | ``without_rhol`` | ``8cls``). Chaque
    modèle est réduit selon SA source (12cls canonique / 11cls SOTA) via
    :func:`src.utils.pass_drops`, exactement comme ``probe.py`` (with_rhol non applicable
    aux runs SOTA → SKIP). ``n_classes`` est dérivé de la passe.
    """
    if pass_tag not in _PASS_NAMES:
        raise ValueError(f"--pass inconnu : {pass_tag!r} (dispo: {sorted(_PASS_NAMES)})")
    n_classes = len(_PASS_NAMES[pass_tag])
    best_C = {} if force_c is not None else _load_best_C(probe_json)

    def _get_c(model_key: str):
        return force_c if force_c is not None else best_C.get(model_key)

    wanted = list(cfg.models)
    if include_finetuned:
        wanted += list(cfg.finetuned_models)
    # Les deux modèles de la paire clé sont toujours tentés (pour la comparaison).
    for m in (PAIR_FROZEN, PAIR_FINETUNED):
        if m not in wanted:
            wanted.append(m)
    # Les modèles des paires explicites sont aussi tentés.
    if pairs:
        for a, b in pairs:
            for m in (a, b):
                if m not in wanted:
                    wanted.append(m)

    emb_dir = cfg.paths.emb_dir
    results: dict[str, dict] = {}
    # On conserve les tableaux bootstrap f1_macro_pres pour la comparaison appariée.
    pres_samples: dict[str, np.ndarray] = {}

    for model_key in wanted:
        c = _get_c(model_key)
        if c is None:
            print(f"[skip] {model_key}: aucun best_C dans {os.path.basename(probe_json)}")
            continue
        # Résolution du chemin test SOURCE-AWARE : les runs SOTA (is_sota_key) vivent
        # dans sota_dir/{regime}/embeddings/{key}/test.npy (convention nue), pas dans
        # emb_dir/{key}_test.npy — sans ce branchement, ils seraient RE-SKIPPÉS ici avant
        # même que load_features (déjà SOTA-aware) ne soit appelé.
        if is_sota_key(model_key):
            test_emb = os.path.join(_sota_run_dir(cfg, model_key), "test.npy")
        else:
            test_emb = os.path.join(emb_dir, f"{model_key}_test.npy")
        if not os.path.exists(test_emb):
            print(f"[skip] {model_key}: embeddings absents ({test_emb})")
            continue
        # Réduction de classes source-aware (identique à probe.py) : None = passe non
        # applicable à cette source (ex. with_rhol sur un run SOTA 11cls) → SKIP.
        schema = source_schema(model_key)
        drops = pass_drops(pass_tag, schema)
        if drops is None:
            print(f"[skip] {model_key}: passe {pass_tag} non applicable (source {schema})")
            continue
        print(f"[bootstrap] {model_key}: re-fit (C={c}, pass={pass_tag}, "
              f"drops={drops}) + {n_boot} tirages")
        y_true, y_pred = _refit_predict(cfg, model_key, c, drops=drops, seed=seed)
        obs_all, obs_pres = _observed_f1(y_true, y_pred, n_classes=n_classes)
        f1_all, f1_pres = _bootstrap_metrics(y_true, y_pred, n_boot,
                                             n_classes=n_classes, seed=seed)
        pres_samples[model_key] = f1_pres
        results[model_key] = {
            "best_C": c,
            "n_test": int(np.asarray(y_true).shape[0]),
            "f1_macro_all": _summary(f1_all, obs_all),
            "f1_macro_pres": _summary(f1_pres, obs_pres),
        }

    comparison = _compare_pair(results, pres_samples)
    out = {
        "config_models": wanted,
        "n_bootstrap": n_boot,
        "seed": seed,
        "pass": pass_tag,
        "n_classes": n_classes,
        "probe_source": probe_json,
        "remap_v3": cfg.features.remap_v3,
        "models": results,
        "comparison": comparison,
    }
    if pairs:
        out["pairs"] = [
            _compare_pair_generic(a, b, results, pres_samples)
            for a, b in pairs
        ]
    return out


def _compare_pair(results: dict, pres_samples: dict[str, np.ndarray]) -> dict:
    """Compare la paire clé (frozen vs fine-tuné) sur f1_macro_pres."""
    fz, ft = PAIR_FROZEN, PAIR_FINETUNED
    base = {"frozen": fz, "finetuned": ft}
    if fz not in results or ft not in results:
        missing = [m for m in (fz, ft) if m not in results]
        return {**base, "available": False,
                "reason": f"modèle(s) indisponible(s) dans l'état actuel : {missing}"}

    fz_pres = results[fz]["f1_macro_pres"]
    ft_pres = results[ft]["f1_macro_pres"]
    delta = fz_pres["observed"] - ft_pres["observed"]

    a, b = pres_samples[fz], pres_samples[ft]
    if a.shape == b.shape:
        # Comparaison appariée (même séquence de tirages, mêmes tailles).
        p_frozen_gt = float(np.mean(a > b))
    else:
        # Tailles différentes : comparaison non appariée (produit des deux distributions).
        p_frozen_gt = float(np.mean(a[:, None] > b[None, :]))

    # IC95 disjoints -> distinguishable.
    disjoint = (fz_pres["ci95_low"] > ft_pres["ci95_high"]) or \
               (ft_pres["ci95_low"] > fz_pres["ci95_high"])
    return {
        **base,
        "available": True,
        "delta_observed": float(delta),
        "p_frozen_gt_finetuned": p_frozen_gt,
        "ci95_frozen": [fz_pres["ci95_low"], fz_pres["ci95_high"]],
        "ci95_finetuned": [ft_pres["ci95_low"], ft_pres["ci95_high"]],
        "conclusion": "distinguishable" if disjoint else "not distinguishable",
    }


def _compare_pair_generic(model_a: str, model_b: str,
                          results: dict, pres_samples: dict[str, np.ndarray]) -> dict:
    """Compare deux modèles quelconques sur f1_macro_pres (bootstrap apparié)."""
    base = {"model_a": model_a, "model_b": model_b}
    if model_a not in results or model_b not in results:
        missing = [m for m in (model_a, model_b) if m not in results]
        return {**base, "available": False,
                "reason": f"modèle(s) indisponible(s) : {missing}"}

    a_pres = results[model_a]["f1_macro_pres"]
    b_pres = results[model_b]["f1_macro_pres"]
    delta = a_pres["observed"] - b_pres["observed"]

    sa, sb = pres_samples[model_a], pres_samples[model_b]
    assert sa.shape == sb.shape and len(sa) > 0, (
        f"shapes incompatibles pour comparaison appariée : {sa.shape} vs {sb.shape}"
    )
    p_a_gt_b = float(np.mean(sa > sb))

    disjoint = (a_pres["ci95_low"] > b_pres["ci95_high"]) or \
               (b_pres["ci95_low"] > a_pres["ci95_high"])
    return {
        **base,
        "available": True,
        "delta_observed_a_minus_b": float(delta),
        "p_a_gt_b": p_a_gt_b,
        "ci95_a": [a_pres["ci95_low"], a_pres["ci95_high"]],
        "ci95_b": [b_pres["ci95_low"], b_pres["ci95_high"]],
        "ci95_disjoint": disjoint,
        "conclusion": "distinguishable" if disjoint else "not distinguishable",
    }


def _print_table(results: dict) -> None:
    """Tableau récapitulatif trié par f1_macro_pres mean (décroissant)."""
    rows = sorted(results.items(), key=lambda kv: kv[1]["f1_macro_pres"]["mean"],
                  reverse=True)
    print(f"\n{'modèle':<24} {'pres mean':>10} {'std':>8} {'IC95 bas':>10} {'IC95 haut':>10}")
    print("-" * 66)
    for name, r in rows:
        p = r["f1_macro_pres"]
        print(f"{name:<24} {p['mean']:>10.4f} {p['std']:>8.4f} "
              f"{p['ci95_low']:>10.4f} {p['ci95_high']:>10.4f}")


def _print_comparison(cmp: dict) -> None:
    print(f"\nPaire clé : {cmp['frozen']} (gelé) vs {cmp['finetuned']} (fine-tuné)")
    if not cmp.get("available"):
        print(f"  indisponible — {cmp['reason']}")
        return
    print(f"  Δ observé (f1_macro_pres)   : {cmp['delta_observed']:+.4f}")
    print(f"  P(gelé > fine-tuné)         : {cmp['p_frozen_gt_finetuned']:.3f}")
    print(f"  IC95 gelé                   : "
          f"[{cmp['ci95_frozen'][0]:.4f}, {cmp['ci95_frozen'][1]:.4f}]")
    print(f"  IC95 fine-tuné              : "
          f"[{cmp['ci95_finetuned'][0]:.4f}, {cmp['ci95_finetuned'][1]:.4f}]")
    print(f"  Conclusion                  : {cmp['conclusion']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/frozen_eval.yaml",
                    help="config YAML (défaut: configs/frozen_eval.yaml)")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="nombre de tirages bootstrap (défaut: 1000)")
    ap.add_argument("--pass", dest="pass_tag", default=DEFAULT_PASS,
                    choices=sorted(_PASS_NAMES),
                    help="passe de classes (défaut: with_rhol = schéma A, 12cls, aucun "
                         "drop). '8cls' exige --probe-json .../8cls/probe_knn.json. "
                         "REMPLACE --n-classes : le re-fit applique la MÊME cascade de "
                         "drops source-aware que probe.py (sinon f1_macro_all ET "
                         "f1_macro_pres seraient faux hors with_rhol).")
    ap.add_argument("--include-finetuned", action="store_true",
                    help="inclure les modèles fine-tunés (resnet50_arctic, etc.)")
    ap.add_argument("--probe-json", default=None,
                    help="probe JSON source des best_C. "
                         "DÉFAUT = <results_dir>/with_rhol/probe_knn_cgrid.json "
                         "(canonique, grille étendue C∈{1e-4..10}). "
                         "Ancienne valeur probe_knn.json (grille restreinte, "
                         "F1≈0.4675 périmé) marquée comme .deprecated.")
    ap.add_argument("--force-c", type=float, default=None,
                    help="forcer ce C pour TOUS les modèles (bypass best_C du JSON "
                         "et du skip 'aucun best_C')")
    ap.add_argument("--output", default="results/bootstrap_ci.json",
                    help="fichier de sortie JSON (défaut: results/bootstrap_ci.json)")
    ap.add_argument("--pairs", action="append", default=[], metavar="A:B",
                    help="paire à comparer A:B sur f1_macro_pres (répétable)")
    ap.add_argument("--pairs-output", default="results/bootstrap_pairs.json",
                    help="sortie JSON pour --pairs (défaut: results/bootstrap_pairs.json)")
    args = ap.parse_args()

    parsed_pairs: list[tuple[str, str]] = []
    for p in args.pairs:
        parts = p.split(":")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"--pairs : format attendu A:B, reçu '{p}'")
        parsed_pairs.append((parts[0], parts[1]))

    cfg = load_config(args.config)
    probe_json = args.probe_json or os.path.join(cfg.paths.results_dir,
                                                 "with_rhol", "probe_knn_cgrid.json")
    out = run(cfg, probe_json, args.n_bootstrap, args.include_finetuned,
              force_c=args.force_c, pairs=parsed_pairs if parsed_pairs else None,
              pass_tag=args.pass_tag)

    print(f"[pass] {out['pass']} ({out['n_classes']} classes) — source {probe_json}")
    _print_table(out["models"])
    _print_comparison(out["comparison"])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {args.output}  ({len(out['models'])} modèles)")

    if parsed_pairs:
        pairs_out = {
            "n_bootstrap": out["n_bootstrap"],
            "seed": out["seed"],
            "metric": "f1_macro_pres",
            "probe_source": out["probe_source"],
            "pairs": out.get("pairs", []),
            "models": out["models"],
        }
        pairs_path = os.path.abspath(args.pairs_output)
        os.makedirs(os.path.dirname(pairs_path), exist_ok=True)
        with open(pairs_path, "w") as f:
            json.dump(pairs_out, f, indent=2)
        print(f"[saved] {pairs_path}  ({len(parsed_pairs)} paires)")
        for cmp in pairs_out["pairs"]:
            a, b = cmp["model_a"], cmp["model_b"]
            if not cmp.get("available"):
                print(f"  {a}:{b}  SKIP — {cmp['reason']}")
            else:
                print(f"  {a} vs {b}  Δ={cmp['delta_observed_a_minus_b']:+.4f}  "
                      f"P(A>B)={cmp['p_a_gt_b']:.3f}  "
                      f"IC_disjoints={cmp['ci95_disjoint']}  {cmp['conclusion']}")


if __name__ == "__main__":
    main()
