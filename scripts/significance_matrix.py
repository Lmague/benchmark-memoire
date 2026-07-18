"""Matrice de significativité bootstrap — toutes paires entre groupes de modèles.

Gère les modèles multi-seed (fine-tunés avec N seeds) via bootstrap hiérarchique
(seed → tuile) et les modèles single-seed (gelés) via bootstrap simple sur les
tuiles.  Un groupe = un nom d'affichage + 1 à N clés de seed.

Lit best_C depuis --probe-json (probe_knn_cgrid.json), re-fitte + bootstrappe
chaque seed, puis agrège au niveau groupe et calcule toutes les paires K*(K-1)/2.

Sorties :
  --output-json  results/significance_matrix.json
  --output-png   results/significance_matrix.png  (heatmap P(A>B), RdBu_r)

Usage (groupes explicites) :
  python scripts/significance_matrix.py \\
      --config configs/frozen_eval.yaml \\
      --probe-json results/without_rhol/probe_knn_cgrid.json \\
      --group "DINOv3 ViT-L16:dinov3_vitl16_lvd" \\
      --group "SimDINOv2 ViT-L16:simdinov2_vitl16" \\
      --group "ViT-B Full FT:vitb16_full_frac100_seed0,vitb16_full_frac100_seed1,vitb16_full_frac100_seed2" \\
      --group "MHSA FT:vitb16_mhsa_frac100_seed0,vitb16_mhsa_frac100_seed1,vitb16_mhsa_frac100_seed2" \\
      --schema without_rhol \\
      --default-c 0.01 \\
      --n-bootstrap 10000

Usage (rétrocompatible : une clé = un groupe) :
  python scripts/significance_matrix.py \\
      --config configs/frozen_eval.yaml \\
      --probe-json results/with_rhol/probe_knn_cgrid.json \\
      --n-bootstrap 1000
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections import OrderedDict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.config import load_config
from src.features import load_features
from src.probe import _standardize
from src.utils import is_sota_key, make_canonical_lr, pass_drops, source_schema


# ================================================================== bootstrap utils


def _load_best_C(probe_json: str) -> dict[str, float]:
    """Lit best_C depuis un JSON probe_knn_cgrid (format canonique ou bootstrap_ci)."""
    with open(probe_json) as f:
        data = json.load(f)
    for section_key in ("probe", "models"):
        section = data.get(section_key)
        if section:
            return {m: float(d["best_C"]) for m, d in section.items() if "best_C" in d}
    # fallback : dict plat
    return {m: float(d["best_C"]) for m, d in data.items()
            if isinstance(d, dict) and "best_C" in d}


def _resolve_best_C(model_key: str, probe_best_C: dict[str, float],
                   default_c: float, sota_dir: str | None = None) -> float:
    """Retourne best_C pour une clé.

    Ordre : 1) probe_best_C, 2) metrics.json SOTA si sota_dir, 3) default_c.
    """
    if model_key in probe_best_C:
        return probe_best_C[model_key]

    # Tenter metrics.json SOTA (les dossiers runs utilisent le nom court: fracXXX_seedN)
    if sota_dir and is_sota_key(model_key):
        from src.utils import sota_regime as _sota_regime
        regime = _sota_regime(model_key)
        # Extraire le nom court: vitb16_full_frac100_seed0 → frac100_seed0
        short = re.search(r"(frac\d{3}_seed\d+)", model_key)
        run_dir = short.group(1) if short else model_key
        metrics_path = os.path.join(sota_dir, regime, "runs", run_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                d = json.load(f)
            c = d.get("best_C")
            if c is not None:
                print(f"[info] {model_key}: best_C={c} (metrics.json)")
                return float(c)

    print(f"[warn] {model_key}: best_C absent → default_c={default_c}")
    return default_c


def _refit_predict(cfg, model_key: str, best_C: float,
                   schema: str, seed: int = 42):
    """Re-fit LogisticRegression (C fixé) et renvoie (y_true, y_pred) sur le test.

    En schéma 11cls/8cls, les classes sont filtrées AVANT standardisation via
    ``drop_class`` (cascade décroissante), reproduisant probe.py --pass.
    """
    from src.latent import drop_class

    feats = load_features(cfg, model_key)
    src_schema = source_schema(model_key)

    # Appliquer drop_class selon la passe (sauf mode auto = schéma natif)
    if schema != "auto":
        drops = pass_drops(schema, src_schema)
        if drops:
            for d in drops:
                feats = {s: drop_class(*feats[s], d) for s in feats}

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xtr = sc.fit_transform(np.asarray(feats["train"][0], dtype=np.float32))
    xte = sc.transform(np.asarray(feats["test"][0], dtype=np.float32))
    ytr = np.asarray(feats["train"][1])
    yte = np.asarray(feats["test"][1])

    clf = make_canonical_lr(C=best_C, max_iter=cfg.probe.max_iter, random_state=seed)
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)

    return yte, y_pred


def _f1_pres(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 macro sur les classes présentes dans y_true."""
    from sklearn.metrics import f1_score
    present = sorted({int(v) for v in y_true})
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present))


# --- Indices de bootstrap partagés (appariement entre modèles) ---

def _make_bootstrap_indices(n_boot: int, n_tiles: int, seed: int = 42):
    """Pré-génère les indices de bootstrap : tuiles ET seed (pour hiérarchique).

    Retourne :
      tile_idx  : (n_boot, n_tiles) int32 — indices de tuiles avec remise
      seed_idx  : (n_boot,) int32        — indices de seed (0..N-1), utilisés
                  uniquement par les groupes multi-seed (modulo n_seeds)

    Ces indices sont PARTAGÉS par tous les modèles → bootstrap apparié.
    """
    rng = np.random.RandomState(seed)
    tile_idx = rng.randint(0, n_tiles, size=(n_boot, n_tiles), dtype=np.int32)
    seed_idx = rng.randint(0, 2**30, size=n_boot, dtype=np.int32)  # large range, mod n_seeds
    return tile_idx, seed_idx


def _bootstrap_tiles_paired(y_true: np.ndarray, y_pred: np.ndarray,
                            tile_idx: np.ndarray) -> np.ndarray:
    """Bootstrap simple sur tuiles, avec indices PRÉ-GÉNÉRÉS (apparié)."""
    n_boot = tile_idx.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = tile_idx[b]
        out[b] = _f1_pres(y_true[idx], y_pred[idx])
    return out


def _bootstrap_hierarchical_paired(
    seed_preds: list[tuple[np.ndarray, np.ndarray]],
    tile_idx: np.ndarray,
    seed_idx: np.ndarray,
) -> np.ndarray:
    """Bootstrap hiérarchique apparié.

    ``tile_idx`` : (n_boot, n_tiles) — indices de tuiles partagés
    ``seed_idx`` : (n_boot,) — indices de seed partagés, modulo n_seeds
    """
    n_seeds = len(seed_preds)
    n_boot = tile_idx.shape[0]
    out = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        s = seed_idx[b] % n_seeds
        yt, yp = seed_preds[s]
        idx = tile_idx[b]
        out[b] = _f1_pres(yt[idx], yp[idx])
    return out


def _summary(samples: np.ndarray, observed: float) -> dict:
    return {
        "observed": float(observed),
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
    }


# ================================================================== core logic


def parse_groups(group_args: list[str] | None, probe_best_C: dict[str, float]) -> list[dict]:
    """Parse les arguments --group en une liste de définitions de groupes.

    Format : ``"Display Name:key1,key2,..."``
    Sans --group : chaque clé de probe_best_C devient un groupe de 1 seed.
    """
    if not group_args:
        # Rétrocompatible : une clé = un groupe
        return [{"name": k, "seed_keys": [k]} for k in sorted(probe_best_C.keys())]

    groups = []
    seen_keys: set[str] = set()
    for arg in group_args:
        if ":" not in arg:
            raise SystemExit(f"[ERR] --group mal formé (attendu 'Nom:key1,key2') : {arg!r}")
        name, keys_str = arg.split(":", 1)
        seed_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if not seed_keys:
            raise SystemExit(f"[ERR] --group sans seed keys : {arg!r}")
        for k in seed_keys:
            if k in seen_keys:
                raise SystemExit(f"[ERR] clé seed dupliquée entre groupes : {k!r}")
            seen_keys.add(k)
        groups.append({"name": name, "seed_keys": seed_keys})
    return groups


def run_bootstrap(
    cfg,
    groups: list[dict],
    probe_best_C: dict[str, float],
    default_c: float,
    schema: str,
    n_boot: int,
    seed: int = 42,
    sota_dir: str | None = None,
):
    """Bootstrap APPARIÉ : mêmes indices de tuiles (et de seed pour hiérarchique)
    pour tous les groupes.  Comparaisons plus puissantes que le non-apparié.
    """
    group_samples: dict[str, np.ndarray] = {}
    group_observed: dict[str, float] = {}
    group_stats: dict[str, dict] = {}
    group_n_seeds: dict[str, int] = {}
    group_seed_obs: dict[str, list[float]] = {}

    # --- Étape 1 : fit tous les modèles, collecter les prédictions ---
    all_seeds: list[dict] = []  # [{group_name, seed_key, y_true, y_pred, obs}, ...]
    n_tiles = None
    for g in groups:
        name = g["name"]
        seed_keys = g["seed_keys"]
        group_n_seeds[name] = len(seed_keys)
        obs_per_seed: list[float] = []
        for sk in seed_keys:
            c = _resolve_best_C(sk, probe_best_C, default_c, sota_dir)
            print(f"[fit] {sk}  (C={c})", flush=True)
            y_true, y_pred = _refit_predict(cfg, sk, c, schema, seed=seed)
            obs = _f1_pres(y_true, y_pred)
            obs_per_seed.append(obs)
            print(f"       observed f1_macro_pres = {obs:.4f}  (n_test={len(y_true)})", flush=True)
            if n_tiles is None:
                n_tiles = len(y_true)
            else:
                assert len(y_true) == n_tiles, \
                    f"{sk}: n_test={len(y_true)} ≠ {n_tiles}"
            all_seeds.append({
                "group": name,
                "seed_key": sk,
                "y_true": y_true,
                "y_pred": y_pred,
                "obs": obs,
            })
        group_seed_obs[name] = obs_per_seed

    # --- Étape 2 : pré-générer les indices de bootstrap (partagés) ---
    print(f"\n[bootstrap] {n_boot} tirages appariés sur {n_tiles} tuiles", flush=True)
    tile_idx, seed_idx = _make_bootstrap_indices(n_boot, n_tiles, seed)

    # --- Étape 3 : bootstrap pour chaque seed ---
    # On calcule la distribution par seed, puis on agrège par groupe
    seed_dists: dict[str, np.ndarray] = {}  # seed_key → (n_boot,)
    for entry in all_seeds:
        sk = entry["seed_key"]
        dist = _bootstrap_tiles_paired(entry["y_true"], entry["y_pred"], tile_idx)
        seed_dists[sk] = dist

    # --- Étape 4 : agréger par groupe ---
    for g in groups:
        name = g["name"]
        seed_keys = g["seed_keys"]
        if len(seed_keys) == 1:
            dist = seed_dists[seed_keys[0]]
            obs_mean = group_seed_obs[name][0]
        else:
            # Hiérarchique apparié : à chaque iteration b, on prend la seed
            # dictée par seed_idx[b] % n_seeds, de la distribution de cette seed
            n_seeds = len(seed_keys)
            dist = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                s = seed_idx[b] % n_seeds
                dist[b] = seed_dists[seed_keys[s]][b]
            obs_mean = float(np.mean(group_seed_obs[name]))
            print(f"       inter-seed mean = {obs_mean:.4f}  "
                  f"std = {np.std(group_seed_obs[name], ddof=1):.4f}  "
                  f"seeds = {[round(x, 4) for x in group_seed_obs[name]]}", flush=True)

        group_samples[name] = dist
        group_observed[name] = obs_mean
        group_stats[name] = _summary(dist, obs_mean)

    return group_samples, group_observed, group_stats, group_n_seeds, group_seed_obs


def compute_all_pairs(
    group_names: list[str],
    group_samples: dict[str, np.ndarray],
    group_stats: dict[str, dict],
) -> dict:
    """Toutes paires (A, B) avec A < B (ordre d'apparition) : bootstrap APPARIÉ.

    P(A > B) = fraction des itérations bootstrap où A > B, élément par élément
    (les distributions partagent les mêmes indices, donc l'appariement est direct).
    """
    pairs: dict[str, dict] = {}
    for a, b in itertools.combinations(group_names, 2):
        key = f"{a} : {b}"
        da, db = group_samples[a], group_samples[b]
        a_s, b_s = group_stats[a], group_stats[b]

        # Apparié : même indice → même pseudo-dataset
        p_a_gt_b = float(np.mean(da > db))

        delta = a_s["observed"] - b_s["observed"]
        disjoint = (a_s["ci95_low"] > b_s["ci95_high"]) or \
                   (b_s["ci95_low"] > a_s["ci95_high"])

        # P-value two-sided à partir du bootstrap apparié centré sur H0: Δ = 0.
        # On centre la distribution empirique sur 0 : δ*_centered = (δ_boot - δ_obs)
        # p = fraction où |δ*_centered| >= |δ_obs|
        delta_boot = da - db
        delta_boot_centered = delta_boot - delta  # centre sur 0 (H0)
        p_two_sided = float(np.mean(np.abs(delta_boot_centered) >= np.abs(delta)))

        pairs[key] = {
            "model_a": a,
            "model_b": b,
            "delta_observed_a_minus_b": float(delta),
            "p_a_gt_b": p_a_gt_b,
            "p_two_sided": p_two_sided,
            "ci95_a": [a_s["ci95_low"], a_s["ci95_high"]],
            "ci95_b": [b_s["ci95_low"], b_s["ci95_high"]],
            "ci95_disjoint": disjoint,
            "method": "paired bootstrap (shared tile + seed indices)",
        }
    return pairs


def benjamini_hochberg(pairs: dict, alpha: float = 0.05) -> dict:
    """Correction Benjamini-Hochberg sur les p-values two-sided des paires.

    Retourne un dict {pair_key: {..., "bh_reject": bool, "bh_rank": int, ...}}
    et ajoute les champs dans les paires.
    """
    # Trier les paires par p-value croissante
    sorted_pairs = sorted(pairs.items(), key=lambda kv: kv[1]["p_two_sided"])
    m = len(sorted_pairs)

    for rank, (key, v) in enumerate(sorted_pairs, start=1):
        p = v["p_two_sided"]
        bh_threshold = alpha * rank / m
        v["bh_rank"] = rank
        v["bh_threshold"] = float(bh_threshold)
        v["bh_reject"] = bool(p <= bh_threshold)

    # BH reject doit être monotone : si rang k rejecte, tous les rangs < k aussi
    # Mais BH garantit ça si on prend le plus grand k où p <= alpha*k/m
    # On parcourt du plus petit p au plus grand ; le dernier qui passe fixe le seuil
    max_reject_rank = 0
    for key, v in sorted_pairs:
        if v["bh_reject"]:
            max_reject_rank = max(max_reject_rank, v["bh_rank"])
    for key, v in sorted_pairs:
        v["bh_reject"] = v["bh_rank"] <= max_reject_rank

    return pairs


# ================================================================== heatmap


def _short_name(m: str) -> str:
    """Noms d'affichage compacts (utilisés si le nom brut est trop long)."""
    mapping = {
        "vitb16_imagenet": "ViT-B16\nImageNet",
        "dinov3_vitb16_lvd": "DINOv3\nViT-B16\nLVD",
        "simdinov2_vitb16": "SimDINOv2\nViT-B16",
        "dinov3_vitl16_lvd": "DINOv3\nViT-L16\nLVD",
        "simdinov2_vitl16": "SimDINOv2\nViT-L16",
        "resnet50_imagenet": "ResNet50\nImageNet",
        "dinov3_vitl16_sat": "DINOv3\nViT-L16\nSAT",
    }
    return mapping.get(m, m)


def plot_heatmap(group_names: list[str], pairs: dict, output_png: str,
                 n_boot: int, schema: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(group_names)
    mat = np.full((n, n), np.nan)
    bh_mat = np.zeros((n, n), dtype=bool)  # BH significance
    idx = {m: i for i, m in enumerate(group_names)}

    for _key, v in pairs.items():
        a, b = v["model_a"], v["model_b"]
        i, j = idx[a], idx[b]
        mat[i, j] = v["p_a_gt_b"]
        mat[j, i] = 1.0 - v["p_a_gt_b"]
        bh_mat[i, j] = v["bh_reject"]
        bh_mat[j, i] = v["bh_reject"]

    labels = [_short_name(m) for m in group_names]
    fig, ax = plt.subplots(figsize=(max(7, n * 1.8), max(6, n * 1.6)))

    # RdBu_r : rouge = valeur haute (ligne > colonne), blanc = 0.5, bleu = valeur basse
    im = ax.imshow(mat, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("modèle B (colonne)", fontsize=10)
    ax.set_ylabel("modèle A (ligne)", fontsize=10)
    ax.set_title(f"P(A > B)  |  f1_macro_pres  |  {schema}  |  "
                 f"bootstrap n={n_boot}\n"
                 "† = BH-significatif (α=0.05)", fontsize=10)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="lightgrey", zorder=2))
                continue
            if np.isnan(mat[i, j]):
                continue
            val = mat[i, j]
            txt = f"{val:.3f}"
            if bh_mat[i, j]:
                txt += "†"
            # Texte blanc si la cellule est assez foncée (loin de 0.5)
            color = "white" if abs(val - 0.5) > 0.25 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold" if bh_mat[i, j] else "normal")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("P(ligne > colonne)", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {output_png}")


# ================================================================== main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default="configs/frozen_eval.yaml")
    ap.add_argument("--probe-json", default="results/with_rhol/probe_knn_cgrid.json",
                    help="JSON source pour best_C par seed key")
    ap.add_argument("--group", dest="groups", action="append", default=None,
                    help="Groupe de modèles : 'Nom:key1,key2,...' (répétable). "
                         "Sans --group : chaque clé du probe-json = 1 groupe.")
    ap.add_argument("--schema", choices=["with_rhol", "without_rhol", "8cls", "auto"],
                    default="auto",
                    help="Passe de classes (identique à --pass de probe.py). "
                         "with_rhol=12cls, without_rhol=11cls, auto=schéma natif.")
    ap.add_argument("--default-c", type=float, default=0.01,
                    help="best_C fallback pour les clés absentes du probe-json")
    ap.add_argument("--sota-metrics", action="store_true", default=False,
                    help="Lire best_C depuis sota_screening/*/runs/*/metrics.json "
                         "pour les clés SOTA absentes du probe-json.")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", default="results/significance_matrix.json")
    ap.add_argument("--output-png", default="results/significance_matrix.png")
    args = ap.parse_args()

    # ------------------------------------------------------------------ config
    cfg = load_config(args.config)
    probe_best_C = _load_best_C(args.probe_json)
    print(f"[info] {len(probe_best_C)} best_C chargés depuis {args.probe_json}")

    # ------------------------------------------------------------------ groupes
    groups = parse_groups(args.groups, probe_best_C)
    print(f"[info] {len(groups)} groupes :")
    for g in groups:
        print(f"       {g['name']}  ({len(g['seed_keys'])} seeds: {g['seed_keys']})")

    # ------------------------------------------------------------------ bootstrap
    sota_dir = cfg.paths.sota_dir if args.sota_metrics else None
    group_samples, group_observed, group_stats, group_n_seeds, group_seed_obs = \
        run_bootstrap(cfg, groups, probe_best_C, args.default_c, args.schema,
                      args.n_bootstrap, args.seed, sota_dir)

    if len(groups) < 2:
        print("[STOP] Moins de 2 groupes — aucune paire possible.")
        return

    group_names = [g["name"] for g in groups]
    pairs = compute_all_pairs(group_names, group_samples, group_stats)

    # ------------------------------------------------------------------ Benjamini-Hochberg
    pairs = benjamini_hochberg(pairs, alpha=0.05)
    n_bh_reject = sum(1 for v in pairs.values() if v["bh_reject"])

    # ------------------------------------------------------------------ sortie JSON
    out = {
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "schema": args.schema,
        "metric": "f1_macro_pres",
        "probe_source": args.probe_json,
        "default_c": args.default_c,
        "sota_metrics": args.sota_metrics,
        "bh_alpha": 0.05,
        "bh_n_rejected": n_bh_reject,
        "groups": [
            {
                "name": g["name"],
                "seed_keys": g["seed_keys"],
                "n_seeds": group_n_seeds[g["name"]],
                "observed_f1_per_seed": group_seed_obs[g["name"]],
                "stats": group_stats[g["name"]],
            }
            for g in groups
        ],
        "pairs": pairs,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.output_json}  ({len(groups)} groupes, {len(pairs)} paires)")

    # ------------------------------------------------------------------ résumé
    print(f"\nBenjamini-Hochberg (α=0.05, m={len(pairs)} tests) : {n_bh_reject} rejet(s)")
    if n_bh_reject > 0:
        # Afficher par p-value croissante
        sorted_by_p = sorted(pairs.items(), key=lambda kv: kv[1]["p_two_sided"])
        for key, v in sorted_by_p:
            if v["bh_reject"]:
                print(f"  {key:45s}  p={v['p_two_sided']:.4f}  "
                      f"BH-seuil={v['bh_threshold']:.4f}  ✓")
            else:
                break  # BH garantit la monotonie

    print("\nPaires IC95 disjoints :")
    n_disjoint = 0
    for key, v in pairs.items():
        if v["ci95_disjoint"]:
            n_disjoint += 1
            print(f"  {key}  Δ={v['delta_observed_a_minus_b']:+.4f}  "
                  f"P(A>B)={v['p_a_gt_b']:.3f}  DISJOINTS")
    if n_disjoint == 0:
        print("  (aucune)")

    # ------------------------------------------------------------------ heatmap
    plot_heatmap(group_names, pairs, args.output_png, args.n_bootstrap, args.schema)


if __name__ == "__main__":
    main()
