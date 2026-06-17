"""Linear probe (régression logistique) + k-NN sur features cachées.

Protocole linear-eval standard (DINOv2/v3), reproduit EXACTEMENT depuis les notebooks /
``latent_v3_from_embeddings.py`` :
- ``StandardScaler`` fit sur train, appliqué à val/test.
- ``LogisticRegression(solver='lbfgs', max_iter=2000)`` NON pondéré, multinomial
  (``src.utils.make_canonical_lr`` gère l'épinglage compatible sklearn<1.8 et ≥1.8) ;
  grille ``C``, sélection sur **val F1-macro (classes présentes)**.
- k-NN euclidien sur embeddings **L2-normalisés** (k = 5, 10, 20).

Réutilise les MÊMES features cachées que l'analyse latente. scikit-learn paresseux.

Optimisation (passe nocturne 2) : la sélection du C est faite sur un sous-échantillon
train de taille ``cgrid_subsample`` (20 000 par défaut — 1.5× plus rapide que full 49K
sur ResNet-50), puis le fit final utilise TOUTES les données d'entraînement avec le
C sélectionné.  Reproductibilité : seed=42 + ``np.random.RandomState`` séparé du reste.
"""
from __future__ import annotations

import os

import numpy as np

from .latent import l2norm
from .metrics import eval_classifier, per_class_f1
from .utils import make_canonical_lr


def _standardize(feats: dict):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xtr = sc.fit_transform(np.asarray(feats["train"][0], dtype=np.float32))
    xva = sc.transform(np.asarray(feats["val"][0], dtype=np.float32))
    xte = sc.transform(np.asarray(feats["test"][0], dtype=np.float32))
    return xtr, xva, xte


def linear_probe(feats: dict, n_classes: int, class_names: list[str],
                 C_grid=(0.01, 0.1, 1.0, 10.0), max_iter: int = 2000, seed: int = 42,
                 selection_metric: str = "f1_macro_all",
                 cgrid_subsample: int | None = None) -> dict:
    """Régression logistique multi-classes ; le C est sélectionné sur val.

    ``selection_metric`` aligne le critère de sélection sur la métrique reportée :
    - ``f1_macro_all``     : F1-macro sur les ``n_classes`` (labels=range(n)) — DÉFAUT,
      identique à la métrique de benchmark (RHOL=0, zero_division=0).
    - ``f1_macro_present`` : F1-macro sur les seules classes présentes dans val (ancien
      comportement).

    ``cgrid_subsample`` : taille du sous-échantillon train pour la grille C (None =
    train complet, défaut méthodologique). Le fit FINAL utilise toutes les données
    d'entraînement. NOTE : le défaut ``None`` (train complet) est délibéré — un
    sous-échantillon (ex. 20k) change le best_C pour les modèles à fort effectif
    d'entraînement (ex. satmae : C=0.01 → C=0.1) et décale le F1 final de ~0.01.
    """
    from sklearn.metrics import f1_score
    ytr = feats["train"][1]
    yva = feats["val"][1]
    yte = feats["test"][1]
    xtr_full, xva, xte = _standardize(feats)
    if selection_metric == "f1_macro_all":
        sel_labels = list(range(n_classes))
    elif selection_metric == "f1_macro_present":
        sel_labels = sorted({int(v) for v in yva})
    else:
        raise ValueError(f"selection_metric inconnu : '{selection_metric}' "
                         "(f1_macro_all | f1_macro_present).")

    # Sous-échantillon pour la sélection du C (séparé du fit final).
    # Méthodo canonique : train complet (cgrid_subsample=None). Un subsample
    # peut être passé via CLI (ex. pour debug rapide), mais ne change PAS le
    # résultat canonique.
    if cgrid_subsample is not None and cgrid_subsample < xtr_full.shape[0]:
        rng = np.random.RandomState(seed)
        idx_sub = rng.choice(xtr_full.shape[0], cgrid_subsample, replace=False)
        xtr_sub, ytr_sub = xtr_full[idx_sub], ytr[idx_sub]
    else:
        xtr_sub, ytr_sub = xtr_full, ytr

    def _fit_one(c):
        clf = make_canonical_lr(C=c, max_iter=max_iter, random_state=seed)
        clf.fit(xtr_sub, ytr_sub)
        f1v = f1_score(yva, clf.predict(xva), average="macro", zero_division=0, labels=sel_labels)
        return c, f1v, clf

    # Grille de C petite (≤6 valeurs) + modèles 768-2048 dim × 49 433 train
    # → joblib n'apporte rien (overhead > gain). Fit séquentiel, déterministe
    # (les workers joblib peuvent réordonner les résultats de façon subtile).
    # On garde la possibilité de paralleliser via env PROBE_PARALLEL=1 si besoin.
    if os.environ.get("PROBE_PARALLEL", "0") == "1":
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(delayed(_fit_one)(c) for c in C_grid)
    else:
        results = [_fit_one(c) for c in C_grid]
    best_c, best_f1, _ = max(results, key=lambda x: x[1])

    # Fit final sur TOUT le train avec best_c (le fit qui sera évalué sur test).
    best = make_canonical_lr(C=best_c, max_iter=max_iter, random_state=seed)
    best.fit(xtr_full, ytr)
    ypte = best.predict(xte)
    return {
        "best_C": best_c,
        "selection_metric": selection_metric,
        "val": eval_classifier(yva, best.predict(xva), n_classes),
        "test": eval_classifier(yte, ypte, n_classes),
        "f1_per_class_test": per_class_f1(yte, ypte, class_names),
    }


def knn(feats: dict, n_classes: int, ks=(5, 10, 20)) -> dict:
    """k-NN euclidien sur embeddings L2-normalisés, pour chaque k."""
    from sklearn.neighbors import KNeighborsClassifier
    etr, ytr = feats["train"]
    eva, yva = feats["val"]
    ete, yte = feats["test"]
    etr, eva, ete = l2norm(etr), l2norm(eva), l2norm(ete)
    res = {}
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean", n_jobs=-1)
        clf.fit(etr, ytr)
        res[f"k={k}"] = {
            "val": eval_classifier(yva, clf.predict(eva), n_classes),
            "test": eval_classifier(yte, clf.predict(ete), n_classes),
        }
    return res
