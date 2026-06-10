"""Linear probe (régression logistique) + k-NN sur features cachées.

Protocole linear-eval standard (DINOv2/v3), reproduit EXACTEMENT depuis les notebooks /
``latent_v3_from_embeddings.py`` :
- ``StandardScaler`` fit sur train, appliqué à val/test.
- ``LogisticRegression(solver='lbfgs', max_iter=2000)`` NON pondéré ; grille ``C``,
  sélection sur **val F1-macro (classes présentes)**.
- k-NN euclidien sur embeddings **L2-normalisés** (k = 5, 10, 20).

Réutilise les MÊMES features cachées que l'analyse latente. scikit-learn paresseux.
"""
from __future__ import annotations

import numpy as np

from .latent import l2norm
from .metrics import eval_classifier, per_class_f1


def _standardize(feats: dict):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xtr = sc.fit_transform(np.asarray(feats["train"][0], dtype=np.float32))
    xva = sc.transform(np.asarray(feats["val"][0], dtype=np.float32))
    xte = sc.transform(np.asarray(feats["test"][0], dtype=np.float32))
    return xtr, xva, xte


def linear_probe(feats: dict, n_classes: int, class_names: list[str],
                 C_grid=(0.01, 0.1, 1.0, 10.0), max_iter: int = 2000, seed: int = 42,
                 selection_metric: str = "f1_macro_all") -> dict:
    """Régression logistique multi-classes ; le C est sélectionné sur val.

    ``selection_metric`` aligne le critère de sélection sur la métrique reportée :
    - ``f1_macro_all``     : F1-macro sur les ``n_classes`` (labels=range(n)) — DÉFAUT,
      identique à la métrique de benchmark (RHOL=0, zero_division=0).
    - ``f1_macro_present`` : F1-macro sur les seules classes présentes dans val (ancien
      comportement).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    ytr = feats["train"][1]
    yva = feats["val"][1]
    yte = feats["test"][1]
    xtr, xva, xte = _standardize(feats)
    if selection_metric == "f1_macro_all":
        sel_labels = list(range(n_classes))
    elif selection_metric == "f1_macro_present":
        sel_labels = sorted({int(v) for v in yva})
    else:
        raise ValueError(f"selection_metric inconnu : '{selection_metric}' "
                         "(f1_macro_all | f1_macro_present).")

    best_c, best_f1, best = None, -1.0, None
    for c in C_grid:
        clf = LogisticRegression(C=c, max_iter=max_iter, solver="liblinear", random_state=seed)
        clf.fit(xtr, ytr)
        f1v = f1_score(yva, clf.predict(xva), average="macro", zero_division=0, labels=sel_labels)
        if f1v > best_f1:
            best_c, best_f1, best = c, f1v, clf

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
