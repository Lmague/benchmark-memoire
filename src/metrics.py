"""Métriques de classification, conventions EXACTES des notebooks.

- ``f1_macro_all``  : F1-macro sur les ``n_classes`` (labels=range(n)), zero_division=0
                      → RHOL absente compte comme 0.
- ``f1_macro_pres`` : F1-macro sur les classes présentes (labels=sorted(set(y_true)))
                      → gère RHOL/ARCA/DRYI/RUBC inévaluables.
- ``f1_weighted``, ``accuracy``.

scikit-learn est importé paresseusement.
"""
from __future__ import annotations

import numpy as np

from .utils import CLASS_NAMES, N_CLASSES


def eval_classifier(y_true, y_pred, n_classes: int = N_CLASSES) -> dict:
    """Dict des 4 métriques scalaires (all / present / weighted / accuracy)."""
    from sklearn.metrics import accuracy_score, f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    present = sorted({int(v) for v in y_true})
    return {
        "f1_macro_all": float(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0, labels=list(range(n_classes)))),
        "f1_macro_pres": float(f1_score(y_true, y_pred, average="macro",
                                        zero_division=0, labels=present)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0, labels=list(range(n_classes)))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_f1(y_true, y_pred, class_names: list[str] = CLASS_NAMES) -> dict:
    """F1 par classe (zero_division=0), indexé par nom de classe."""
    from sklearn.metrics import f1_score
    n = len(class_names)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(n)))
    return {c: float(v) for c, v in zip(class_names, f1)}


def per_class_count(y_true, class_names: list[str] = CLASS_NAMES) -> dict:
    """Nombre d'exemples par classe dans ``y_true``."""
    y_true = np.asarray(y_true)
    return {c: int((y_true == i).sum()) for i, c in enumerate(class_names)}


def confusion(y_true, y_pred, n_classes: int = N_CLASSES):
    """Matrice de confusion (numpy array n×n)."""
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
