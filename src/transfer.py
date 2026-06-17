"""Wrapper LogME pour le benchmark Arctic-TVC.

On VENDORISE :class:`LogME` depuis le dépôt officiel thuml/LogME (MIT) — cf.
``src/_vendor/LogME.py`` (commentaire en tête du fichier pour la source).
PIÈGE ÉVITÉ : le package PyPI ``logme`` est une lib de logging sans rapport ;
ne PAS faire ``pip install logme``.

API du wrapper (tient en 1 ligne) :
    >>> from src.transfer import logme_score
    >>> s = logme_score(F_train, y_train_int, regression=False)

Formule (You et al., ICML 2021 / JMLR 2022 — fixed-point) :
    pour chaque classe c, on optimise l'évidence marginale bayésienne d'un
    modèle linéaire f(X) ≈ y_c via un point fixe sur (α, β) ; le score
    retourné est la moyenne sur les classes de ``evidence / N``.  Plus le
    score est GRAND, plus le modèle est transférable (à interpréter comme
    un RANG, pas une valeur absolue).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ._vendor.LogME import LogME


def logme_score(F: np.ndarray, y: Sequence[int], regression: bool = False) -> float:
    """Renvoie le score LogME (un scalaire ; plus grand = plus transférable).

    Paramètres
    ----------
    F : ndarray [N, D]
        Embeddings (pooler output), DOIT être convertible en float64 (le
        wrapper le fait pour vous).  N = nb d'exemples train, D = dimension
        de la représentation.
    y : séquence d'entiers, longueur N
        Labels d'entraînement, dans ``[0, C)`` où C = ``max(y) + 1``.
    regression : bool
        ``False`` (défaut) pour la classification multi-classes.

    Renvoie
    -------
    score : float
        ``np.mean(evidences_per_class)`` (evidence = log marginal likelihood
        par classe, divisée par N).  Valeur absolue NON interprétable —
        c'est un outil de RANG.
    """
    F = np.asarray(F, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    logme = LogME(regression=regression)
    return float(logme.fit(F, y))
