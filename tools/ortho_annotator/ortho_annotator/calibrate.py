"""Calibration mesurée des deux détecteurs, espèce par espèce.

Aucun seuil n'est choisi à la main : pour chaque espèce, on rejoue le détecteur
sur des fenêtres **non utilisées pour construire les références** et on compare
ses propositions aux points déjà annotés. On garde le seuil qui maximise le F1 et
on conserve le rappel et la précision mesurés.

Deux avertissements à garder en tête en lisant les chiffres produits :

- La « précision » est un **minorant**. Les annotations existantes sont
  incomplètes ; une proposition comptée comme fausse est parfois une vraie plante
  que personne n'avait encore pointée. C'est précisément ce que l'outil cherche.
- Le rappel est mesuré à 35 cm, c'est-à-dire « le détecteur a-t-il pointé à moins
  de 35 cm d'une plante connue », pas « a-t-il détouré la plante ».

Ces mesures servent à deux choses : régler les seuils du balayage, et dire
honnêtement à l'utilisateur pour quelles espèces l'aide fonctionne — le
diagnostic est aussi utile que la détection elle-même.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from rasterio import transform as rio_transform

MATCH_RADIUS_M = 0.35


@dataclass
class SpeciesCalibration:
    threshold: float
    recall: float
    precision: float
    f1: float
    n_windows: int
    n_points: int
    curve: List[Dict] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"threshold": round(self.threshold, 4),
                "recall": round(self.recall, 3),
                "precision": round(self.precision, 3),
                "f1": round(self.f1, 3),
                "n_windows": self.n_windows, "n_points": self.n_points,
                "curve": self.curve}

    @classmethod
    def from_dict(cls, d: Dict) -> "SpeciesCalibration":
        return cls(float(d["threshold"]), float(d["recall"]), float(d["precision"]),
                   float(d["f1"]), int(d["n_windows"]), int(d["n_points"]),
                   list(d.get("curve", [])))


def _match(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    """Rappel et précision par appariement au plus proche, rayon fixe."""
    if gt.shape[0] == 0:
        return float("nan"), float("nan")
    if pred.shape[0] == 0:
        return 0.0, 0.0
    d = np.sqrt(((gt[:, None, :] - pred[None, :, :]) ** 2).sum(-1))
    recall = float((d.min(axis=1) < MATCH_RADIUS_M).mean())
    precision = float((d.min(axis=0) < MATCH_RADIUS_M).mean())
    return recall, precision


def _pick_best(rows: List[Tuple[float, float, float]]) -> Tuple[float, float, float, float]:
    """(seuil, rappel, précision, f1) maximisant le F1."""
    best = (rows[0][0], 0.0, 0.0, -1.0)
    for thr, rec, prec in rows:
        f1 = 0.0 if (rec + prec) <= 0 else 2 * rec * prec / (rec + prec)
        if f1 > best[3]:
            best = (thr, rec, prec, f1)
    return best


# ---------------------------------------------------------------------------
# Détecteur dense (jetons DINOv3)
# ---------------------------------------------------------------------------


def calibrate_dense(
    tilers,
    matcher,
    embedder,
    points_by_species: Dict[str, Tuple[np.ndarray, np.ndarray]],
    thresholds: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12),
    n_windows: int = 6,
    log=print,
) -> Dict[str, SpeciesCalibration]:
    from .dense import _dense_cells

    params = matcher.params
    out: Dict[str, SpeciesCalibration] = {}
    for code in matcher.codes():
        xs, ys = points_by_species.get(code, (np.empty(0), np.empty(0)))
        if xs.size == 0:
            continue
        acc: Dict[float, List[Tuple[float, float]]] = {t: [] for t in thresholds}
        n_win = n_pts = 0
        for tiler in tilers:
            m = tiler.inside_mask(xs, ys)
            if m.sum() < 20:
                continue
            # Fenêtres RÉSERVÉES : celles qui suivent immédiatement celles ayant
            # servi à prélever les jetons de référence.
            cells = _dense_cells(xs[m], ys[m], params.span_m,
                                 params.windows_per_species + n_windows)
            for (wx, wy) in cells[params.windows_per_species:]:
                res = matcher.score_window(tiler, embedder, wx, wy, [code])
                if res is None or code not in res[0]:
                    continue
                smap, left, top = res[0][code], res[1], res[2]
                sel = ((xs >= left) & (xs <= left + params.span_m)
                       & (ys <= top) & (ys >= top - params.span_m))
                gt = np.stack([xs[sel], ys[sel]], axis=1)
                if gt.shape[0] < 3:
                    continue
                n_win += 1
                n_pts += int(gt.shape[0])
                for thr in thresholds:
                    pk = matcher.peaks(smap, left, top, thr, max_peaks=120)
                    pred = np.array([[a, b] for a, b, _ in pk]) if pk else np.empty((0, 2))
                    acc[thr].append(_match(gt, pred))
        if n_win == 0:
            log(f"  {code:8s} : pas assez de fenêtres réservées -> non calibré")
            continue
        rows = [(thr, float(np.mean([r for r, _ in v])), float(np.mean([p for _, p in v])))
                for thr, v in acc.items() if v]
        thr, rec, prec, f1 = _pick_best(rows)
        out[code] = SpeciesCalibration(
            thr, rec, prec, f1, n_win, n_pts,
            [{"threshold": t, "recall": round(r, 3), "precision": round(p, 3)}
             for t, r, p in sorted(rows)])
        log(f"  {code:8s} : seuil={thr:.3f} rappel={rec:.2f} précision≥{prec:.2f} "
            f"(F1={f1:.2f}, {n_win} fenêtres réservées, {n_pts} points)")
    return out


# ---------------------------------------------------------------------------
# Détecteur couleur
# ---------------------------------------------------------------------------


def calibrate_color(
    tilers,
    model,
    points_by_species: Dict[str, Tuple[np.ndarray, np.ndarray]],
    # Balayage large : une espèce abondante (le solidage couvre plus de 1 % du
    # sol par endroits) voit son propre coloris entrer dans les centiles hauts du
    # « fond », ce qui rend un seuil à 99,5 % trop sévère pour elle et trop lâche
    # pour une espèce rare. D'où un choix mesuré, espèce par espèce.
    percentiles: Sequence[float] = (90.0, 95.0, 99.0, 99.5, 99.9),
    span_m: float = 5.0,
    n_windows: int = 6,
    min_diam_m: float = 0.06,
    max_diam_m: float = 0.60,
    log=print,
) -> Dict[str, SpeciesCalibration]:
    from .dense import _dense_cells
    from .prospect import detect_blobs

    out: Dict[str, SpeciesCalibration] = {}
    for code in model.species():
        xs, ys = points_by_species.get(code, (np.empty(0), np.empty(0)))
        if xs.size == 0:
            continue
        acc: Dict[float, List[Tuple[float, float]]] = {p: [] for p in percentiles}
        n_win = n_pts = 0
        for tiler in tilers:
            m = tiler.inside_mask(xs, ys)
            if m.sum() < 20:
                continue
            factor = tiler.best_factor_for(model.gsd_m)
            gsd = tiler.res_x * factor
            size_px = max(16, int(round(span_m / tiler.res_x)))
            side = max(16, size_px // factor)
            for (wx, wy) in _dense_cells(xs[m], ys[m], span_m, n_windows):
                row, col = rio_transform.rowcol(tiler.transform, wx, wy)
                c0, r0 = int(col) - size_px // 2, int(row) - size_px // 2
                if c0 < 0 or r0 < 0 or c0 + size_px > tiler.width \
                        or r0 + size_px > tiler.height:
                    continue
                block = tiler.read_window(c0, r0, size_px, size_px,
                                          out_shape=(side, side))
                if block.size == 0:
                    continue
                left, top = rio_transform.xy(tiler.transform, r0, c0, offset="ul")
                sel = ((xs >= left) & (xs <= left + span_m)
                       & (ys <= top) & (ys >= top - span_m))
                gt = np.stack([xs[sel], ys[sel]], axis=1)
                if gt.shape[0] < 3:
                    continue
                n_win += 1
                n_pts += int(gt.shape[0])
                for pct in percentiles:
                    blobs = detect_blobs(block, model, gsd, [code], {code: pct},
                                         min_diam_m, max_diam_m).get(code, [])
                    if blobs:
                        arr = np.array([[c0 + bc * factor, r0 + br * factor]
                                        for br, bc, _, _ in blobs])
                        X, Y = rio_transform.xy(tiler.transform, arr[:, 1], arr[:, 0],
                                                offset="center")
                        pred = np.stack([np.asarray(X), np.asarray(Y)], axis=1)
                    else:
                        pred = np.empty((0, 2))
                    acc[pct].append(_match(gt, pred))
        if n_win == 0:
            log(f"  {code:8s} : pas assez de fenêtres -> non calibré")
            continue
        rows = [(p, float(np.mean([r for r, _ in v])), float(np.mean([q for _, q in v])))
                for p, v in acc.items() if v]
        thr, rec, prec, f1 = _pick_best(rows)
        out[code] = SpeciesCalibration(
            thr, rec, prec, f1, n_win, n_pts,
            [{"threshold": t, "recall": round(r, 3), "precision": round(p, 3)}
             for t, r, p in sorted(rows)])
        log(f"  {code:8s} : centile={thr} rappel={rec:.2f} précision≥{prec:.2f} "
            f"(F1={f1:.2f}, {n_win} fenêtres, {n_pts} points)")
    return out


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------


def save_calibration(path: Path, color: Dict[str, SpeciesCalibration],
                     dense: Dict[str, SpeciesCalibration]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "match_radius_m": MATCH_RADIUS_M,
        "color": {c: v.as_dict() for c, v in color.items()},
        "dense": {c: v.as_dict() for c, v in dense.items()},
    }, indent=1, ensure_ascii=False), encoding="utf-8")


def load_calibration(path: Path) -> Dict[str, Dict[str, SpeciesCalibration]]:
    p = Path(path)
    if not p.is_file():
        return {"color": {}, "dense": {}}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {
        "color": {c: SpeciesCalibration.from_dict(v)
                  for c, v in raw.get("color", {}).items()},
        "dense": {c: SpeciesCalibration.from_dict(v)
                  for c, v in raw.get("dense", {}).items()},
    }


def recommendation(color: Dict[str, SpeciesCalibration],
                   dense: Dict[str, SpeciesCalibration]) -> Dict[str, str]:
    """Quel détecteur utiliser pour chaque espèce, d'après les mesures."""
    out: Dict[str, str] = {}
    for code in set(color) | set(dense):
        fc = color[code].f1 if code in color else -1.0
        fd = dense[code].f1 if code in dense else -1.0
        if max(fc, fd) < 0.15:
            out[code] = "aucun"          # aucun détecteur n'aide vraiment
        elif fd >= fc:
            out[code] = "dense"
        else:
            out[code] = "couleur"
    return out
