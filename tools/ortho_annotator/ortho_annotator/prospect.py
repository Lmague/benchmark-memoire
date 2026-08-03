"""Prospection : proposer où regarder, au lieu de balayer 200 000 tuiles à la main.

Chaîne en deux étages, choisie parce que l'un est bon marché et l'autre précis :

1. **Étage couleur** (``colormodel``) — balaye l'orthomosaïque entière à ~1 cm/px
   via les overviews internes, en blocs, mémoire bornée. Produit des taches
   candidates à l'échelle d'une inflorescence. Rappel élevé, précision moyenne.
2. **Étage plus-proche-prototype** (``embed``) — ne s'applique qu'aux meilleures
   taches, en pleine résolution, avec un encodeur figé. Précision bien meilleure,
   mais ~19 vignettes/s sur CPU : impossible à passer sur toute l'image, parfait
   en re-classement.

Les candidats déjà couverts par une annotation existante sont éliminés : l'outil
ne propose que du **nouveau**.

Tout est écrit à côté du GeoPackage de sortie, jamais dans ``Dataset_Leo/``.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rasterio import transform as rio_transform

from .colormodel import PAD_THRESHOLD, ColorModel

STATUS_NEW = "new"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"


@dataclass
class Candidate:
    species: str
    x: float
    y: float
    color_score: float
    area_m2: float
    embed_score: Optional[float] = None
    best_code: Optional[str] = None
    score: float = 0.0
    cid: int = -1
    status: str = STATUS_NEW

    def as_dict(self) -> Dict:
        return {
            "id": self.cid, "species": self.species, "x": self.x, "y": self.y,
            "color_score": round(self.color_score, 3),
            "area_m2": round(self.area_m2, 5),
            "embed_score": None if self.embed_score is None else round(self.embed_score, 4),
            "best_code": self.best_code,
            "score": round(self.score, 4), "status": self.status,
        }


# ---------------------------------------------------------------------------
# Détection de taches sur un bloc décimé
# ---------------------------------------------------------------------------


def detect_blobs(
    block: np.ndarray,
    model: ColorModel,
    gsd_m: float,
    species: Sequence[str],
    thresholds: Dict[str, float],
    min_diam_m: float,
    max_diam_m: float,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """Taches par espèce sur un bloc RGB décimé.

    ``thresholds`` donne, par espèce, le **centile de fond** à dépasser : il est
    mesuré par ``calibrate.calibrate_color`` et diffère fortement d'une espèce à
    l'autre. Renvoie ``code -> [(row, col, aire_m2, score_moyen), ...]`` en
    coordonnées pixel du bloc. Le lissage se fait à l'échelle d'une inflorescence : un pixel
    isolé de la bonne couleur n'est pas une plante, une tache de la bonne taille
    l'est probablement.
    """
    from scipy import ndimage

    out: Dict[str, List[Tuple[float, float, float, float]]] = {}
    valid = ~np.all(block[..., :3] >= PAD_THRESHOLD, axis=2)
    if not valid.any():
        return out

    smooth_px = max(2, int(round(min_diam_m / gsd_m)))
    px_area = gsd_m * gsd_m
    min_area = max(2.0, (math.pi / 4.0) * (min_diam_m / gsd_m) ** 2 * 0.35)
    max_area = (math.pi / 4.0) * (max_diam_m / gsd_m) ** 2 * 2.0

    scores = model.score_all(block)
    for code in species:
        s = scores.get(code)
        if s is None:
            continue
        s = ndimage.uniform_filter(s, size=smooth_px, mode="nearest")
        pct = thresholds.get(code, 99.5)
        mask = (s > model.threshold(code, pct)) & valid
        if not mask.any():
            out[code] = []
            continue
        lbl, n = ndimage.label(mask)
        if n == 0:
            out[code] = []
            continue
        idx = np.arange(1, n + 1)
        areas = np.asarray(ndimage.sum_labels(mask, lbl, idx), dtype="float64")
        keep = (areas >= min_area) & (areas <= max_area)
        if not keep.any():
            out[code] = []
            continue
        kept = idx[keep]
        coms = ndimage.center_of_mass(mask, lbl, kept)
        means = np.asarray(ndimage.mean(s, lbl, kept), dtype="float64")
        out[code] = [
            (float(r), float(c), float(a * px_area), float(m))
            for (r, c), a, m in zip(coms, areas[keep], means)
        ]
    return out


def _iter_blocks_halo(tiler, factor: int, block_px: int, halo_px: int):
    """Blocs décimés avec recouvrement, pour ne pas couper les taches en deux."""
    step_full = block_px * factor
    halo_full = halo_px * factor
    for row_off in range(0, tiler.height, step_full):
        for col_off in range(0, tiler.width, step_full):
            r0 = max(0, row_off - halo_full)
            c0 = max(0, col_off - halo_full)
            r1 = min(tiler.height, row_off + step_full + halo_full)
            c1 = min(tiler.width, col_off + step_full + halo_full)
            oh = max(1, (r1 - r0) // factor)
            ow = max(1, (c1 - c0) // factor)
            block = tiler.read_window(c0, r0, c1 - c0, r1 - r0, out_shape=(oh, ow))
            if block.size == 0:
                continue
            # Fenêtre « utile » en coordonnées du bloc décimé.
            core = ((row_off - r0) / factor, (col_off - c0) / factor,
                    (min(tiler.height, row_off + step_full) - r0) / factor,
                    (min(tiler.width, col_off + step_full) - c0) / factor)
            yield c0, r0, block, core


# ---------------------------------------------------------------------------
# Suppression des doublons et des points déjà annotés
# ---------------------------------------------------------------------------


class _Grid:
    """Grille de hachage spatiale : voisinage en O(1), sans dépendance externe."""

    def __init__(self, cell_m: float):
        self.cell = max(1e-6, float(cell_m))
        self._d: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(math.floor(x / self.cell)), int(math.floor(y / self.cell)))

    def add(self, x: float, y: float) -> None:
        self._d.setdefault(self._key(x, y), []).append((x, y))

    def add_many(self, xs: np.ndarray, ys: np.ndarray) -> None:
        for x, y in zip(xs, ys):
            self.add(float(x), float(y))

    def has_within(self, x: float, y: float, radius_m: float) -> bool:
        kx, ky = self._key(x, y)
        r2 = radius_m * radius_m
        span = int(math.ceil(radius_m / self.cell))
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for (px, py) in self._d.get((kx + dx, ky + dy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 <= r2:
                        return True
        return False


def deduplicate(cands: List[Candidate], min_sep_m: float) -> List[Candidate]:
    """Suppression non-maximale : un seul candidat par plante, le mieux noté."""
    by_species: Dict[str, List[Candidate]] = {}
    for c in cands:
        by_species.setdefault(c.species, []).append(c)
    out: List[Candidate] = []
    for code, group in by_species.items():
        group.sort(key=lambda c: c.score, reverse=True)
        grid = _Grid(min_sep_m)
        for c in group:
            if grid.has_within(c.x, c.y, min_sep_m):
                continue
            grid.add(c.x, c.y)
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Balayage
# ---------------------------------------------------------------------------


@dataclass
class ScanParams:
    threshold_pct: float = 99.5
    min_diam_m: float = 0.06
    max_diam_m: float = 0.60
    min_sep_m: float = 0.30
    exclude_m: float = 0.40
    max_per_species: int = 1500
    block_px: int = 1024
    rerank_top: int = 400
    crop_m: float = 0.30

    def as_dict(self) -> Dict:
        return dict(self.__dict__)


def scan_raster(
    tiler,
    model: ColorModel,
    known_xy: Tuple[np.ndarray, np.ndarray],
    params: ScanParams,
    species: Optional[Sequence[str]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    bank=None,
    embedder=None,
    log=print,
    progress=None,
) -> List[Candidate]:
    """Balaye une orthomosaïque entière et renvoie les candidats retenus."""
    codes = list(species) if species else model.species()
    codes = [c for c in codes if c in model.llr]
    if not codes:
        return []
    thr = {c: (thresholds or {}).get(c, params.threshold_pct) for c in codes}

    factor = tiler.best_factor_for(model.gsd_m)
    gsd = tiler.res_x * factor
    halo = max(8, int(round(params.max_diam_m / gsd)))
    n_blocks = (math.ceil(tiler.height / (params.block_px * factor))
                * math.ceil(tiler.width / (params.block_px * factor)))
    log(f"  balayage {tiler.path.name} : facteur {factor} "
        f"(GSD {gsd*100:.2f} cm/px), {n_blocks} blocs, espèces {', '.join(codes)}")

    found: List[Candidate] = []
    done = 0
    for c0, r0, block, core in _iter_blocks_halo(tiler, factor, params.block_px, halo):
        done += 1
        if progress:
            progress(done, n_blocks)
        blobs = detect_blobs(block, model, gsd, codes, thr,
                             params.min_diam_m, params.max_diam_m)
        cr0, cc0, cr1, cc1 = core
        for code, items in blobs.items():
            for (br, bc, area, mean) in items:
                # Le halo sert à mesurer correctement les taches de bord, pas à
                # les compter deux fois : seul le coeur du bloc produit.
                if not (cr0 <= br < cr1 and cc0 <= bc < cc1):
                    continue
                full_row = r0 + br * factor
                full_col = c0 + bc * factor
                x, y = rio_transform.xy(tiler.transform, full_row, full_col,
                                        offset="center")
                found.append(Candidate(
                    species=code, x=float(x), y=float(y),
                    color_score=float(mean * math.log1p(area / (gsd * gsd))),
                    area_m2=float(area),
                ))
        if done % 25 == 0 or done == n_blocks:
            log(f"    bloc {done}/{n_blocks} — {len(found)} taches brutes")

    for c in found:
        c.score = c.color_score
    log(f"  {len(found)} taches brutes -> déduplication…")
    found = deduplicate(found, params.min_sep_m)

    # On ne propose que du nouveau : tout ce qui est déjà annoté est retiré.
    kx, ky = known_xy
    if kx.size:
        grid = _Grid(max(params.exclude_m, 0.5))
        grid.add_many(kx, ky)
        before = len(found)
        found = [c for c in found if not grid.has_within(c.x, c.y, params.exclude_m)]
        log(f"  {before - len(found)} candidat(s) déjà annoté(s) écarté(s)")

    # Plafond par espèce.
    per: Dict[str, List[Candidate]] = {}
    for c in found:
        per.setdefault(c.species, []).append(c)
    found = []
    for code, group in per.items():
        group.sort(key=lambda c: c.score, reverse=True)
        found.extend(group[: params.max_per_species])
        log(f"  {code:8s} : {len(group[: params.max_per_species])} candidat(s) retenus")

    if bank is not None and embedder is not None and params.rerank_top > 0:
        found = rerank(tiler, found, bank, embedder, params, log=log)
    return found


def rerank(tiler, cands: List[Candidate], bank, embedder, params: ScanParams,
           log=print) -> List[Candidate]:
    """Re-classe les meilleurs candidats couleur par plus proche prototype."""
    per: Dict[str, List[Candidate]] = {}
    for c in cands:
        per.setdefault(c.species, []).append(c)
    todo: List[Candidate] = []
    for code, group in per.items():
        group.sort(key=lambda c: c.color_score, reverse=True)
        todo.extend(group[: params.rerank_top])
    if not todo:
        return cands

    crop_px = max(16, int(round(params.crop_m / tiler.res_x)))
    log(f"  re-classement DINOv3 de {len(todo)} candidat(s) "
        f"(vignettes {params.crop_m*100:.0f} cm)…")
    batch: List[np.ndarray] = []
    keep: List[Candidate] = []
    for c in todo:
        arr = tiler.read_centered(c.x, c.y, crop_px, out_size=224)
        if arr.size:
            batch.append(arr)
            keep.append(c)
    step = 64
    for i in range(0, len(batch), step):
        emb = embedder.embed(batch[i: i + step])
        scores = bank.score(emb)
        best_codes, _ = bank.best(emb)
        for j, c in enumerate(keep[i: i + step]):
            c.embed_score = float(scores.get(c.species, np.zeros(1))[j]) \
                if c.species in scores else None
            c.best_code = best_codes[j] if best_codes else None
        log(f"    {min(i + step, len(batch))}/{len(batch)} vignettes encodées")

    # Score final = similarité discriminante. Un candidat que l'encodeur rattache
    # à une AUTRE espèce est rétrogradé, pas supprimé : c'est souvent une vraie
    # plante, simplement mal étiquetée par la couleur.
    kept_out: List[Candidate] = []
    for c in cands:
        if c.embed_score is None:
            continue        # non re-classé : son score couleur n'est pas comparable
        penalty = 0.0 if (c.best_code in (None, c.species)) else 0.15
        c.score = c.embed_score - penalty
        kept_out.append(c)
    dropped = len(cands) - len(kept_out)
    if dropped:
        log(f"  {dropped} candidat(s) couleur au-delà de --rerank-top écartés "
            f"(score non comparable à celui des candidats vérifiés)")
    return kept_out


def scan_raster_dense(
    tiler,
    matcher,
    embedder,
    known_xy: Tuple[np.ndarray, np.ndarray],
    params: ScanParams,
    thresholds: Optional[Dict[str, float]] = None,
    species: Optional[Sequence[str]] = None,
    max_windows: int = 0,
    log=print,
    progress=None,
) -> List[Candidate]:
    """Balayage dense par appariement de jetons (voir ``dense.py``).

    Plus lent que l'étage couleur (~1 s par fenêtre de 5 m) mais il voit la
    texture et la forme : c'est le seul recours pour les espèces défleuries.
    Les fenêtres entièrement blanches (remplissage de bord) sont sautées.
    """
    codes = [c for c in (species or matcher.codes()) if c in matcher.species]
    if not codes:
        return []
    thr = {c: (thresholds or {}).get(c, 0.03) for c in codes}
    span_px = max(16, int(round(matcher.params.span_m / tiler.res_x)))
    cols = math.ceil(tiler.width / span_px)
    rows = math.ceil(tiler.height / span_px)
    todo = [(c, r) for r in range(rows) for c in range(cols)
            if not tiler.region_empty(c * span_px, r * span_px, span_px, span_px)]
    if max_windows:
        todo = todo[:max_windows]
    log(f"  balayage dense {tiler.path.name} : {len(todo)} fenêtre(s) de "
        f"{matcher.params.span_m} m sur {cols * rows} (bordures sautées), "
        f"espèces {', '.join(codes)}")

    found: List[Candidate] = []
    for n, (c, r) in enumerate(todo, start=1):
        if progress:
            progress(n, len(todo))
        cx, cy = rio_transform.xy(
            tiler.transform, r * span_px + span_px / 2, c * span_px + span_px / 2)
        res = matcher.score_window(tiler, embedder, float(cx), float(cy), codes)
        if res is None:
            continue
        scores, left, top = res
        for code, smap in scores.items():
            for (x, y, s) in matcher.peaks(smap, left, top, thr[code], max_peaks=25):
                found.append(Candidate(species=code, x=x, y=y, color_score=0.0,
                                       area_m2=matcher.params.token_m() ** 2,
                                       embed_score=s, best_code=code, score=s))
        if n % 25 == 0 or n == len(todo):
            log(f"    fenêtre {n}/{len(todo)} — {len(found)} pic(s)")

    found = deduplicate(found, params.min_sep_m)
    kx, ky = known_xy
    if kx.size:
        grid = _Grid(max(params.exclude_m, 0.5))
        grid.add_many(kx, ky)
        before = len(found)
        found = [c for c in found if not grid.has_within(c.x, c.y, params.exclude_m)]
        log(f"  {before - len(found)} candidat(s) déjà annoté(s) écarté(s)")

    per: Dict[str, List[Candidate]] = {}
    for c in found:
        per.setdefault(c.species, []).append(c)
    out: List[Candidate] = []
    for code, group in per.items():
        group.sort(key=lambda c: c.score, reverse=True)
        out.extend(group[: params.max_per_species])
        log(f"  {code:8s} : {len(group[: params.max_per_species])} candidat(s) retenus")
    return out


# ---------------------------------------------------------------------------
# Stockage des candidats (SQLite, à côté du GeoPackage de sortie)
# ---------------------------------------------------------------------------


class CandidateStore:
    """Base SQLite des candidats, indexée pour les requêtes par fenêtre."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(self.path), check_same_thread=False)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._con.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    raster TEXT NOT NULL,
                    species TEXT NOT NULL,
                    x REAL NOT NULL, y REAL NOT NULL,
                    color_score REAL, area_m2 REAL,
                    embed_score REAL, best_code TEXT,
                    score REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new'
                );
                CREATE INDEX IF NOT EXISTS idx_cand_rs
                    ON candidates(raster, status, score DESC);
                CREATE INDEX IF NOT EXISTS idx_cand_xy ON candidates(raster, x, y);
                CREATE TABLE IF NOT EXISTS scans (
                    raster TEXT PRIMARY KEY, created TEXT, params TEXT, n INTEGER
                );
                """
            )
            self._con.commit()

    # ---- écriture --------------------------------------------------------------

    def replace_raster(self, raster: str, cands: Iterable[Candidate],
                       params: Optional[Dict] = None) -> int:
        """Remplace les candidats d'une orthomosaïque, en gardant les décisions.

        Les candidats déjà acceptés ou rejetés par l'utilisateur ne sont pas
        effacés : un nouveau balayage ne doit jamais faire réapparaître un
        candidat écarté à la main.
        """
        with self._lock:
            decided = self._con.execute(
                "SELECT x, y, status FROM candidates WHERE raster = ? AND status != ?",
                (raster, STATUS_NEW),
            ).fetchall()
            self._con.execute("DELETE FROM candidates WHERE raster = ? AND status = ?",
                              (raster, STATUS_NEW))
            grid = _Grid(0.5)
            for x, y, _st in decided:
                grid.add(float(x), float(y))
            rows = []
            for c in cands:
                if grid.has_within(c.x, c.y, 0.25):
                    continue
                rows.append((raster, c.species, c.x, c.y, c.color_score, c.area_m2,
                             c.embed_score, c.best_code, c.score, STATUS_NEW))
            self._con.executemany(
                "INSERT INTO candidates (raster, species, x, y, color_score, area_m2,"
                " embed_score, best_code, score, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._con.execute(
                "INSERT INTO scans (raster, created, params, n) VALUES (?,?,?,?)"
                " ON CONFLICT(raster) DO UPDATE SET created=excluded.created,"
                " params=excluded.params, n=excluded.n",
                (raster, datetime.now().isoformat(timespec="seconds"),
                 json.dumps(params or {}), len(rows)),
            )
            self._con.commit()
        return len(rows)

    def add(self, raster: str, cands: Iterable[Candidate],
            min_sep_m: float = 0.25) -> int:
        """Ajoute des candidats (analyse à la demande) sans toucher aux autres.

        Un candidat déjà présent — ou déjà accepté/rejeté — au même endroit n'est
        pas réinséré : relancer l'analyse d'une tuile ne doit pas empiler les
        doublons ni faire réapparaître ce que l'utilisateur a écarté.
        """
        cands = list(cands)
        if not cands:
            return 0
        xs = [c.x for c in cands]
        ys = [c.y for c in cands]
        pad = min_sep_m + 1.0
        with self._lock:
            rows = self._con.execute(
                "SELECT x, y, species FROM candidates WHERE raster = ?"
                " AND x >= ? AND x <= ? AND y >= ? AND y <= ?",
                (raster, min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad),
            ).fetchall()
            grids: Dict[str, _Grid] = {}
            for x, y, sp in rows:
                grids.setdefault(sp, _Grid(min_sep_m)).add(float(x), float(y))
            new = []
            for c in cands:
                g = grids.setdefault(c.species, _Grid(min_sep_m))
                if g.has_within(c.x, c.y, min_sep_m):
                    continue
                g.add(c.x, c.y)
                new.append((raster, c.species, c.x, c.y, c.color_score, c.area_m2,
                            c.embed_score, c.best_code, c.score, STATUS_NEW))
            self._con.executemany(
                "INSERT INTO candidates (raster, species, x, y, color_score, area_m2,"
                " embed_score, best_code, score, status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                new,
            )
            self._con.commit()
        return len(new)

    def set_status(self, cid: int, status: str) -> bool:
        with self._lock:
            cur = self._con.execute("UPDATE candidates SET status = ? WHERE id = ?",
                                    (status, int(cid)))
            self._con.commit()
        return cur.rowcount > 0

    def reset_status(self, raster: str) -> int:
        with self._lock:
            cur = self._con.execute(
                "UPDATE candidates SET status = ? WHERE raster = ?", (STATUS_NEW, raster))
            self._con.commit()
        return cur.rowcount

    # ---- lecture ---------------------------------------------------------------

    @staticmethod
    def _row(r) -> Candidate:
        return Candidate(species=r[1], x=r[2], y=r[3], color_score=r[4] or 0.0,
                         area_m2=r[5] or 0.0, embed_score=r[6], best_code=r[7],
                         score=r[8], cid=int(r[0]), status=r[9])

    _COLS = ("id, species, x, y, color_score, area_m2, embed_score, best_code, "
             "score, status")

    def in_bbox(self, raster: str, left: float, bottom: float, right: float,
                top: float, include_decided: bool = True) -> List[Candidate]:
        q = (f"SELECT {self._COLS} FROM candidates WHERE raster = ?"
             " AND x >= ? AND x <= ? AND y >= ? AND y <= ?")
        args = [raster, left, right, bottom, top]
        if not include_decided:
            q += " AND status = ?"
            args.append(STATUS_NEW)
        with self._lock:
            rows = self._con.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def ranked(self, raster: str, species: Optional[str] = None,
               limit: int = 500) -> List[Candidate]:
        q = (f"SELECT {self._COLS} FROM candidates WHERE raster = ? AND status = ?")
        args: List = [raster, STATUS_NEW]
        if species:
            q += " AND species = ?"
            args.append(species)
        q += " ORDER BY score DESC LIMIT ?"
        args.append(int(limit))
        with self._lock:
            rows = self._con.execute(q, args).fetchall()
        return [self._row(r) for r in rows]

    def get(self, cid: int) -> Optional[Candidate]:
        with self._lock:
            r = self._con.execute(f"SELECT {self._COLS} FROM candidates WHERE id = ?",
                                  (int(cid),)).fetchone()
        return self._row(r) if r else None

    def counts(self, raster: str) -> Dict[str, Dict[str, int]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT species, status, COUNT(*) FROM candidates WHERE raster = ?"
                " GROUP BY species, status", (raster,)).fetchall()
        out: Dict[str, Dict[str, int]] = {}
        for sp, st, n in rows:
            out.setdefault(sp, {})[st] = int(n)
        return out

    def scanned_rasters(self) -> Dict[str, Dict]:
        with self._lock:
            rows = self._con.execute("SELECT raster, created, n FROM scans").fetchall()
        return {r[0]: {"created": r[1], "n": int(r[2])} for r in rows}

    def all_new_xy(self, raster: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Tous les candidats non décidés (pour la carte de chaleur de la minicarte)."""
        with self._lock:
            rows = self._con.execute(
                "SELECT x, y, species FROM candidates WHERE raster = ? AND status = ?",
                (raster, STATUS_NEW)).fetchall()
        if not rows:
            return np.empty(0), np.empty(0), []
        xs = np.array([r[0] for r in rows], dtype="float64")
        ys = np.array([r[1] for r in rows], dtype="float64")
        return xs, ys, [r[2] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._con.commit()
            self._con.close()
