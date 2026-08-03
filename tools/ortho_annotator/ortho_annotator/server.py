"""Application FastAPI : sert le frontend statique et l'API d'annotation.

Lectures raster fenêtrées (déléguées à ``RasterTiler``) ; écritures dans l'unique
GeoPackage de sortie (``AnnotationStore``). L'orthomosaïque courante est
commutable en cours de session, sans changer de fichier de sortie.

S'y ajoute la **prospection** : les candidats trouvés hors ligne par
``prospect scan`` sont affichés sur la tuile et acceptables d'un clic, et une
analyse à la demande peut être lancée sur la tuile courante (~1 s).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import (CANDIDATES_FILE, COLOR_MODEL_FILE, DENSE_FILE,
                     PROTOTYPES_FILE, AppConfig)
from .geopackage import AnnotationStore
from .overlay import ExistingAnnotations
from .raster import RasterTiler, list_rasters
from .references import ReferenceLibrary

_STATIC_DIR = Path(__file__).resolve().parent / "static"

_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00b8b8",
    "#bcf60c", "#fabebe", "#008080", "#9a6324", "#800000", "#808000",
]


class AnnotateBody(BaseModel):
    index: int
    px: float
    py: float
    species: str


class DeleteBody(BaseModel):
    species: str
    fid: int


class ReviewBody(BaseModel):
    index: int
    px0: float
    py0: float
    px1: float
    py1: float
    note: str = ""


class ReviewDeleteBody(BaseModel):
    fid: int


class SwitchBody(BaseModel):
    name: str


class CandidateActionBody(BaseModel):
    id: int
    species: Optional[str] = None


class AnalyzeBody(BaseModel):
    index: int
    mode: str = "auto"          # "auto" | "color" | "dense"
    species: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Contexte de prospection
# ---------------------------------------------------------------------------


class ProspectContext:
    """Modèles et candidats de prospection, chargés à la demande.

    Rien ici n'est obligatoire : si aucun artefact n'existe, l'annotateur
    fonctionne exactement comme avant. L'encodeur (torch) n'est chargé qu'au
    premier appel réel, pour ne pas ralentir le démarrage de deux secondes.
    """

    def __init__(self, folder: Path, model_id: Optional[str] = None, threads: int = 4):
        self.folder = Path(folder)
        self.model_id = model_id
        self.threads = int(threads)
        self.color_model = None
        self.matcher = None
        self.bank = None
        self.candidates = None
        self.calibration = {"color": {}, "dense": {}}
        self._embedder = None

    @classmethod
    def load(cls, folder: Path, model_id: Optional[str] = None,
             threads: int = 4) -> "ProspectContext":
        from .calibrate import load_calibration
        from .colormodel import ColorModel
        from .prospect import CandidateStore

        ctx = cls(folder, model_id, threads)
        if (folder / COLOR_MODEL_FILE).is_file():
            ctx.color_model = ColorModel.load(folder / COLOR_MODEL_FILE)
        if (folder / DENSE_FILE).is_file():
            from .dense import DenseMatcher

            ctx.matcher = DenseMatcher.load(folder / DENSE_FILE)
        if (folder / PROTOTYPES_FILE).is_file():
            from .embed import PrototypeBank

            ctx.bank = PrototypeBank.load(folder / PROTOTYPES_FILE)
        if folder.is_dir():
            # Créée même si aucun balayage n'a encore eu lieu : l'analyse à la
            # demande d'une tuile a besoin d'y ranger ses candidats pour qu'ils
            # soient acceptables d'un clic.
            ctx.candidates = CandidateStore(folder / CANDIDATES_FILE)
        ctx.calibration = load_calibration(folder / "calibration.json")
        return ctx

    @property
    def is_empty(self) -> bool:
        return self.color_model is None and self.matcher is None

    def embedder(self):
        if self._embedder is None:
            from .embed import Embedder

            self._embedder = Embedder(self.model_id, threads=self.threads)
        return self._embedder

    def thresholds(self, kind: str) -> Dict[str, float]:
        return {c: v.threshold for c, v in self.calibration.get(kind, {}).items()}

    def summary(self) -> Dict:
        from .calibrate import recommendation

        cal = self.calibration
        return {
            "available": not self.is_empty,
            "has_color": self.color_model is not None,
            "has_dense": self.matcher is not None,
            "has_candidates": self.candidates is not None,
            "recommendation": recommendation(cal.get("color", {}), cal.get("dense", {})),
            "calibration": {
                kind: {c: v.as_dict() for c, v in cal.get(kind, {}).items()}
                for kind in ("color", "dense")
            },
        }

    # ---- analyse à la demande d'une tuile --------------------------------------

    def analyze_tile(self, tiler: RasterTiler, index: int, mode: str,
                     species: Optional[List[str]], known_xy) -> List:
        """Détecte des candidats sur la seule tuile affichée.

        Sert quand l'utilisateur veut une réponse tout de suite sur ce qu'il a
        sous les yeux, sans attendre un balayage complet.
        """
        from .calibrate import recommendation
        from .prospect import Candidate, ScanParams, deduplicate, detect_blobs, _Grid

        info = tiler.tile_info(index)
        params = ScanParams()
        reco = recommendation(self.calibration.get("color", {}),
                              self.calibration.get("dense", {}))
        found: List[Candidate] = []

        codes_color: List[str] = []
        codes_dense: List[str] = []
        pool = species or (list(self.color_model.species()) if self.color_model else [])
        for c in pool:
            r = reco.get(c, "couleur") if mode == "auto" else mode
            if r == "dense" and self.matcher is not None and c in self.matcher.species:
                codes_dense.append(c)
            elif r != "aucun" and self.color_model is not None and c in self.color_model.llr:
                codes_color.append(c)

        if codes_color and self.color_model is not None:
            factor = tiler.best_factor_for(self.color_model.gsd_m)
            gsd = tiler.res_x * factor
            block = tiler.read_window(
                info.col_off, info.row_off, info.width, info.height,
                out_shape=(max(1, info.height // factor), max(1, info.width // factor)))
            blobs = detect_blobs(block, self.color_model, gsd, codes_color,
                                 self.thresholds("color"),
                                 params.min_diam_m, params.max_diam_m)
            from rasterio import transform as rio_transform

            for code, items in blobs.items():
                for (br, bc, area, mean) in items:
                    x, y = rio_transform.xy(tiler.transform,
                                            info.row_off + br * factor,
                                            info.col_off + bc * factor, offset="center")
                    import math

                    found.append(Candidate(
                        species=code, x=float(x), y=float(y),
                        color_score=float(mean * math.log1p(area / (gsd * gsd))),
                        area_m2=float(area),
                        score=float(mean * math.log1p(area / (gsd * gsd)))))

        if codes_dense and self.matcher is not None:
            emb = self.embedder()
            cx = (info.left + info.right) / 2.0
            cy = (info.bottom + info.top) / 2.0
            res = self.matcher.score_window(tiler, emb, cx, cy, codes_dense)
            if res is not None:
                scores, left, top = res
                thr = self.thresholds("dense")
                for code, smap in scores.items():
                    for (x, y, s) in self.matcher.peaks(
                            smap, left, top, thr.get(code, 0.03), max_peaks=30):
                        found.append(Candidate(
                            species=code, x=x, y=y, color_score=0.0,
                            area_m2=self.matcher.params.token_m() ** 2,
                            embed_score=s, best_code=code, score=s))

        found = deduplicate(found, params.min_sep_m)
        kx, ky = known_xy
        if len(kx):
            grid = _Grid(max(params.exclude_m, 0.5))
            grid.add_many(kx, ky)
            found = [c for c in found if not grid.has_within(c.x, c.y, params.exclude_m)]
        # On ne garde que ce qui tombe réellement dans la tuile affichée.
        return [c for c in found
                if info.left <= c.x <= info.right and info.bottom <= c.y <= info.top]

    def close(self) -> None:
        if self.candidates is not None:
            self.candidates.close()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def create_app(
    config: AppConfig,
    tiler: RasterTiler,
    store: AnnotationStore,
    references: ReferenceLibrary,
    overlay: ExistingAnnotations,
    prospect: Optional[ProspectContext] = None,
) -> FastAPI:
    app = FastAPI(title="Annotateur d'orthomosaïques")

    species = store.species
    colors = {sp: _PALETTE[i % len(_PALETTE)] for i, sp in enumerate(species)}
    folder = config.raster_path.parent
    ctx = {"tiler": tiler, "name": config.raster_path.name}

    def T() -> RasterTiler:
        return ctx["tiler"]

    def known_xy():
        """Tous les points déjà connus (existants + posés), pour filtrer les candidats."""
        kx, ky = overlay.all_xy()
        own = store.iter_all_points()
        ox = [p[0] for pts in own.values() for p in pts]
        oy = [p[1] for pts in own.values() for p in pts]
        if ox:
            kx = np.concatenate([kx, np.asarray(ox)])
            ky = np.concatenate([ky, np.asarray(oy)])
        return kx, ky

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/api/config")
    def api_config() -> JSONResponse:
        return JSONResponse({
            "raster": T().metadata(),
            "current_raster": ctx["name"],
            "available_rasters": list_rasters(folder),
            "species": species,
            "colors": colors,
            "panel_position": config.panel_position,
            "output_gpkg": str(config.output_gpkg),
            "counts": store.counts(),
            "references": references.as_dict(),
            "reference_crop_m": references.crop_m,
            "existing": {
                "available": overlay.available,
                "path": str(overlay.path) if overlay.path else None,
            },
            "prospect": prospect.summary() if prospect else {"available": False},
        })

    @app.get("/api/thumbnail.png")
    def api_thumbnail() -> Response:
        return Response(content=T().thumbnail_png(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/coverage")
    def api_coverage() -> JSONResponse:
        t = T()
        tw, th = t.thumb_size()
        return JSONResponse({
            "n_cols": t.n_cols, "n_rows": t.n_rows, "n_tiles": t.n_tiles,
            "thumb_w": tw, "thumb_h": th,
            "coverage": t.coverage_list(), "first_nonempty": t.first_nonempty,
        })

    @app.get("/api/nav/next-nonempty")
    def api_next_nonempty(index: int, step: int = 1) -> JSONResponse:
        return JSONResponse({"index": T().next_nonempty(index, step)})

    def _to_thumb_fn(t: RasterTiler):
        left, bottom, right, top = t.bounds
        W, H = right - left, top - bottom
        tw, th = t.thumb_size()

        def f(x, y):
            return [round((x - left) / W * tw, 1), round((top - y) / H * th, 1)]

        return f, tw, th

    @app.get("/api/minimap-points")
    def api_minimap_points(own_only: int = 0) -> JSONResponse:
        t = T()
        to_thumb, tw, th = _to_thumb_fn(t)
        own: Dict[str, List] = {}
        for code, pts in store.iter_all_points().items():
            own[code] = [to_thumb(x, y) for (x, y, _lab) in pts]

        existing: List = []
        if not own_only and overlay.available:
            left, bottom, right, top = t.bounds
            raw = overlay.points_in_bbox(left, bottom, right, top)
            for _sp, items in raw.items():
                for it in items:
                    existing.append(to_thumb(it["x"], it["y"]))
                    if len(existing) >= 8000:
                        break
        return JSONResponse({"thumb_w": tw, "thumb_h": th,
                             "own": own, "existing": existing,
                             "own_only": bool(own_only)})

    @app.get("/api/minimap-candidates")
    def api_minimap_candidates() -> JSONResponse:
        t = T()
        if not (prospect and prospect.candidates):
            return JSONResponse({"points": [], "thumb_w": 0, "thumb_h": 0})
        to_thumb, tw, th = _to_thumb_fn(t)
        xs, ys, codes = prospect.candidates.all_new_xy(ctx["name"])
        pts = [to_thumb(x, y) + [c] for x, y, c in zip(xs, ys, codes)]
        return JSONResponse({"points": pts[:8000], "thumb_w": tw, "thumb_h": th})

    @app.get("/api/tile/{index}.img")
    def api_tile_img(index: int, background_tasks: BackgroundTasks,
                     dir: int = 1, r: str = "") -> Response:
        t = T()
        try:
            blob = t.read_tile_image(index)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        background_tasks.add_task(t.prefetch, t.neighbor_indices(index, dir))
        return Response(content=blob, media_type=t.media_type,
                        headers={"Cache-Control": "private, max-age=600"})

    @app.get("/api/tile/{index}/info")
    def api_tile_info(index: int) -> JSONResponse:
        t = T()
        try:
            info = t.tile_info(index)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        pts = store.points_in_bbox(info.left, info.bottom, info.right, info.top)
        by_species: Dict[str, List[Dict]] = {}
        for sp, items in pts.items():
            drawn = []
            for it in items:
                px, py = t.utm_to_pixel(index, it["x"], it["y"])
                drawn.append({"fid": it["fid"], "px": px, "py": py})
            by_species[sp] = drawn
        # zones à revoir intersectant la tuile -> rectangles en pixels
        boxes = []
        for b in store.review_boxes_in_bbox(info.left, info.bottom, info.right, info.top):
            px0, py0 = t.utm_to_pixel(index, b["minx"], b["maxy"])  # coin haut-gauche
            px1, py1 = t.utm_to_pixel(index, b["maxx"], b["miny"])  # coin bas-droit
            boxes.append({"fid": b["fid"], "px0": px0, "py0": py0,
                          "px1": px1, "py1": py1, "note": b["note"]})
        return JSONResponse({
            "index": info.index, "row": info.row, "col": info.col,
            "width": info.width, "height": info.height,
            "bounds": {"left": info.left, "bottom": info.bottom,
                       "right": info.right, "top": info.top},
            "points": by_species, "review_boxes": boxes,
            "is_empty": t.is_empty(index),
        })

    @app.get("/api/tile/{index}/existing")
    def api_tile_existing(index: int) -> JSONResponse:
        t = T()
        try:
            info = t.tile_info(index)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        if not overlay.available:
            return JSONResponse({"available": False, "points": {}})
        raw = overlay.points_in_bbox(info.left, info.bottom, info.right, info.top)
        by_species: Dict[str, List[Dict]] = {}
        for sp, items in raw.items():
            drawn = []
            for it in items:
                px, py = t.utm_to_pixel(index, it["x"], it["y"])
                drawn.append({"px": px, "py": py})
            by_species[sp] = drawn
        return JSONResponse({"available": True, "points": by_species})

    @app.post("/api/annotate")
    def api_annotate(body: AnnotateBody) -> JSONResponse:
        if body.species not in species:
            raise HTTPException(status_code=400, detail=f"espèce inconnue: {body.species}")
        t = T()
        try:
            x, y = t.pixel_to_utm(body.index, body.px, body.py)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        fid = store.add_point(body.species, x, y)
        px, py = t.utm_to_pixel(body.index, x, y)
        return JSONResponse({"fid": fid, "species": body.species, "x": x, "y": y,
                             "px": px, "py": py, "counts": store.counts()})

    @app.post("/api/delete")
    def api_delete(body: DeleteBody) -> JSONResponse:
        ok = store.delete_point(body.species, body.fid)
        if not ok:
            raise HTTPException(status_code=404, detail="point introuvable")
        return JSONResponse({"deleted": True, "counts": store.counts()})

    @app.post("/api/undo")
    def api_undo() -> JSONResponse:
        result = store.undo()
        if result is None:
            return JSONResponse({"undone": None, "counts": store.counts()})
        result["counts"] = store.counts()
        return JSONResponse(result)

    # ---- zones à revoir --------------------------------------------------------

    @app.post("/api/review")
    def api_review_add(body: ReviewBody) -> JSONResponse:
        t = T()
        try:
            x0, y0 = t.pixel_to_utm(body.index, body.px0, body.py0)
            x1, y1 = t.pixel_to_utm(body.index, body.px1, body.py1)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        minx, maxx = min(x0, x1), max(x0, x1)
        miny, maxy = min(y0, y1), max(y0, y1)
        fid = store.add_review_box(minx, miny, maxx, maxy, body.note)
        return JSONResponse({"fid": fid, "review_count": len(store.list_review_boxes())})

    @app.post("/api/review/delete")
    def api_review_delete(body: ReviewDeleteBody) -> JSONResponse:
        ok = store.delete_review_box(body.fid)
        if not ok:
            raise HTTPException(status_code=404, detail="zone introuvable")
        return JSONResponse({"deleted": True})

    @app.get("/api/review/list")
    def api_review_list() -> JSONResponse:
        t = T()
        boxes = store.list_review_boxes()
        for b in boxes:
            cx = (b["minx"] + b["maxx"]) / 2.0
            cy = (b["miny"] + b["maxy"]) / 2.0
            b["tile_index"] = t.index_for_utm(cx, cy)
        return JSONResponse({"boxes": boxes})

    # ---- prospection -----------------------------------------------------------

    def _need_prospect() -> "ProspectContext":
        if prospect is None:
            raise HTTPException(status_code=404, detail="prospection non activée")
        return prospect

    def _candidates_payload(index: int, cands) -> List[Dict]:
        t = T()
        out = []
        for c in cands:
            px, py = t.utm_to_pixel(index, c.x, c.y)
            d = c.as_dict()
            d["px"], d["py"] = px, py
            out.append(d)
        return out

    @app.get("/api/tile/{index}/candidates")
    def api_tile_candidates(index: int) -> JSONResponse:
        p = _need_prospect()
        if p.candidates is None:
            return JSONResponse({"candidates": []})
        t = T()
        try:
            info = t.tile_info(index)
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        cands = p.candidates.in_bbox(ctx["name"], info.left, info.bottom,
                                     info.right, info.top, include_decided=False)
        return JSONResponse({"candidates": _candidates_payload(index, cands)})

    @app.post("/api/analyze-tile")
    def api_analyze_tile(body: AnalyzeBody) -> JSONResponse:
        p = _need_prospect()
        if p.color_model is None and p.matcher is None:
            raise HTTPException(status_code=400, detail="aucun détecteur disponible")
        t = T()
        try:
            found = p.analyze_tile(t, body.index, body.mode, body.species, known_xy())
        except IndexError:
            raise HTTPException(status_code=404, detail="tuile hors bornes")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        if p.candidates is not None and found:
            # Persistés pour être acceptables d'un clic (et conserver la décision).
            p.candidates.add(ctx["name"], found)
            info = t.tile_info(body.index)
            found = p.candidates.in_bbox(ctx["name"], info.left, info.bottom,
                                         info.right, info.top, include_decided=False)
        return JSONResponse({"candidates": _candidates_payload(body.index, found),
                             "count": len(found)})

    @app.post("/api/candidates/accept")
    def api_candidate_accept(body: CandidateActionBody) -> JSONResponse:
        p = _need_prospect()
        if p.candidates is None:
            raise HTTPException(status_code=404, detail="aucun candidat")
        cand = p.candidates.get(body.id)
        if cand is None:
            raise HTTPException(status_code=404, detail="candidat introuvable")
        code = body.species or cand.species
        if code not in species:
            raise HTTPException(status_code=400, detail=f"espèce inconnue: {code}")
        fid = store.add_point(code, cand.x, cand.y)
        p.candidates.set_status(body.id, "accepted")
        t = T()
        idx = t.index_for_utm(cand.x, cand.y)
        px, py = t.utm_to_pixel(idx, cand.x, cand.y)
        return JSONResponse({"fid": fid, "species": code, "px": px, "py": py,
                             "counts": store.counts()})

    @app.post("/api/candidates/reject")
    def api_candidate_reject(body: CandidateActionBody) -> JSONResponse:
        p = _need_prospect()
        if p.candidates is None or not p.candidates.set_status(body.id, "rejected"):
            raise HTTPException(status_code=404, detail="candidat introuvable")
        return JSONResponse({"rejected": True})

    @app.get("/api/candidates/next")
    def api_candidates_next(index: int, species_code: str = "",
                            skip: int = 0) -> JSONResponse:
        """Tuile du meilleur candidat suivant (parcours par score décroissant)."""
        p = _need_prospect()
        if p.candidates is None:
            raise HTTPException(status_code=404, detail="aucun candidat")
        t = T()
        rows = p.candidates.ranked(ctx["name"], species_code or None, limit=2000)
        seen, order = set(), []
        for c in rows:
            i = t.index_for_utm(c.x, c.y)
            if i in seen:
                continue
            seen.add(i)
            order.append({"index": i, "species": c.species, "score": c.score})
        if not order:
            return JSONResponse({"index": None, "remaining": 0})
        pos = next((k for k, o in enumerate(order) if o["index"] == index), -1)
        nxt = order[(pos + 1 + skip) % len(order)]
        return JSONResponse({"index": nxt["index"], "species": nxt["species"],
                             "score": nxt["score"], "remaining": len(order)})

    @app.get("/api/candidates/counts")
    def api_candidates_counts() -> JSONResponse:
        p = _need_prospect()
        if p.candidates is None:
            return JSONResponse({"counts": {}})
        return JSONResponse({"counts": p.candidates.counts(ctx["name"])})

    # ---- changement d'orthomosaïque -------------------------------------------

    @app.post("/api/switch-raster")
    def api_switch_raster(body: SwitchBody) -> JSONResponse:
        available = list_rasters(folder, include_padded=True)
        if body.name not in available:
            raise HTTPException(status_code=400, detail="orthomosaïque inconnue")
        new_path = folder / body.name
        new_tiler = RasterTiler(new_path, config.tile_size_m, config.overlap_m,
                                config.cache_size, tile_format=config.tile_format,
                                tile_quality=config.tile_quality)
        if new_tiler.crs.to_epsg() != T().crs.to_epsg():
            new_tiler.close()
            raise HTTPException(status_code=400,
                                detail="CRS différent, non pris en charge")
        old = ctx["tiler"]
        ctx["tiler"] = new_tiler
        ctx["name"] = body.name
        old.close()
        return JSONResponse({
            "current_raster": body.name,
            "raster": new_tiler.metadata(),
        })

    @app.get("/api/reference-image")
    def api_reference_image(code: str, kind: str = "aerial", idx: int = 0) -> FileResponse:
        path = references.resolve_image(code, kind, idx)
        if path is None:
            raise HTTPException(status_code=404, detail="image de référence introuvable")
        return FileResponse(str(path),
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/crop")
    def api_crop(x: float, y: float, size_m: float = 0.6, out: int = 256) -> Response:
        """Vignette carrée centrée sur une coordonnée UTM (lecture fenêtrée).

        Utilisée pour montrer un candidat à côté des références, à la même
        échelle, sans quitter la tuile courante.
        """
        t = T()
        size_px = max(8, int(round(size_m / t.res_x)))
        arr = t.read_centered(x, y, size_px, out_size=max(32, min(512, out)))
        if arr.size == 0:
            raise HTTPException(status_code=404, detail="hors emprise")
        return Response(content=t.encode_array(arr), media_type=t.media_type,
                        headers={"Cache-Control": "private, max-age=600"})

    return app
