"""Points d'entrée en ligne de commande : ``serve``, ``validate``, ``export-flat``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import (
    CANDIDATES_FILE,
    COLOR_MODEL_FILE,
    DEFAULT_EXISTING_ANNOTATIONS,
    DENSE_FILE,
    FORBIDDEN_OUTPUT_NAMES,
    PROTOTYPES_FILE,
    AppConfig,
    prospect_dir,
)


def _guard_output_path(output: Path, raster: Path, existing: Optional[Path]) -> None:
    """Refuse toute sortie qui écraserait une source ou toucherait Dataset_Leo/."""
    if output.name in FORBIDDEN_OUTPUT_NAMES:
        raise SystemExit(
            f"REFUS : nom de sortie interdit ({output.name}). "
            "Ne jamais écrire dans Annotations.gpkg ni Annotations_polyg.gpkg."
        )
    if raster.exists() and output.exists() and output.samefile(raster):
        raise SystemExit("REFUS : la sortie est identique au raster source.")
    if existing and existing.exists() and output.exists() and output.samefile(existing):
        raise SystemExit("REFUS : la sortie est identique au jeu d'annotations existant.")
    if "Dataset_Leo" in output.parts:
        raise SystemExit(
            "REFUS : écriture interdite sous Dataset_Leo/ (lecture seule stricte). "
            "Choisir un chemin de sortie ailleurs."
        )


def _resolve_species(existing: Optional[Path], output: Path) -> List[str]:
    """Détermine la liste d'espèces (noms de couches), sans en inventer aucun."""
    from .geopackage import REVIEW_LAYER, read_layer_names

    def _filter(names: List[str]) -> List[str]:
        return [n for n in names if n != REVIEW_LAYER]

    if existing and existing.is_file():
        names = _filter(read_layer_names(existing))
        if names:
            return names
    if output.is_file():
        names = _filter(read_layer_names(output))
        if names:
            return names
    raise SystemExit(
        "Impossible de déterminer la liste d'espèces : fournir --existing-annotations "
        "pointant vers un GeoPackage de couches d'espèces, ou un --output déjà initialisé."
    )


def _build_config(args: argparse.Namespace) -> AppConfig:
    existing = Path(args.existing_annotations) if args.existing_annotations else None
    reference_dir = Path(args.reference_dir) if args.reference_dir else None
    return AppConfig(
        raster_path=Path(args.raster),
        output_gpkg=Path(args.output),
        existing_annotations=existing,
        reference_dir=reference_dir,
        tile_size_m=args.tile_size_m,
        overlap_m=args.overlap_m,
        cache_size=args.cache_size,
        panel_position=args.panel_position,
        host=args.host,
        port=args.port,
        tile_format=args.tile_format,
        tile_quality=args.tile_quality,
    ).normalized()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .geopackage import AnnotationStore
    from .overlay import ExistingAnnotations
    from .raster import RasterTiler
    from .references import ReferenceLibrary
    from .server import create_app
    from .validation import run_coordinate_validation

    config = _build_config(args)
    _guard_output_path(config.output_gpkg, config.raster_path, config.existing_annotations)

    # Validation obligatoire des coordonnées AVANT toute annotation réelle.
    ok, report = run_coordinate_validation(
        config.raster_path, config.tile_size_m, config.overlap_m
    )
    print("== Validation des coordonnées (pré-lancement) ==")
    for line in report:
        print("  " + line)
    if not ok:
        print("ÉCHEC de la validation des coordonnées : lancement annulé.", file=sys.stderr)
        return 2

    tiler = RasterTiler(
        config.raster_path, config.tile_size_m, config.overlap_m, config.cache_size,
        tile_format=config.tile_format, tile_quality=config.tile_quality,
    )
    epsg = tiler.crs.to_epsg()
    if epsg is None:
        tiler.close()
        raise SystemExit("Le raster n'a pas de code EPSG exploitable.")

    species = _resolve_species(config.existing_annotations, config.output_gpkg)

    # Cohérence CRS entre le raster et la source d'espèces (les points sont stockés
    # dans le CRS du raster ; le jeu existant vérifié est EPSG:32618).
    if config.existing_annotations and config.existing_annotations.is_file():
        import fiona

        with fiona.open(str(config.existing_annotations), layer=species[0]) as src:
            src_epsg = src.crs.to_epsg() if src.crs else None
        if src_epsg is not None and src_epsg != epsg:
            tiler.close()
            raise SystemExit(
                f"CRS incohérent : raster EPSG:{epsg} vs annotations EPSG:{src_epsg}."
            )

    store = AnnotationStore(config.output_gpkg, species, epsg)
    mapping = ReferenceLibrary.load_mapping_file(
        Path(args.reference_map) if args.reference_map else None
    )
    references = ReferenceLibrary(
        ReferenceLibrary.default_roots(config.reference_dir), species, mapping
    )
    overlay = ExistingAnnotations(config.existing_annotations, epsg)

    prospect = _load_prospect(config, args) if not args.no_prospect else None

    app = create_app(config, tiler, store, references, overlay, prospect)

    print(f"\nOrthomosaïque : {config.raster_path.name}")
    print(f"Sortie (UNIQUE) : {config.output_gpkg}")
    print(f"Espèces : {', '.join(species)}")
    print(f"Grille : {tiler.n_cols} x {tiler.n_rows} = {tiler.n_tiles} tuiles "
          f"({tiler.tile_px_x}x{tiler.tile_px_y} px, {config.tile_size_m} m, "
          f"encodage {config.tile_format})")
    print(f"Points déjà posés : {store.counts()}")
    if prospect and prospect.candidates is not None:
        n = sum(sum(st.values())
                for st in prospect.candidates.counts(config.raster_path.name).values())
        print(f"Candidats de prospection pour cette orthomosaïque : {n}")
    print(f"\nOuvrir : http://{config.host}:{config.port}/\n")

    try:
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    finally:
        if prospect:
            prospect.close()
        store.close()
        tiler.close()
    return 0


def _load_prospect(config: AppConfig, args: argparse.Namespace):
    """Charge les artefacts de prospection s'ils existent (jamais bloquant)."""
    from .server import ProspectContext

    out_dir = prospect_dir(config.output_gpkg)
    try:
        ctx = ProspectContext.load(out_dir, model_id=args.model, threads=args.threads)
    except Exception as exc:                      # jamais fatal : l'outil marche sans
        print(f"Prospection indisponible : {exc}")
        return None
    if ctx.is_empty:
        print("Prospection : aucun modèle trouvé "
              "(lancer « prospect learn » pour l'activer).")
        return None
    bits = []
    if ctx.color_model is not None:
        bits.append("couleur")
    if ctx.matcher is not None:
        bits.append("dense")
    if ctx.candidates is not None:
        bits.append("candidats")
    print(f"Prospection : {', '.join(bits)} — {out_dir}")
    return ctx


def cmd_validate(args: argparse.Namespace) -> int:
    from .validation import run_coordinate_validation, run_roundtrip_validation

    raster = Path(args.raster).expanduser().resolve()
    existing = (
        Path(args.existing_annotations).expanduser().resolve()
        if args.existing_annotations
        else None
    )
    output_placeholder = Path("/tmp/__unused__.gpkg")
    species = _resolve_species(existing, output_placeholder)

    print("== 1. Validation conversion pixel -> UTM ==")
    ok1, rep1 = run_coordinate_validation(raster, args.tile_size_m, args.overlap_m)
    for line in rep1:
        print("  " + line)

    print("\n== 2. Validation aller-retour API (GeoPackage temporaire) ==")
    ok2, rep2 = run_roundtrip_validation(raster, species, args.tile_size_m, args.overlap_m)
    for line in rep2:
        print("  " + line)

    ok = ok1 and ok2
    print(f"\nRésultat global : {'SUCCÈS' if ok else 'ÉCHEC'}")
    return 0 if ok else 1


def cmd_export_flat(args: argparse.Namespace) -> int:
    """Export secondaire optionnel : UNE seule couche avec tous les points + Label."""
    import geopandas as gpd
    import pandas as pd

    from .geopackage import AnnotationStore, read_layer_names

    source = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if "Dataset_Leo" in dest.parts:
        raise SystemExit("REFUS : export interdit sous Dataset_Leo/.")
    if dest.name in FORBIDDEN_OUTPUT_NAMES:
        raise SystemExit(f"REFUS : nom de sortie interdit ({dest.name}).")
    if not source.is_file():
        raise SystemExit(f"Source introuvable : {source}")

    from .geopackage import REVIEW_LAYER

    species = [n for n in read_layer_names(source) if n != REVIEW_LAYER]
    import fiona

    with fiona.open(str(source), layer=species[0]) as src:
        epsg = src.crs.to_epsg() if src.crs else None
    store = AnnotationStore(source, species, epsg or 32618)
    try:
        all_pts = store.iter_all_points()
    finally:
        store.close()

    from shapely.geometry import Point

    labels: List[str] = []
    geoms: List[Point] = []
    for sp, pts in all_pts.items():
        for x, y, label in pts:
            labels.append(label)
            geoms.append(Point(x, y))
    gdf = gpd.GeoDataFrame({"Label": labels}, geometry=geoms, crs=f"EPSG:{epsg or 32618}")
    layer_name = args.layer or "annotations"
    gdf.to_file(dest, layer=layer_name, driver="GPKG")
    print(f"Export à plat écrit : {dest} (couche '{layer_name}', {len(gdf)} points)")
    return 0


def cmd_rebuild_crops(args: argparse.Namespace) -> int:
    """Régénère les vignettes aériennes de référence, à emprise au sol connue.

    Chaque vignette est une fenêtre carrée de ``--crop-m`` mètres centrée sur un
    point déjà annoté, lue de façon fenêtrée. L'emprise est écrite dans
    ``meta.json`` : c'est elle qui permet ensuite à l'interface d'afficher les
    références **à la même échelle** que la tuile en cours d'annotation.
    """
    import json

    import numpy as np
    from PIL import Image

    from .references import AERIAL_DIR

    existing = Path(args.existing_annotations).expanduser().resolve()
    folder = Path(args.raster_dir).expanduser().resolve() if args.raster_dir \
        else existing.parent
    tilers = _open_tilers(folder, args.rasters, 5.0)
    epsg = tilers[0].crs.to_epsg()
    ex, points = _points_by_species(existing, epsg)

    out_root = Path(args.dest).expanduser().resolve() if args.dest else AERIAL_DIR
    out_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    total = 0
    for code, (xs, ys) in points.items():
        crops = []
        for tiler in tilers:
            m = tiler.inside_mask(xs, ys)
            if not m.any():
                continue
            cx, cy = xs[m], ys[m]
            take = min(args.per_species, cx.size)
            sel = rng.choice(cx.size, size=take, replace=False)
            size_px = max(16, int(round(args.crop_m / tiler.res_x)))
            for i in sel:
                arr = tiler.read_centered(float(cx[i]), float(cy[i]), size_px,
                                          out_size=args.out_px)
                if arr.size:
                    crops.append(arr)
                if len(crops) >= args.per_species:
                    break
            if len(crops) >= args.per_species:
                break
        d = out_root / code
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("crop_*.png"):
            old.unlink()
        for i, arr in enumerate(crops):
            Image.fromarray(arr[..., :3]).save(d / f"crop_{i:02d}.png")
        total += len(crops)
        print(f"  {code:10s} {len(crops)} vignette(s) de {args.crop_m} m")
    (out_root / "meta.json").write_text(
        json.dumps({"crop_m": args.crop_m, "out_px": args.out_px}, indent=1),
        encoding="utf-8")
    print(f"\n{total} vignettes écrites dans {out_root} (emprise {args.crop_m} m)")
    for t in tilers:
        t.close()
    return 0


# ---------------------------------------------------------------------------
# Prospection (aide à la recherche de plantes)
# ---------------------------------------------------------------------------


def _points_by_species(existing: Optional[Path], epsg: int):
    """Points annotés par espèce, lus en LECTURE SEULE."""
    from .overlay import ExistingAnnotations

    ex = ExistingAnnotations(existing, epsg)
    if not ex.available:
        raise SystemExit(
            "Aucun jeu d'annotations existant : la prospection s'appuie dessus "
            "(--existing-annotations)."
        )
    return ex, {code: ex.arrays(code) for code in ex.species()}


def _open_tilers(folder: Path, names, tile_size_m: float, log=print):
    from .raster import RasterTiler, list_rasters

    names = list(names) if names else list_rasters(folder)
    tilers = []
    for name in names:
        p = folder / name
        if not p.is_file():
            log(f"  (ignoré, introuvable : {name})")
            continue
        tilers.append(RasterTiler(p, tile_size_m, 0.0, cache_size=2))
    if not tilers:
        raise SystemExit("Aucune orthomosaïque exploitable.")
    return tilers


def _load_embedder(model_id: Optional[str], threads: int, log=print,
                    device: Optional[str] = None):
    """Charge l'encodeur figé si les poids sont là ; sinon renvoie None."""
    try:
        from .embed import Embedder

        emb = Embedder(model_id, threads=threads, device=device)
        log(f"  encodeur : {emb.model_id} ({emb.device}, {threads} threads)")
        return emb
    except Exception as exc:
        log(f"  encodeur indisponible ({exc}) -> étage couleur uniquement")
        return None


def cmd_prospect_learn(args: argparse.Namespace) -> int:
    """Apprend les modèles de détection sur les annotations existantes."""
    import json

    from .calibrate import (calibrate_color, calibrate_dense, recommendation,
                            save_calibration)
    from .colormodel import ColorModel
    from .dense import DenseMatcher, DenseParams
    from .embed import PrototypeBank

    output = Path(args.output).expanduser().resolve()
    existing = Path(args.existing_annotations).expanduser().resolve()
    folder = Path(args.raster_dir).expanduser().resolve() if args.raster_dir \
        else existing.parent
    out_dir = prospect_dir(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Orthomosaïques : {folder}")
    tilers = _open_tilers(folder, args.rasters, args.tile_size_m)
    epsg = tilers[0].crs.to_epsg()
    ex, points = _points_by_species(existing, epsg)
    print(f"  {len(tilers)} orthomosaïque(s), points annotés : {ex.counts()}\n")

    print("== 1. Modèle couleur ==")
    model = ColorModel.learn(
        tilers, points,
        factor_for=lambda t: t.best_factor_for(args.gsd_m),
        gsd_m=args.gsd_m, max_points=args.max_points)
    model.save(out_dir / COLOR_MODEL_FILE)

    embedder = None
    matcher = None
    if not args.no_embed:
        print("\n== 2. Encodeur figé ==")
        embedder = _load_embedder(args.model, args.threads, device=args.device)

    if embedder is not None:
        print("\n== 3. Jetons de référence (détection dense) ==")
        matcher = DenseMatcher.build(
            tilers, embedder, points,
            DenseParams(span_m=args.span_m, side_px=args.side_px,
                        windows_per_species=args.windows_per_species))
        matcher.save(out_dir / DENSE_FILE)

        print("\n== 4. Prototypes de vignettes (re-classement) ==")
        bank = PrototypeBank.build(tilers, embedder, points, crop_m=args.crop_m)
        bank.save(out_dir / PROTOTYPES_FILE)

    print("\n== 5. Calibration mesurée (fenêtres réservées) ==")
    print(" couleur :")
    cal_color = calibrate_color(tilers, model, points)
    cal_dense = {}
    if matcher is not None:
        print(" dense :")
        cal_dense = calibrate_dense(tilers, matcher, embedder, points)
    save_calibration(out_dir / "calibration.json", cal_color, cal_dense)

    reco = recommendation(cal_color, cal_dense)
    print("\n== Résumé : quel détecteur pour quelle espèce ==")
    print(f"  {'espèce':10s} {'couleur (R/P)':>16s} {'dense (R/P)':>16s}   conseil")
    for code in sorted(reco):
        c = cal_color.get(code)
        d = cal_dense.get(code)
        cs = f"{c.recall:.2f}/{c.precision:.2f}" if c else "—"
        ds = f"{d.recall:.2f}/{d.precision:.2f}" if d else "—"
        print(f"  {code:10s} {cs:>16s} {ds:>16s}   {reco[code]}")
    print("\n  R = rappel, P = précision MINORÉE (les annotations existantes sont "
          "incomplètes :\n  une proposition comptée fausse est parfois une plante "
          "jamais pointée).")
    print(f"\nArtefacts écrits dans {out_dir}")
    for t in tilers:
        t.close()
    return 0


def cmd_prospect_scan(args: argparse.Namespace) -> int:
    """Balaye une orthomosaïque et enregistre les candidats."""
    from .calibrate import load_calibration, recommendation
    from .colormodel import ColorModel
    from .dense import DenseMatcher
    from .embed import PrototypeBank
    from .geopackage import AnnotationStore, read_layer_names
    from .prospect import (CandidateStore, ScanParams, scan_raster,
                           scan_raster_dense)
    from .raster import RasterTiler

    output = Path(args.output).expanduser().resolve()
    raster = Path(args.raster).expanduser().resolve()
    out_dir = prospect_dir(output)
    if not (out_dir / COLOR_MODEL_FILE).is_file():
        raise SystemExit("Modèles absents : lancer d'abord « prospect learn ».")

    tiler = RasterTiler(raster, args.tile_size_m, 0.0, cache_size=2)
    epsg = tiler.crs.to_epsg()
    model = ColorModel.load(out_dir / COLOR_MODEL_FILE)
    cal = load_calibration(out_dir / "calibration.json")
    reco = recommendation(cal["color"], cal["dense"])

    # Points déjà connus = jeu existant + ce que l'utilisateur a déjà posé.
    existing = Path(args.existing_annotations).expanduser().resolve()
    ex, _pts = _points_by_species(existing, epsg)
    kx, ky = ex.all_xy()
    if output.is_file():
        species = [n for n in read_layer_names(output) if n != "zones_a_revoir"]
        store = AnnotationStore(output, species, epsg)
        try:
            own = store.iter_all_points()
        finally:
            store.close()
        ox = [p[0] for pts in own.values() for p in pts]
        oy = [p[1] for pts in own.values() for p in pts]
        if ox:
            import numpy as np

            kx = np.concatenate([kx, np.array(ox)])
            ky = np.concatenate([ky, np.array(oy)])

    params = ScanParams(min_sep_m=args.min_sep_m, exclude_m=args.exclude_m,
                        max_per_species=args.max_per_species,
                        rerank_top=0 if args.no_rerank else args.rerank_top,
                        crop_m=args.crop_m)

    wanted = args.species or None
    mode = args.mode
    if mode == "auto":
        color_codes = [c for c in model.species()
                       if reco.get(c, "couleur") == "couleur"]
        dense_codes = [c for c in model.species() if reco.get(c) == "dense"]
        skipped = [c for c in model.species() if reco.get(c) == "aucun"]
        if skipped:
            print(f"Espèces sans détecteur fiable, ignorées : {', '.join(skipped)}")
    else:
        color_codes = list(wanted or model.species()) if mode == "color" else []
        dense_codes = list(wanted or model.species()) if mode == "dense" else []
    if wanted:
        color_codes = [c for c in color_codes if c in wanted]
        dense_codes = [c for c in dense_codes if c in wanted]

    found = []
    if color_codes:
        print("\n== Étage couleur ==")
        bank = embedder = None
        if not args.no_rerank and (out_dir / PROTOTYPES_FILE).is_file():
            embedder = _load_embedder(args.model, args.threads, device=args.device)
            if embedder is not None:
                bank = PrototypeBank.load(out_dir / PROTOTYPES_FILE)
        found += scan_raster(
            tiler, model, (kx, ky), params, species=color_codes,
            thresholds={c: v.threshold for c, v in cal["color"].items()},
            bank=bank, embedder=embedder, embed_batch_size=args.embed_batch_size)

    if dense_codes:
        if not (out_dir / DENSE_FILE).is_file():
            print("Banque de jetons absente : étage dense sauté.")
        else:
            print("\n== Étage dense ==")
            embedder = _load_embedder(args.model, args.threads, device=args.device)
            if embedder is not None:
                matcher = DenseMatcher.load(out_dir / DENSE_FILE)
                found += scan_raster_dense(
                    tiler, matcher, embedder, (kx, ky), params,
                    thresholds={c: v.threshold for c, v in cal["dense"].items()},
                    species=dense_codes, max_windows=args.max_windows,
                    batch_size=args.embed_batch_size)

    cstore = CandidateStore(out_dir / CANDIDATES_FILE)
    try:
        n = cstore.replace_raster(raster.name, found, params.as_dict())
        print(f"\n{n} candidat(s) enregistrés pour {raster.name}")
        for sp, st in sorted(cstore.counts(raster.name).items()):
            print(f"  {sp:10s} {st}")
    finally:
        cstore.close()
    tiler.close()
    return 0


def cmd_prospect_status(args: argparse.Namespace) -> int:
    import json

    from .calibrate import load_calibration
    from .prospect import CandidateStore

    out_dir = prospect_dir(Path(args.output))
    print(f"Dossier de prospection : {out_dir}")
    for name in (COLOR_MODEL_FILE, DENSE_FILE, PROTOTYPES_FILE, "calibration.json"):
        p = out_dir / name
        print(f"  {'✓' if p.is_file() else '✗'} {name}")
    cal = load_calibration(out_dir / "calibration.json")
    if cal["color"] or cal["dense"]:
        print("\nCalibration (rappel / précision minorée) :")
        for kind in ("color", "dense"):
            for code, v in sorted(cal[kind].items()):
                print(f"  {kind:7s} {code:10s} seuil={v.threshold:<7g} "
                      f"R={v.recall:.2f} P≥{v.precision:.2f}")
    db = out_dir / CANDIDATES_FILE
    if db.is_file():
        cs = CandidateStore(db)
        try:
            print("\nCandidats :")
            for raster, meta in cs.scanned_rasters().items():
                print(f"  {raster} — balayé le {meta['created']}")
                for sp, st in sorted(cs.counts(raster).items()):
                    print(f"      {sp:10s} {st}")
        finally:
            cs.close()
    return 0


def cmd_prospect_export(args: argparse.Namespace) -> int:
    """Exporte les candidats en GeoPackage (pour les ouvrir dans QGIS)."""
    import geopandas as gpd
    from shapely.geometry import Point

    from .prospect import CandidateStore

    dest = Path(args.dest).expanduser().resolve()
    if "Dataset_Leo" in dest.parts:
        raise SystemExit("REFUS : export interdit sous Dataset_Leo/.")
    if dest.name in FORBIDDEN_OUTPUT_NAMES:
        raise SystemExit(f"REFUS : nom de sortie interdit ({dest.name}).")
    db = prospect_dir(Path(args.output)) / CANDIDATES_FILE
    if not db.is_file():
        raise SystemExit("Aucun candidat : lancer d'abord « prospect scan ».")
    cs = CandidateStore(db)
    try:
        rasters = [args.raster] if args.raster else list(cs.scanned_rasters())
        rows = []
        for r in rasters:
            rows.extend(cs.ranked(r, limit=10 ** 7))
    finally:
        cs.close()
    if not rows:
        raise SystemExit("Aucun candidat à exporter.")
    gdf = gpd.GeoDataFrame(
        {"species": [c.species for c in rows],
         "score": [c.score for c in rows],
         "color_score": [c.color_score for c in rows],
         "embed_score": [c.embed_score for c in rows],
         "status": [c.status for c in rows]},
        geometry=[Point(c.x, c.y) for c in rows], crs=f"EPSG:{args.epsg}")
    gdf.to_file(dest, layer=args.layer, driver="GPKG")
    print(f"{len(gdf)} candidat(s) exportés vers {dest} (couche '{args.layer}')")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ortho_annotator",
        description="Annotateur local ponctuel sur orthomosaïques géoréférencées.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tile-size-m", type=float, default=5.0,
                       help="Taille de tuile en mètres (défaut 5.0).")
        p.add_argument("--overlap-m", type=float, default=0.0,
                       help="Chevauchement entre tuiles en mètres (défaut 0).")
        p.add_argument("--existing-annotations", type=str,
                       default=str(DEFAULT_EXISTING_ANNOTATIONS),
                       help="GeoPackage existant (source des noms d'espèces + calque, LECTURE SEULE).")

    p_serve = sub.add_parser("serve", help="Lancer le serveur d'annotation.")
    p_serve.add_argument("--raster", required=True, help="Chemin du .tif à annoter.")
    p_serve.add_argument("--output", required=True,
                         help="GeoPackage de sortie UNIQUE (partagé entre sessions).")
    add_common(p_serve)
    p_serve.add_argument("--cache-size", type=int, default=9,
                         help="Taille du cache LRU de tuiles (défaut 9).")
    p_serve.add_argument("--reference-dir", type=str, default=None,
                         help="Dossier d'images de référence SUPPLÉMENTAIRE (le dossier "
                              "empaqueté avec l'outil est toujours utilisé).")
    p_serve.add_argument("--reference-map", type=str, default=None,
                         help="JSON optionnel code_espèce -> nom de référence.")
    p_serve.add_argument("--panel-position", choices=["top", "side"], default="side",
                         help="Position du panneau de référence (défaut side, à gauche).")
    p_serve.add_argument("--tile-format", choices=["jpeg", "png"], default="jpeg",
                         help="Encodage des tuiles : jpeg (20x plus rapide) ou png.")
    p_serve.add_argument("--tile-quality", type=int, default=90,
                         help="Qualité JPEG des tuiles (défaut 90).")
    p_serve.add_argument("--no-prospect", action="store_true",
                         help="Ne pas charger les candidats de prospection.")
    p_serve.add_argument("--model", type=str, default=None,
                         help="Identifiant de l'encodeur figé (défaut : le premier "
                              "disponible dans le cache local).")
    p_serve.add_argument("--threads", type=int, default=4,
                         help="Threads CPU pour l'encodeur (défaut 4).")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    # ---- prospection --------------------------------------------------------
    p_pro = sub.add_parser(
        "prospect",
        help="Chercher automatiquement où sont les plantes (aide à l'annotation).")
    psub = p_pro.add_subparsers(dest="prospect_command", required=True)

    p_learn = psub.add_parser(
        "learn", help="Apprendre les détecteurs depuis les annotations existantes.")
    p_learn.add_argument("--output", required=True,
                         help="GeoPackage de sortie (fixe où écrire les artefacts).")
    p_learn.add_argument("--raster-dir", type=str, default=None,
                         help="Dossier des orthomosaïques (défaut : celui du "
                              "jeu d'annotations existant).")
    p_learn.add_argument("--rasters", nargs="*", default=None,
                         help="Noms de fichiers à utiliser (défaut : tous).")
    p_learn.add_argument("--gsd-m", type=float, default=0.01,
                         help="Résolution au sol de l'étage couleur (défaut 1 cm).")
    p_learn.add_argument("--max-points", type=int, default=600,
                         help="Points annotés échantillonnés par espèce (défaut 600).")
    p_learn.add_argument("--span-m", type=float, default=5.0,
                         help="Côté de la fenêtre du détecteur dense (défaut 5 m).")
    p_learn.add_argument("--side-px", type=int, default=784,
                         help="Résolution d'entrée du détecteur dense (défaut 784).")
    p_learn.add_argument("--windows-per-species", type=int, default=6,
                         help="Fenêtres de référence par espèce (défaut 6).")
    p_learn.add_argument("--crop-m", type=float, default=0.30,
                         help="Taille des vignettes de re-classement (défaut 30 cm).")
    p_learn.add_argument("--no-embed", action="store_true",
                         help="Étage couleur seulement (aucun réseau de neurones).")
    p_learn.add_argument("--model", type=str, default=None)
    p_learn.add_argument("--threads", type=int, default=4)
    p_learn.add_argument("--device", type=str, default=None,
                         help="'cuda' ou 'cpu' (défaut : cuda si disponible).")
    add_common(p_learn)
    p_learn.set_defaults(func=cmd_prospect_learn)

    p_scan = psub.add_parser("scan", help="Balayer une orthomosaïque.")
    p_scan.add_argument("--output", required=True)
    p_scan.add_argument("--raster", required=True)
    p_scan.add_argument("--mode", choices=["auto", "color", "dense"], default="auto",
                        help="auto = suit la calibration mesurée par espèce.")
    p_scan.add_argument("--species", nargs="*", default=None)
    p_scan.add_argument("--min-sep-m", type=float, default=0.30)
    p_scan.add_argument("--exclude-m", type=float, default=0.40,
                        help="Distance en deçà de laquelle un candidat est jugé "
                             "déjà annoté (défaut 40 cm).")
    p_scan.add_argument("--max-per-species", type=int, default=1500)
    p_scan.add_argument("--rerank-top", type=int, default=900,
                        help="Candidats couleur re-classés par espèce (défaut 900). "
                             "Au-delà, ils sont écartés : leur score ne serait pas "
                             "comparable à celui des candidats vérifiés.")
    p_scan.add_argument("--no-rerank", action="store_true")
    p_scan.add_argument("--crop-m", type=float, default=0.30)
    p_scan.add_argument("--max-windows", type=int, default=0,
                        help="Limiter le nombre de fenêtres du balayage dense "
                             "(0 = pas de limite).")
    p_scan.add_argument("--model", type=str, default=None)
    p_scan.add_argument("--threads", type=int, default=4)
    p_scan.add_argument("--device", type=str, default=None,
                        help="'cuda' ou 'cpu' (défaut : cuda si disponible).")
    p_scan.add_argument("--embed-batch-size", type=int, default=16,
                        help="Fenêtres encodées ensemble par passage modèle (défaut 16). "
                             "Sur GPU, augmenter (32/64) tant que la VRAM le permet — "
                             "batch=1 laisse l'essentiel de la carte inutilisée. Sans "
                             "effet notable sur CPU.")
    add_common(p_scan)
    p_scan.set_defaults(func=cmd_prospect_scan)

    p_stat = psub.add_parser("status", help="État des modèles et des candidats.")
    p_stat.add_argument("--output", required=True)
    p_stat.set_defaults(func=cmd_prospect_status)

    p_exp2 = psub.add_parser("export", help="Exporter les candidats en GeoPackage.")
    p_exp2.add_argument("--output", required=True)
    p_exp2.add_argument("--dest", required=True)
    p_exp2.add_argument("--raster", type=str, default=None)
    p_exp2.add_argument("--layer", type=str, default="candidats")
    p_exp2.add_argument("--epsg", type=int, default=32618)
    p_exp2.set_defaults(func=cmd_prospect_export)

    p_crops = sub.add_parser(
        "rebuild-crops",
        help="Régénérer les vignettes aériennes de référence (emprise connue).")
    p_crops.add_argument("--raster-dir", type=str, default=None)
    p_crops.add_argument("--rasters", nargs="*", default=None)
    p_crops.add_argument("--dest", type=str, default=None,
                         help="Dossier de sortie (défaut : celui empaqueté).")
    p_crops.add_argument("--crop-m", type=float, default=1.2,
                         help="Emprise au sol de chaque vignette (défaut 1,2 m).")
    p_crops.add_argument("--out-px", type=int, default=384)
    p_crops.add_argument("--per-species", type=int, default=16)
    p_crops.add_argument("--existing-annotations", type=str,
                         default=str(DEFAULT_EXISTING_ANNOTATIONS))
    p_crops.set_defaults(func=cmd_rebuild_crops)

    p_val = sub.add_parser("validate", help="Exécuter les tests de validation.")
    p_val.add_argument("--raster", required=True, help="Chemin du .tif à valider.")
    add_common(p_val)
    p_val.set_defaults(func=cmd_validate)

    p_exp = sub.add_parser("export-flat", help="Export secondaire à plat (1 couche).")
    p_exp.add_argument("--source", required=True, help="GeoPackage multi-couches à aplatir.")
    p_exp.add_argument("--dest", required=True, help="GeoPackage de sortie à plat.")
    p_exp.add_argument("--layer", type=str, default="annotations",
                       help="Nom de la couche unique en sortie (défaut 'annotations').")
    p_exp.set_defaults(func=cmd_export_flat)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
