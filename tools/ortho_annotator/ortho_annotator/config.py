"""Configuration de l'annotateur (arguments CLI centralisés).

Toutes les valeurs par défaut sont explicites ici. Aucune valeur dépendante des
données (résolution, dimensions, liste d'espèces) n'est codée en dur : elle est
lue à l'exécution depuis les fichiers réels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Emplacement par défaut du jeu d'annotations existant (LECTURE SEULE).
# Sert à la fois de source canonique des noms d'espèces et de calque optionnel.
DEFAULT_EXISTING_ANNOTATIONS = (
    Path(__file__).resolve().parents[3]
    / "Dataset_Leo"
    / "Orthomosaiques"
    / "Annotations.gpkg"
)

# Dossier de référence par défaut (images iNaturalist).
DEFAULT_REFERENCE_DIR = (
    Path(__file__).resolve().parents[3] / "Dataset_Leo" / "INaturalist"
)

# Fichiers sources qu'il est STRICTEMENT interdit d'écraser en sortie.
FORBIDDEN_OUTPUT_NAMES = ("Annotations.gpkg", "Annotations_polyg.gpkg")


def prospect_dir(output_gpkg: Path) -> Path:
    """Dossier des artefacts de prospection, à côté du GeoPackage de sortie.

    Jamais sous ``Dataset_Leo/`` : ce sont des fichiers de travail dérivés.
    """
    return Path(output_gpkg).expanduser().resolve().parent / "prospect"


# Noms des artefacts produits par `prospect learn`.
COLOR_MODEL_FILE = "colormodel.npz"
PROTOTYPES_FILE = "prototypes.npz"
DENSE_FILE = "dense_tokens.npz"
CANDIDATES_FILE = "candidates.sqlite"


@dataclass
class AppConfig:
    """Paramètres résolus d'une session d'annotation."""

    raster_path: Path
    output_gpkg: Path
    existing_annotations: Optional[Path] = None
    reference_dir: Optional[Path] = None
    tile_size_m: float = 5.0
    overlap_m: float = 0.0
    cache_size: int = 9
    panel_position: str = "side"  # "top" | "side"
    host: str = "127.0.0.1"
    port: int = 8000
    tile_format: str = "jpeg"     # "jpeg" (rapide) | "png" (sans perte)
    tile_quality: int = 90

    def normalized(self) -> "AppConfig":
        """Renvoie une copie avec chemins absolus et champs validés."""
        raster_path = self.raster_path.expanduser().resolve()
        output_gpkg = self.output_gpkg.expanduser().resolve()
        existing = (
            self.existing_annotations.expanduser().resolve()
            if self.existing_annotations is not None
            else None
        )
        reference_dir = (
            self.reference_dir.expanduser().resolve()
            if self.reference_dir is not None
            else None
        )
        if self.panel_position not in ("top", "side"):
            raise ValueError(
                f"panel_position doit être 'top' ou 'side', reçu {self.panel_position!r}"
            )
        if self.tile_size_m <= 0:
            raise ValueError("tile_size_m doit être strictement positif")
        if self.overlap_m < 0:
            raise ValueError("overlap_m ne peut pas être négatif")
        if self.overlap_m >= self.tile_size_m:
            raise ValueError("overlap_m doit être strictement inférieur à tile_size_m")
        if self.cache_size < 1:
            raise ValueError("cache_size doit valoir au moins 1")
        if self.tile_format not in ("jpeg", "png"):
            raise ValueError("tile_format doit être 'jpeg' ou 'png'")
        if not (30 <= self.tile_quality <= 100):
            raise ValueError("tile_quality doit être entre 30 et 100")
        return AppConfig(
            raster_path=raster_path,
            output_gpkg=output_gpkg,
            existing_annotations=existing,
            reference_dir=reference_dir,
            tile_size_m=float(self.tile_size_m),
            overlap_m=float(self.overlap_m),
            cache_size=int(self.cache_size),
            panel_position=self.panel_position,
            host=self.host,
            port=int(self.port),
            tile_format=self.tile_format,
            tile_quality=int(self.tile_quality),
        )
