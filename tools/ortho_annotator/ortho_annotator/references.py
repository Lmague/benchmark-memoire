"""Images de référence par espèce (LECTURE SEULE, aucun réseau à l'exécution).

Deux catégories, complémentaires pour l'identification :

- ``aerial`` : vignettes **vue du dessus** extraites des orthomosaïques aux points
  déjà annotés dans ``Annotations.gpkg`` (dossier ``reference_crops/<code>/``).
  MÊME point de vue que l'annotation -> aide directement à différencier les espèces.
- ``ground`` : photos iNaturalist (vue au sol), dans ``reference_images/<Nom_sci>/``
  (+ éventuel dossier utilisateur). Complément affiché dans la galerie.

Les codes d'espèces sont ceux du GeoPackage. La correspondance code -> nom
scientifique (pour le dossier iNaturalist) est surchargée par un fichier optionnel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

_PKG_DIR = Path(__file__).resolve().parent
AERIAL_DIR = _PKG_DIR / "reference_crops"       # vues du dessus (dataset)
GROUND_DIR = _PKG_DIR / "reference_images"      # iNaturalist (vue au sol)

# Emprise au sol des vignettes aériennes, en mètres. Enregistrée dans
# ``reference_crops/meta.json`` par ``rebuild-crops`` : c'est ce qui permet à
# l'interface d'afficher les références **à la même échelle** que la tuile,
# seule façon de comparer utilement une forme vue du dessus.
AERIAL_CROP_M_DEFAULT = 1.2
_META_NAME = "meta.json"


def aerial_crop_m(root: Optional[Path] = None) -> float:
    root = Path(root) if root else AERIAL_DIR
    meta = root / _META_NAME
    if meta.is_file():
        try:
            return float(json.loads(meta.read_text(encoding="utf-8"))["crop_m"])
        except Exception:
            pass
    return AERIAL_CROP_M_DEFAULT

DEFAULT_SCIENTIFIC_NAMES: Dict[str, str] = {
    "Lotcorn": "Lotus_corniculatus",
    "Leuvul": "Leucanthemum_vulgare",
    "Ascsyr": "Asclepias_syriaca",
    "Daucar": "Daucus_carota",
    "Eumac": "Eutrochium_maculatum",
    "Solcan": "Solidago_canadensis",
}


@dataclass
class SpeciesReference:
    code: str
    scientific_name: str
    aerial: List[Path] = field(default_factory=list)
    ground: List[Path] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.aerial or self.ground)


class ReferenceLibrary:
    def __init__(
        self,
        ground_roots: Sequence[Optional[Path]],
        species: List[str],
        mapping: Optional[Dict[str, str]] = None,
        aerial_root: Optional[Path] = None,
    ):
        self.aerial_root = Path(aerial_root).resolve() if aerial_root else AERIAL_DIR
        self.ground_roots: List[Path] = []
        for r in ground_roots:
            if r is None:
                continue
            rp = Path(r).resolve()
            if rp.is_dir() and rp not in self.ground_roots:
                self.ground_roots.append(rp)
        self.species = list(species)
        self.mapping = dict(DEFAULT_SCIENTIFIC_NAMES)
        if mapping:
            self.mapping.update(mapping)
        self._refs: Dict[str, SpeciesReference] = {}
        self._scan()

    @staticmethod
    def default_roots(user_reference_dir: Optional[Path]) -> List[Optional[Path]]:
        return [GROUND_DIR, user_reference_dir]

    @staticmethod
    def load_mapping_file(path: Optional[Path]) -> Optional[Dict[str, str]]:
        if path is None:
            return None
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items()}

    @staticmethod
    def _images_in(folder: Path) -> List[Path]:
        if not folder.is_dir():
            return []
        return [p for p in sorted(folder.rglob("*"))
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]

    def _scan(self) -> None:
        for code in self.species:
            sci = self.mapping.get(code, code)
            aerial = self._images_in(self.aerial_root / code)
            ground: List[Path] = []
            for root in self.ground_roots:
                ground.extend(self._images_in(root / sci))
                ground.extend(self._images_in(root / code))
                # fichiers à la racine commençant par le nom scientifique/code
                if root.is_dir():
                    for p in sorted(root.iterdir()):
                        if (p.is_file() and p.suffix.lower() in _IMAGE_EXTS
                                and (p.stem.lower().startswith(sci.lower())
                                     or p.stem.lower().startswith(code.lower()))):
                            ground.append(p)
            # dédoublonnage
            ground = list(dict.fromkeys(str(p) for p in ground))
            ground = [Path(s) for s in ground]
            self._refs[code] = SpeciesReference(code, sci, aerial, ground)

    @property
    def crop_m(self) -> float:
        return aerial_crop_m(self.aerial_root)

    def as_dict(self) -> Dict[str, Dict]:
        return {
            code: {
                "code": ref.code,
                "scientific_name": ref.scientific_name,
                "available": ref.available,
                "aerial_count": len(ref.aerial),
                "ground_count": len(ref.ground),
            }
            for code, ref in self._refs.items()
        }

    def resolve_image(self, code: str, kind: str, idx: int) -> Optional[Path]:
        ref = self._refs.get(code)
        if ref is None:
            return None
        images = ref.aerial if kind == "aerial" else ref.ground
        if idx < 0 or idx >= len(images):
            return None
        candidate = images[idx].resolve()
        roots = [self.aerial_root] if kind == "aerial" else self.ground_roots
        for root in roots:
            try:
                candidate.relative_to(root)
                return candidate if candidate.is_file() else None
            except ValueError:
                continue
        return None
