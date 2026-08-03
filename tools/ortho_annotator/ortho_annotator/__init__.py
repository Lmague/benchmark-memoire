"""Outil local d'annotation ponctuelle sur orthomosaïques géoréférencées.

Le package fournit :
- `raster` : lecture fenêtrée (rasterio.windows.Window) et grille de tuiles en mètres ;
- `geopackage` : persistance incrémentale dans UN GeoPackage de sortie multi-couches ;
- `overlay` : lecture seule du jeu d'annotations existant, indexé en mémoire ;
- `references` : découverte des images de référence par espèce ;
- `server` : application FastAPI + frontend statique ;
- `cli` : points d'entrée `serve`, `validate`, `export-flat`, `prospect`, `rebuild-crops`.

Prospection (aide à la recherche de plantes, sans entraînement) :
- `colormodel` : rapport de vraisemblance couleur appris sur les points annotés ;
- `embed` : encodeur figé (DINOv3) et prototypes de vignettes ;
- `dense` : appariement de jetons de patch sur une fenêtre entière ;
- `calibrate` : mesure du rappel et de la précision, seuils choisis sur données ;
- `prospect` : balayage par blocs et magasin de candidats.

Aucune fonction ne charge un raster entier en mémoire : toute lecture pixel passe
par une fenêtre bornée à la tuile affichée ou au bloc traité.
"""

__all__ = ["config", "raster", "geopackage", "overlay", "references", "server",
           "cli", "colormodel", "embed", "dense", "calibrate", "prospect"]
