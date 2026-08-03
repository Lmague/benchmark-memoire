"""Re-classement zero-shot par plus proche prototype dans un espace figé (DINOv3).

Aucun entraînement, aucun réglage fin : on se sert des **10 757 points déjà
annotés** comme banque d'exemples. Pour chaque espèce on encode des vignettes
centrées sur ces points avec un encodeur figé, on garde les vecteurs (L2-normés),
et on note un candidat par

    score(c) = max_i cos(e_c, e_i^espèce) - max_j cos(e_c, e_j^fond)

c'est-à-dire « ressemble davantage à cette espèce qu'à n'importe quoi d'autre du
terrain ». Les prototypes de fond sont tirés au hasard sur l'orthomosaïque : ils
capturent l'herbe, la terre, l'ombre, le feuillage, ce qui rend le score bien plus
discriminant qu'une simple similarité à la classe.

Contraintes respectées : **hors ligne** (poids déjà présents dans le cache
Hugging Face local), **CPU** (mesuré : ~19 vignettes/s en ViT-S/16 sur 4 threads),
et **lecture fenêtrée** uniquement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ordre de préférence : le plus petit modèle suffit et tient sur cette machine.
DEFAULT_MODELS = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")
_INPUT_PX = 224


def available_models() -> List[str]:
    """Modèles réellement présents dans le cache local (aucun téléchargement)."""
    root = Path.home() / ".cache" / "huggingface" / "hub"
    out = []
    for mid in DEFAULT_MODELS:
        folder = root / ("models--" + mid.replace("/", "--"))
        if folder.is_dir():
            out.append(mid)
    return out


class Embedder:
    """Encodeur figé chargé depuis le cache local (CPU) ou hors ligne (GPU si dispo).

    ``device=None`` (défaut) : CUDA si disponible, sinon CPU — inchangé sur la
    machine d'annotation locale (pas de GPU), utile sur Colab.
    """

    def __init__(self, model_id: Optional[str] = None, threads: int = 4,
                 device: Optional[str] = None):
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModel

        if model_id is None:
            found = available_models()
            if not found:
                raise RuntimeError(
                    "Aucun encodeur DINOv3 dans le cache Hugging Face local. "
                    "Le re-classement par embedding est indisponible (l'heuristique "
                    "couleur, elle, fonctionne sans modèle)."
                )
            model_id = found[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(max(1, int(threads)))
        self.torch = torch
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)

    def _preprocess(self, crops: Sequence[np.ndarray]):
        from PIL import Image

        batch = np.empty((len(crops), _INPUT_PX, _INPUT_PX, 3), dtype="float32")
        for i, c in enumerate(crops):
            img = Image.fromarray(np.ascontiguousarray(c[..., :3]))
            if img.size != (_INPUT_PX, _INPUT_PX):
                img = img.resize((_INPUT_PX, _INPUT_PX), Image.BILINEAR)
            batch[i] = np.asarray(img, dtype="float32") / 255.0
        batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
        return self.torch.from_numpy(batch.transpose(0, 3, 1, 2))

    def embed(self, crops: Sequence[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """Vignettes (h, w, 3) uint8 -> matrice (n, d) L2-normée."""
        if not crops:
            return np.zeros((0, self.dim), dtype="float32")
        outs: List[np.ndarray] = []
        with self.torch.inference_mode():
            for i in range(0, len(crops), batch_size):
                x = self._preprocess(crops[i: i + batch_size]).to(self.device)
                o = self.model(pixel_values=x)
                v = o.pooler_output if getattr(o, "pooler_output", None) is not None \
                    else o.last_hidden_state[:, 0]
                outs.append(v.float().cpu().numpy())
        e = np.concatenate(outs, axis=0)
        n = np.linalg.norm(e, axis=1, keepdims=True)
        return (e / np.maximum(n, 1e-8)).astype("float32")

    def dense(self, image: np.ndarray, side: int) -> np.ndarray:
        """Carte de jetons (g, g, d) L2-normée pour une grande vignette carrée.

        Un seul passage avant sur une entrée ``side x side`` donne
        ``(side/16)²`` jetons de patch, soit une carte de descripteurs à la
        résolution d'un patch — bien moins cher que de promener une fenêtre
        glissante sur la même surface (mesuré : 0,77 s pour 784², contre plus
        d'une minute pour l'équivalent en fenêtres de 224²).

        Traite une seule fenêtre (batch=1) : sur GPU, préférer ``dense_batch``
        pour un balayage — batch=1 laisse l'essentiel de la VRAM inutilisée.
        """
        return self.dense_batch([image], side, batch_size=1)[0]

    def dense_batch(self, images: Sequence[np.ndarray], side: int,
                    batch_size: int = 16) -> np.ndarray:
        """``dense()`` pour plusieurs fenêtres en une (ou quelques) passe(s) avant.

        Sur GPU, batcher les fenêtres plutôt que les traiter une par une est ce
        qui utilise réellement la VRAM disponible (un ViT-L à batch=1 sur un L4
        24 Go n'en occupe qu'une fraction). ``batch_size`` : à augmenter tant que
        ça ne sature pas la VRAM (essayer 16 → 32 → 64 sur un L4 pour 784px).
        """
        from PIL import Image

        if not images:
            return np.zeros((0, 0, 0, self.dim), dtype="float32")
        side = int(side) // 16 * 16
        g = side // 16
        outs: List[np.ndarray] = []
        with self.torch.inference_mode():
            for i in range(0, len(images), batch_size):
                chunk = images[i: i + batch_size]
                batch = np.empty((len(chunk), side, side, 3), dtype="float32")
                for j, image in enumerate(chunk):
                    img = Image.fromarray(np.ascontiguousarray(image[..., :3]))
                    if img.size != (side, side):
                        img = img.resize((side, side), Image.BILINEAR)
                    batch[j] = np.asarray(img, dtype="float32") / 255.0
                batch = (batch - _IMAGENET_MEAN) / _IMAGENET_STD
                x = self.torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(self.device)
                out = self.model(pixel_values=x).last_hidden_state  # (b, seq, d)
                # last_hidden_state = [CLS] + registres + jetons de patch, par image.
                patches = out[:, -(g * g):, :].float().cpu().numpy()
                outs.append(patches.reshape(len(chunk), g, g, -1))
        e = np.concatenate(outs, axis=0)
        n = np.linalg.norm(e, axis=-1, keepdims=True)
        return (e / np.maximum(n, 1e-8)).astype("float32")


# ---------------------------------------------------------------------------
# Banque de prototypes
# ---------------------------------------------------------------------------


def _crops_at(tiler, xs: np.ndarray, ys: np.ndarray, crop_px: int,
              max_n: int, rng: np.random.Generator) -> List[np.ndarray]:
    keep = np.array([tiler.contains_utm(x, y) for x, y in zip(xs, ys)], dtype=bool) \
        if xs.size else np.zeros(0, dtype=bool)
    xs, ys = xs[keep], ys[keep]
    if xs.size > max_n:
        sel = rng.choice(xs.size, size=max_n, replace=False)
        xs, ys = xs[sel], ys[sel]
    crops = []
    for x, y in zip(xs, ys):
        c = tiler.read_centered(x, y, crop_px, out_size=_INPUT_PX)
        if c.size:
            crops.append(c)
    return crops


class PrototypeBank:
    """Vecteurs de référence par espèce + vecteurs de fond."""

    def __init__(self, model_id: str, crop_m: float,
                 species: Dict[str, np.ndarray], background: np.ndarray):
        self.model_id = model_id
        self.crop_m = float(crop_m)
        self.species = species
        self.background = background

    # ---- construction ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        tilers,
        embedder: Embedder,
        points_by_species: Dict[str, Tuple[np.ndarray, np.ndarray]],
        crop_m: float = 0.30,
        max_per_species: int = 150,
        n_background: int = 250,
        seed: int = 0,
        log=print,
    ) -> "PrototypeBank":
        """Encode les prototypes en mutualisant toutes les orthomosaïques.

        Même raison que pour le modèle couleur : les espèces ne sont pas
        annotées sur les mêmes vols. La taille de vignette est fixée en **mètres**,
        donc convertie en pixels selon la GSD propre à chaque orthomosaïque.
        """
        tilers = list(tilers)
        rng = np.random.default_rng(seed)

        species: Dict[str, np.ndarray] = {}
        for code, (xs, ys) in points_by_species.items():
            crops: List[np.ndarray] = []
            per_raster = max(20, max_per_species // max(1, len(tilers)))
            for t in tilers:
                crop_px = max(16, int(round(crop_m / t.res_x)))
                crops.extend(_crops_at(t, xs, ys, crop_px, per_raster, rng))
            if len(crops) < 5:
                log(f"  {code:8s} : {len(crops)} vignette(s) -> ignoré")
                continue
            species[code] = embedder.embed(crops)
            log(f"  {code:8s} : {len(crops)} prototypes encodés")

        # Fond : positions aléatoires réparties sur toutes les orthomosaïques,
        # en écartant le remplissage blanc des bords.
        bg_crops: List[np.ndarray] = []
        per_raster_bg = max(20, n_background // max(1, len(tilers)))
        for t in tilers:
            crop_px = max(16, int(round(crop_m / t.res_x)))
            got, tries = 0, 0
            while got < per_raster_bg and tries < per_raster_bg * 6:
                tries += 1
                col = int(rng.integers(0, max(1, t.width - crop_px)))
                row = int(rng.integers(0, max(1, t.height - crop_px)))
                c = t.read_window(col, row, crop_px, crop_px,
                                  out_shape=(_INPUT_PX, _INPUT_PX))
                if c.size == 0:
                    continue
                if float(np.all(c[..., :3] >= 248, axis=2).mean()) > 0.3:
                    continue
                bg_crops.append(c)
                got += 1
        background = embedder.embed(bg_crops)
        log(f"  fond     : {len(bg_crops)} prototypes encodés")
        if not species:
            raise RuntimeError("Aucune espèce n'a assez de points annotés.")
        return cls(embedder.model_id, crop_m, species, background)

    # ---- application ----------------------------------------------------------

    def codes(self) -> List[str]:
        return list(self.species.keys())

    def score(self, emb: np.ndarray) -> Dict[str, np.ndarray]:
        """(n, d) -> score discriminant par espèce, dans [-2, 2] en pratique."""
        if emb.size == 0:
            return {c: np.zeros(0, dtype="float32") for c in self.species}
        bg = self.background @ emb.T if self.background.size else None
        bg_max = bg.max(axis=0) if bg is not None and bg.size else np.zeros(emb.shape[0],
                                                                           dtype="float32")
        out = {}
        for code, mat in self.species.items():
            sim = (mat @ emb.T).max(axis=0)
            out[code] = (sim - bg_max).astype("float32")
        return out

    def best(self, emb: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """Meilleure espèce et son score pour chaque vecteur."""
        scores = self.score(emb)
        codes = list(scores.keys())
        mat = np.stack([scores[c] for c in codes]) if codes else np.zeros((0, emb.shape[0]))
        if mat.size == 0:
            return [], np.zeros(0, dtype="float32")
        arg = mat.argmax(axis=0)
        return [codes[i] for i in arg], mat.max(axis=0)

    # ---- persistance ----------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"sp__{c}": v for c, v in self.species.items()}
        np.savez_compressed(
            path,
            meta=np.array([json.dumps({"model_id": self.model_id,
                                       "crop_m": self.crop_m,
                                       "codes": list(self.species.keys())})]),
            background=self.background,
            **arrays,
        )

    @classmethod
    def load(cls, path: Path) -> "PrototypeBank":
        d = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(d["meta"][0]))
        species = {c: d[f"sp__{c}"] for c in meta["codes"]}
        return cls(meta["model_id"], meta["crop_m"], species, d["background"])

    def summary(self) -> Dict:
        return {
            "model_id": self.model_id,
            "crop_m": self.crop_m,
            "n_background": int(self.background.shape[0]),
            "species": {c: int(v.shape[0]) for c, v in self.species.items()},
        }
