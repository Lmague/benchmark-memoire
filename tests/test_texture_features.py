"""Tests des descripteurs de texture (src/texture.py) — images synthétiques, aucune donnée.

Ces tests protègent trois invariants qui, s'ils cassent, produisent des chiffres faux
SANS erreur visible :

1. **Contrat de colonnes** — ``tile_features`` doit produire EXACTEMENT
   ``expected_feature_names``, dans le même ordre, pour toute tuile. Un ordre qui dépend
   du contenu de l'image décalerait silencieusement les colonnes du design matrix entre
   le train et le test, et la sonde apprendrait sur des colonnes mal appariées.
2. **Déterminisme** — deux appels sur la même image donnent le même vecteur (le dépôt
   promet du bit-pour-bit reproductible).
3. **Sens physique** — les features doivent aller dans le bon sens sur des cas limites
   construits (uniforme vs damier vs bruit) : contraste, entropie, taille des zones,
   longueur des segments, rugosité spectrale.

Compatible pytest (fonctions ``test_*``) et exécution directe :
    python3 tests/test_texture_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.texture import (  # noqa: E402
    FAMILIES,
    expected_feature_names,
    family_of,
    quantize,
    tile_features,
    to_gray,
)

# ── images de référence (64×64 : petit mais assez grand pour 3 niveaux de DWT et la PSD)
_UNIFORM = np.full((64, 64), 128, dtype=np.uint8)
_CHECKS = (((np.indices((64, 64)).sum(0) // 2) % 2) * 255).astype(np.uint8)
_STRIPES = ((np.indices((64, 64))[0] // 4) % 2 * 255).astype(np.uint8)
_rng = np.random.RandomState(42)
_NOISE = (_rng.rand(64, 64) * 255).astype(np.uint8)


def test_expected_names_shape_and_families():
    """Le contrat couvre les 7 familles et sa taille est celle des features produites."""
    names = expected_feature_names()
    fams = {family_of(n) for n in names}
    assert fams == set(FAMILIES), f"familles du contrat {fams} != {set(FAMILIES)}"
    assert len(names) == len(set(names)), "noms de features dupliqués dans le contrat"


def test_column_contract_is_respected_on_every_image():
    """Invariant n°1 : mêmes noms, même ordre, quelle que soit l'image."""
    names = list(expected_feature_names())
    for img in (_UNIFORM, _CHECKS, _STRIPES, _NOISE):
        assert list(tile_features(img).keys()) == names


def test_determinism():
    """Invariant n°2 : deux appels identiques → vecteurs identiques."""
    a = np.array(list(tile_features(_NOISE).values()))
    b = np.array(list(tile_features(_NOISE).values()))
    assert np.array_equal(a, b), "features non déterministes"


def test_all_values_finite():
    for img in (_UNIFORM, _CHECKS, _STRIPES, _NOISE):
        vals = np.array(list(tile_features(img).values()))
        assert np.isfinite(vals).all(), "valeur non finie dans les features"


def test_quantize_fixed_range_and_levels():
    """La quantification se fait sur une plage FIXE : 0 et 255 restent séparés au max."""
    q = quantize(np.array([[0, 255]], dtype=np.uint8), levels=16)
    assert q.min() == 0 and q.max() == 15, "la plage de quantification n'est pas 0–255"
    # Deux tuiles de luminosité globale différente restent dans les MÊMES bins.
    assert quantize(np.array([[128]], np.uint8), 16)[0, 0] == 8
    assert quantize(np.array([[160]], np.uint8), 16)[0, 0] == 10


def test_to_gray_rgb_and_passthrough():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255                          # rouge pur → luminance BT.601 = 0.299·255
    g = to_gray(rgb)
    assert g.shape == (4, 4) and g.dtype == np.uint8
    assert abs(int(g[0, 0]) - round(0.299 * 255)) <= 1
    gray2d = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert np.array_equal(to_gray(gray2d), gray2d)


def test_glcm_uniform_is_degenerate():
    """Uniforme : une seule transition (i,i) → contraste nul, homogénéité et énergie max."""
    f = tile_features(_UNIFORM)
    assert f["glcm_d1_contrast"] == 0.0
    assert f["glcm_d1_dissimilarity"] == 0.0
    assert abs(f["glcm_d1_homogeneity"] - 1.0) < 1e-9
    assert abs(f["glcm_d1_entropy"]) < 1e-9
    assert abs(f["glcm_d1_energy"] - 1.0) < 1e-9      # ASM = 1 → energy = sqrt(1) = 1
    assert abs(f["glcm_d1_maximum"] - 1.0) < 1e-9


def test_glcm_orders_contrast_and_entropy():
    """Damier > uniforme pour le contraste et l'entropie ; énergie dans l'autre sens."""
    u, c = tile_features(_UNIFORM), tile_features(_CHECKS)
    assert c["glcm_d1_contrast"] > u["glcm_d1_contrast"]
    assert c["glcm_d1_entropy"] > u["glcm_d1_entropy"]
    assert u["glcm_d1_energy"] > c["glcm_d1_energy"]


def test_glcm_distance_matters():
    """Sur des bandes de 4 px, d=1 et d=4 ne voient pas la même chose."""
    f = tile_features(_STRIPES)
    assert f["glcm_d1_contrast"] != f["glcm_d4_contrast"], "les distances GLCM sont ignorées"


def test_glszm_zones_uniform_vs_chessboard():
    """GLSZM mesure la TAILLE des zones : uniforme = 1 grande zone, damier = des milliers
    de petites zones. C'est la propriété qui la rend complémentaire de GLCM."""
    u, c = tile_features(_UNIFORM), tile_features(_CHECKS)
    assert u["glszm_lae"] > c["glszm_lae"], "grandes zones : uniforme devrait gagner"
    assert c["glszm_sae"] > u["glszm_sae"], "petites zones : le damier devrait gagner"
    assert c["glszm_zp"] > u["glszm_zp"], "zone percentage = nb_zones / nb_pixels"


def test_glrlm_long_runs_for_stripes():
    """GLRLM mesure la LONGUEUR des segments : des bandes de 4 px horizontales donnent des
    segments plus longs qu'un damier de 2 px, et l'effet dépend de la direction."""
    s, c = tile_features(_STRIPES), tile_features(_CHECKS)
    assert s["glrlm_lre"] > c["glrlm_lre"], "segments longs : les bandes devraient gagner"
    assert c["glrlm_sre"] > s["glrlm_sre"], "segments courts : le damier devrait gagner"


def test_gldm_dependence_drops_with_noise():
    """GLDM mesure la dépendance locale : le bruit la fait chuter."""
    u, n = tile_features(_UNIFORM), tile_features(_NOISE)
    assert u["gldm_lde"] > n["gldm_lde"], "dépendance : l'uniforme devrait dominer"
    assert u["gldm_sde"] < n["gldm_sde"]


def test_ngtdm_coarseness_ordering():
    """NGTDM : coarseness élevée = voisinage homogène."""
    u, n = tile_features(_UNIFORM), tile_features(_NOISE)
    assert u["ngtdm_coarseness"] > n["ngtdm_coarseness"] > 0.0
    assert u["ngtdm_busyness"] < n["ngtdm_busyness"]


def test_dwt_energy_is_relative_and_high_frequency_sensitive():
    """Les énergies DWT sont relatives (somme ≈ 1) et le damier pousse l'énergie en HF."""
    u, c = tile_features(_UNIFORM), tile_features(_CHECKS)
    names = [n for n in expected_feature_names() if family_of(n) == "dwt"]
    su = sum(u[n] for n in names)
    sc = sum(c[n] for n in names)
    assert abs(su - 1.0) < 1e-6, f"énergies DWT non normalisées (somme {su})"
    assert abs(sc - 1.0) < 1e-6
    hf = [n for n in names if n.endswith(("LH", "HL", "HH")) and "_L1_" in n]
    assert sum(c[n] for n in hf) > sum(u[n] for n in hf), "le damier doit avoir plus d'énergie HF"


def test_fourier_bands_and_angular_profile_normalised():
    f = tile_features(_NOISE)
    names = [n for n in expected_feature_names() if family_of(n) == "fourier"]
    assert all(np.isfinite(f[n]) for n in names)
    bands = [n for n in names if n.startswith("fourier_band")]
    assert abs(sum(f[n] for n in bands) - 1.0) < 1e-6, "profil radial non normalisé"
    angs = [n for n in names if n.startswith("fourier_ang")]
    assert abs(sum(f[n] for n in angs) - 1.0) < 1e-6, "profil angulaire non normalisé"
    assert 0.0 <= f["fourier_anisotropy"] <= 1.0


def test_fourier_beta_orders_smooth_vs_rough():
    """Convention P(r) ∝ r^(−β) : β > 0 pour une image lisse, ≈ 0 pour du bruit blanc."""
    grad = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))   # gradient lisse
    b_smooth = tile_features(grad)["fourier_beta"]
    b_noise = tile_features(_NOISE)["fourier_beta"]
    assert b_smooth > 3.0, f"un gradient lisse doit avoir un β élevé (reçu {b_smooth:.2f})"
    assert b_smooth > b_noise > -1.0, f"β(lissé)={b_smooth:.2f} vs β(bruit)={b_noise:.2f}"
    # Uniforme : image nulle après centrage → cas dégénéré, β = 0 par convention.
    assert tile_features(_UNIFORM)["fourier_beta"] == 0.0


def test_levels_change_feature_set_only_via_quantization():
    """``levels`` paramètre la quantification, pas le nombre de colonnes : le contrat de
    noms doit rester identique (sinon les caches de deux réglages seraient incompatibles)."""
    assert list(tile_features(_NOISE, levels=8).keys()) == list(tile_features(_NOISE, levels=32).keys())
    assert (tile_features(_NOISE, levels=8)["glcm_d1_entropy"]
            != tile_features(_NOISE, levels=32)["glcm_d1_entropy"])


def test_dwt_level_extends_contract_monotonically():
    n2 = len(expected_feature_names(dwt_level=2))
    n3 = len(expected_feature_names(dwt_level=3))
    assert n3 - n2 == 3, "chaque niveau DWT ajoute LH/HL/HH"
    assert len(tile_features(_NOISE, dwt_level=2)) == n2


def test_family_of_prefix():
    assert family_of("glcm_d1_contrast") == "glcm"
    assert family_of("glszm_lae") == "glszm"
    assert family_of("dwt_L1_HH") == "dwt"


def test_missing_data_raises():
    """Une image non 2D doit lever, pas produire silencieusement un vecteur tronqué."""
    try:
        tile_features(np.zeros((8, 8, 3), dtype=np.uint8))
    except ValueError:
        return
    raise AssertionError("tile_features aurait dû refuser une image 3 canaux (utiliser to_gray)")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"OK: {t.__name__}")
    print(f"\n{len(_TESTS)} tests passed.")
