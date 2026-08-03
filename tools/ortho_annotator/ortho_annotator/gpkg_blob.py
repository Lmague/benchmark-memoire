"""Encodage/décodage du format de géométrie GeoPackage (points et polygones 2D).

En-tête vérifié (créé par OGR/Fiona) :

  [ 'G' 'P' ][version u8][flags u8][srs_id i32]  puis WKB standard.

Avec ``flags = 0x01`` : little-endian, sans enveloppe (indicateur = 0), en-tête de
8 octets. On écrit les points et des polygones simples (anneau extérieur, sans
enveloppe). En lecture on gère l'ordre d'octets et une éventuelle enveloppe.
"""

from __future__ import annotations

import struct
from typing import List, Tuple

_MAGIC = b"GP"
_ENVELOPE_LEN_BY_INDICATOR = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def encode_point(x: float, y: float, srs_id: int) -> bytes:
    header = _MAGIC + struct.pack("<BBi", 0, 0x01, int(srs_id))
    wkb = struct.pack("<BIdd", 1, 1, float(x), float(y))
    return header + wkb


def encode_polygon(ring: List[Tuple[float, float]], srs_id: int) -> bytes:
    """Encode un polygone simple (un anneau extérieur fermé)."""
    pts = list(ring)
    if pts[0] != pts[-1]:
        pts.append(pts[0])  # fermer l'anneau
    header = _MAGIC + struct.pack("<BBi", 0, 0x01, int(srs_id))
    body = struct.pack("<BII", 1, 3, 1)  # LE, type=3 (Polygon), 1 anneau
    body += struct.pack("<I", len(pts))
    for x, y in pts:
        body += struct.pack("<dd", float(x), float(y))
    return header + body


def encode_bbox_polygon(minx: float, miny: float, maxx: float, maxy: float, srs_id: int) -> bytes:
    ring = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)]
    return encode_polygon(ring, srs_id)


def _wkb_offset(blob: bytes) -> int:
    if blob[:2] != _MAGIC:
        raise ValueError("blob GeoPackage invalide (magic 'GP' absent)")
    flags = blob[3]
    envelope_indicator = (flags >> 1) & 0x07
    if envelope_indicator not in _ENVELOPE_LEN_BY_INDICATOR:
        raise ValueError(f"indicateur d'enveloppe invalide: {envelope_indicator}")
    return 8 + _ENVELOPE_LEN_BY_INDICATOR[envelope_indicator]


def decode_coords(blob: bytes) -> Tuple[int, List[Tuple[float, float]]]:
    """Renvoie ``(geom_type_base, coords)``. Point -> 1 coord ; Polygon -> anneau ext."""
    off = _wkb_offset(blob)
    wkb = blob[off:]
    endian = "<" if wkb[0] == 1 else ">"
    (geom_type,) = struct.unpack_from(endian + "I", wkb, 1)
    base = geom_type & 0xFF
    pos = 5
    if base == 1:  # Point
        x, y = struct.unpack_from(endian + "dd", wkb, pos)
        return 1, [(float(x), float(y))]
    if base == 3:  # Polygon
        (n_rings,) = struct.unpack_from(endian + "I", wkb, pos)
        pos += 4
        coords: List[Tuple[float, float]] = []
        if n_rings >= 1:
            (n_pts,) = struct.unpack_from(endian + "I", wkb, pos)
            pos += 4
            for _ in range(n_pts):
                x, y = struct.unpack_from(endian + "dd", wkb, pos)
                pos += 16
                coords.append((float(x), float(y)))
        return 3, coords
    raise ValueError(f"type WKB non géré : {geom_type}")


def decode_point(blob: bytes) -> Tuple[float, float]:
    base, coords = decode_coords(blob)
    if base != 1 or not coords:
        raise ValueError("géométrie non ponctuelle")
    return coords[0]


def decode_bbox(blob: bytes) -> Tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) de la géométrie."""
    _, coords = decode_coords(blob)
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)
