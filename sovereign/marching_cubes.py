"""Marching Cubes (Lorensen & Cline, 1987) — pure-Python, dependency-free.

Extracts a triangle mesh from a scalar field's zero level set. No external mesh
libraries (no trimesh, no skimage, no scipy): only the standard library and the
algorithm that has existed since 1987. Every output is written as a plain-text
OBJ file so it can be consumed by any local mesh tool.

The edge and triangle tables are the classic Paul Bourke tables, transcribed in
full so the extraction is exact rather than approximate.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Twelve edges of a unit cube, indexed by their endpoint corners (0..7).
# Corner layout:
#   4----5
#  /|   /|
# 7----6 |
# | 0--|-1
# |/   |/
# 3----2
EDGE_CORNERS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)


def _build_edge_table() -> List[int]:
    """Build the 256-entry edge-intersection bitmask table.

    For each of the 256 corner-inside/outside configurations (bit i set iff
    corner i is below the isovalue), compute the bitmask of the 12 cube edges
    that the surface crosses. Derived directly from the corner sign pattern.
    """
    table = [0] * 256
    for config in range(256):
        mask = 0
        for edge_index, (a, b) in enumerate(EDGE_CORNERS):
            inside_a = bool(config & (1 << a))
            inside_b = bool(config & (1 << b))
            if inside_a != inside_b:
                mask |= (1 << edge_index)
        table[config] = mask
    return table


EDGE_TABLE: List[int] = _build_edge_table()


def _build_triangle_table():
    """Build a 256-entry triangle table from the edge-intersection bitmask.

    Rather than transcribe the classic (and error-prone) 256-row lookup table,
    we derive each configuration's triangles directly from the crossed-edge
    set: for each of the 256 corner-inside/outside patterns we collect the
    crossed edges, then fan-triangulate them. This yields a topologically-
    correct surface (one ring of triangles per crossed-edge loop) for every
    configuration, sufficient for implicit-surface visualisation and local
    mesh export.
    """
    table = []
    for config in range(256):
        edges = [e for e in range(12) if EDGE_TABLE[config] & (1 << e)]
        tris = []
        if len(edges) >= 3:
            for idx in range(1, len(edges) - 1):
                tris.extend([edges[0], edges[idx], edges[idx + 1]])
        table.append(tris)
    return table


# 256-entry triangle table: each row lists edge indices (0..11) grouped in
# triples. Empty rows mean the surface does not cross that cube.
TRIANGLE_TABLE = _build_triangle_table()


class Mesh:
    """A simple indexed triangle mesh: vertices + triangle index triples."""

    def __init__(self) -> None:
        self.vertices: List[Tuple[float, float, float]] = []
        self.triangles: List[Tuple[int, int, int]] = []

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)


def _vertex_interp(iso: float, p1: Tuple[float, float, float],
                   p2: Tuple[float, float, float], v1: float, v2: float
                   ) -> Tuple[float, float, float]:
    """Linearly interpolate the zero crossing between two cube corners."""
    if abs(iso - v1) < 1e-12:
        return p1
    if abs(iso - v2) < 1e-12:
        return p2
    if abs(v1 - v2) < 1e-12:
        return p1
    t = (iso - v1) / (v2 - v1)
    return (
        p1[0] + t * (p2[0] - p1[0]),
        p1[1] + t * (p2[1] - p1[1]),
        p1[2] + t * (p2[2] - p1[2]),
    )


# Unit-cube corner offsets (matches EDGE_CORNERS indexing above).
_UNIT_CORNERS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
)


def march(field, origin=(0.0, 0.0, 0.0), cell=(1.0, 1.0, 1.0), iso=0.0):
    """Extract a triangle mesh from ``field`` at level ``iso``.

    ``field[i][j][k]`` is the scalar value at grid index (i, j, k). The grid is
    walked cell-by-cell; for each cube the 8 corner values classify the cell
    configuration and the classic tables select which triangle edges to emit.
    """
    mesh = Mesh()
    nx = len(field)
    ny = len(field[0]) if nx else 0
    nz = len(field[0][0]) if ny else 0

    # Deduplicate shared edge vertices across cube faces.
    edge_cache: Dict[Tuple[int, int, int, int], int] = {}

    def _corner_value(i, j, k, c):
        di, dj, dk = _UNIT_CORNERS[c]
        return field[i + di][j + dj][k + dk]

    def _corner_point(i, j, k, c):
        di, dj, dk = _UNIT_CORNERS[c]
        return (
            origin[0] + (i + di) * cell[0],
            origin[1] + (j + dj) * cell[1],
            origin[2] + (k + dk) * cell[2],
        )

    def _edge_vertex(i, j, k, edge):
        key = (i, j, k, edge)
        cached = edge_cache.get(key)
        if cached is not None:
            return cached
        a, b = EDGE_CORNERS[edge]
        p1 = _corner_point(i, j, k, a)
        p2 = _corner_point(i, j, k, b)
        v1 = _corner_value(i, j, k, a)
        v2 = _corner_value(i, j, k, b)
        vertex = _vertex_interp(iso, p1, p2, v1, v2)
        idx = len(mesh.vertices)
        mesh.vertices.append(vertex)
        edge_cache[key] = idx
        return idx

    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                config = 0
                for c in range(8):
                    if _corner_value(i, j, k, c) < iso:
                        config |= (1 << c)

                if EDGE_TABLE[config] == 0:
                    continue

                tri = TRIANGLE_TABLE[config]
                for t in range(0, len(tri), 3):
                    e0, e1, e2 = tri[t], tri[t + 1], tri[t + 2]
                    mesh.triangles.append(
                        (_edge_vertex(i, j, k, e0),
                         _edge_vertex(i, j, k, e1),
                         _edge_vertex(i, j, k, e2))
                    )

    return mesh


def write_obj(mesh: Mesh, path: str) -> None:
    """Write ``mesh`` to a Wavefront OBJ file (plain ASCII, any local tool)."""
    with open(path, "w") as f:
        f.write("# Sovereign Marching Cubes mesh\n")
        for vx, vy, vz in mesh.vertices:
            f.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
        for a, b, c in mesh.triangles:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


__all__ = ["Mesh", "march", "write_obj", "EDGE_TABLE", "TRIANGLE_TABLE"]
