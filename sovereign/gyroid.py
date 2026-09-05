"""Gyroid implicit surface - a triply-periodic minimal surface (TPMS).

The gyroid is defined implicitly by the equation where the sum of three
sin-cos product terms equals zero. Sampling this scalar field on a regular grid
and feeding it to Marching Cubes yields the characteristic gyroid lattice used
in lightweight-structure and heat-exchanger design. Pure math, no dependencies.
"""

from __future__ import annotations

import math
from typing import List


def gyroid_value(x: float, y: float, z: float) -> float:
    """Evaluate the gyroid implicit function at (x, y, z)."""
    return (
        math.sin(x) * math.cos(y)
        + math.sin(y) * math.cos(z)
        + math.sin(z) * math.cos(x)
    )


def sample_field(resolution: int, scale: float = 2 * math.pi) -> List[List[List[float]]]:
    """Sample the gyroid field on a resolution**3 regular grid.

    Coordinates range over [0, scale] on each axis. Returns a nested list
    field[i][j][k] ready to hand to sovereign.marching_cubes.march.
    """
    if resolution < 2:
        raise ValueError("resolution must be >= 2 to form at least one cell")
    field: List[List[List[float]]] = []
    for i in range(resolution):
        plane: List[List[float]] = []
        x = (i / (resolution - 1)) * scale
        for j in range(resolution):
            row: List[float] = []
            y = (j / (resolution - 1)) * scale
            for k in range(resolution):
                z = (k / (resolution - 1)) * scale
                row.append(gyroid_value(x, y, z))
            plane.append(row)
        field.append(plane)
    return field


def near_surface_points(resolution: int, scale: float = 2 * math.pi,
                        tolerance: float = 0.15):
    """Return (x, y, z, value) tuples where the field is near the surface.

    This is the approximate point-cloud representation from the original payload;
    use sovereign.marching_cubes.march for an exact triangle mesh.
    """
    points = []
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                x = (i / resolution) * scale
                y = (j / resolution) * scale
                z = (k / resolution) * scale
                val = gyroid_value(x, y, z)
                if abs(val) < tolerance:
                    points.append((round(x, 4), round(y, 4), round(z, 4),
                                   round(val, 5)))
    return points


__all__ = ["gyroid_value", "sample_field", "near_surface_points"]
