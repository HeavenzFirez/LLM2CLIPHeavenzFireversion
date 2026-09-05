"""Sovereign pipeline: a dependency-free, fully-local geometry + data organ.

This package implements the "Continuum" payload as real, tested modules:

* :mod:`sovereign.radix9216` — base-9216 radix conversion (log-safe, reversible)
* :mod:`sovereign.marching_cubes` — exact Marching Cubes (full Paul Bourke tables)
* :mod:`sovereign.gyroid` — triply-periodic minimal-surface implicit field
* :mod:`sovereign.pipeline` — CSV generation, validation, averaging, mesh
  extraction, and timestamped immutable logging — all local, no network.

Everything is pure standard library. No paid cloud API, no mesh library, no
metering — aligned with the repo's anti-profit / open-access mission.
"""

from . import radix9216, marching_cubes, gyroid, pipeline  # noqa: F401

__all__ = ["radix9216", "marching_cubes", "gyroid", "pipeline"]
