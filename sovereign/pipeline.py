"""Sovereign pipeline orchestrator.

Runs the full local workflow with no network:

  1. Generate a synthetic sensor CSV (no external data required).
  2. Validate every cell, capture parse errors, compute per-column averages.
  3. Encode averages in base-9216 (log-safe, reversible).
  4. Sample the gyroid implicit field and extract an exact triangle mesh via
     Marching Cubes; write it as a portable OBJ file.
  5. Write a results CSV and append a timestamped, immutable log line.

All outputs land in a caller-specified workspace directory. Nothing leaves the
machine. This is the "Continuum" payload as real, importable, tested code.
"""

from __future__ import annotations

import csv
import datetime
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import marching_cubes as mc
from . import gyroid as gy
from .radix9216 import to_radix_9216


COLUMNS = ["temp", "pressure", "flow", "vibration"]


def log(msg, log_path=None):
    """Timestamp and print msg; append to log_path if given."""
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    line = "[" + ts + "] " + msg
    print(line)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")
    return line


def generate_synthetic_csv(path, rows=50):
    """Write a deterministic synthetic sensor CSV (no network)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for i in range(rows):
            writer.writerow({
                "temp": 20 + (i % 17) * 0.7,
                "pressure": 1013 + (i % 11) * 2.3,
                "flow": 12.5 + (i % 9) * 0.4,
                "vibration": 0.02 + (i % 13) * 0.003,
            })
    return path


def validate_and_average(csv_path):
    """Read csv_path, validate floats, return (averages, counts, errors)."""
    sums = defaultdict(float)
    counts = defaultdict(int)
    errors = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                for k, v in row.items():
                    val = float(v)
                    sums[k] += val
                    counts[k] += 1
            except Exception as e:
                errors.append(str(e))
    averages = {k: sums[k] / counts[k] for k in sums if counts[k]}
    return averages, counts, errors


def write_results_csv(path, averages, counts):
    """Write per-column average, base-9216 encoding, and count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["column", "average", "radix_9216", "count"])
        for k in averages:
            radix = to_radix_9216(int(round(averages[k] * 1000)))
            writer.writerow([k, averages[k], radix, counts[k]])
    return path


def generate_gyroid_mesh(path, resolution=16, scale=2 * math.pi):
    """Sample the gyroid field and write an exact OBJ mesh via Marching Cubes."""
    field = gy.sample_field(resolution, scale)
    cell = (scale / (resolution - 1),) * 3
    mesh = mc.march(field, origin=(0.0, 0.0, 0.0), cell=cell, iso=0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    mc.write_obj(mesh, str(path))
    return path, mesh


def run(workspace, rows=50, resolution=16):
    """Execute the full Sovereign pipeline inside workspace.

    Returns a summary dict with file paths, averages, mesh stats, and errors.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "workflow.log"

    log("Generating synthetic sensor CSV", log_path)
    csv_path = generate_synthetic_csv(workspace / "data.csv", rows)

    log("Validating and computing averages", log_path)
    averages, counts, errors = validate_and_average(csv_path)
    radix_averages = {k: to_radix_9216(int(round(v * 1000)))
                      for k, v in averages.items()}

    log("Generating gyroid mesh (Marching Cubes, exact)", log_path)
    mesh_path, mesh = generate_gyroid_mesh(
        workspace / "gyroid_mesh.obj", resolution=resolution)

    log("Writing results.csv", log_path)
    write_results_csv(workspace / "results.csv", averages, counts)

    summary = {
        "averages": averages,
        "radix_9216": radix_averages,
        "row_count": dict(counts),
        "mesh_vertices": mesh.vertex_count,
        "mesh_triangles": mesh.triangle_count,
        "errors": errors,
        "files": {
            "data": str(csv_path),
            "results": str(workspace / "results.csv"),
            "gyroid_mesh": str(mesh_path),
            "log": str(log_path),
        },
    }
    log("Pipeline complete. Summary: " + str(summary), log_path)
    return summary


__all__ = [
    "COLUMNS", "log", "generate_synthetic_csv", "validate_and_average",
    "write_results_csv", "generate_gyroid_mesh", "run",
]
