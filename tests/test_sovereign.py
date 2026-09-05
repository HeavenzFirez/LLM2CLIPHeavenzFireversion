"""Tests for the Sovereign pipeline (radix, gyroid, marching cubes, viz).

Pure-stdlib; runs without torch/pytest via the inline runner, matching the
test_advanced_algorithms.py convention.
"""

import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pytest
except ImportError:
    pytest = None

from sovereign import radix9216, gyroid, marching_cubes as mc, pipeline
from sovereign.viz import html_renderer as hr, manifold as mf


# --- radix9216 -----------------------------------------------------------
def test_radix_roundtrip():
    for v in [0, 1, 9215, 9216, 1000000, 84852480, 9216 ** 3]:
        assert radix9216.from_radix_9216(radix9216.to_radix_9216(v)) == v, v


def test_radix_clamps_negative():
    assert radix9216.to_radix_9216(-5) == radix9216.to_radix_9216(0)
    assert radix9216.from_radix_9216("") == 0


def test_radix_rejects_bad_digits():
    for bad in ["x", "9216", "99999"]:
        try:
            radix9216.from_radix_9216(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % bad)


# --- gyroid ---------------------------------------------------------------
def test_gyroid_field_shape_and_values():
    f = gyroid.sample_field(4, scale=1.0)
    assert len(f) == 4 and len(f[0]) == 4 and len(f[0][0]) == 4
    assert all(-3.0 <= v <= 3.0 for plane in f for row in plane for v in row)


def test_gyroid_value_at_origin():
    # sin(0)cos(0) + sin(0)cos(0) + sin(0)cos(0) = 0 (origin is ON the surface).
    assert abs(gyroid.gyroid_value(0, 0, 0) - 0.0) < 1e-9


def test_gyroid_near_surface_returns_points():
    pts = gyroid.near_surface_points(10, tolerance=0.5)
    assert isinstance(pts, list)
    assert all(len(p) == 4 for p in pts)


# --- marching cubes -------------------------------------------------------
def test_tables_are_256_entries():
    assert len(mc.EDGE_TABLE) == 256
    assert len(mc.TRIANGLE_TABLE) == 256


def test_march_empty_field_yields_no_triangles():
    # A constant positive field never crosses the zero level set.
    field = [[[1.0 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    mesh = mc.march(field, iso=0.0)
    assert mesh.triangle_count == 0
    assert mesh.vertex_count == 0


def test_march_sphere_produces_triangles():
    # A sphere field crosses zero -> expect a closed-ish mesh.
    n = 8
    field = [[[(i - n / 2) ** 2 + (j - n / 2) ** 2 + (k - n / 2) ** 2 - 4
               for k in range(n)] for j in range(n)] for i in range(n)]
    mesh = mc.march(field, iso=0.0)
    assert mesh.triangle_count > 0
    assert mesh.vertex_count > 0


def test_write_obj_roundtrip(tmp_path=None):
    field = [[[(i - 2) ** 2 + (j - 2) ** 2 + (k - 2) ** 2 - 4
               for k in range(6)] for j in range(6)] for i in range(6)]
    mesh = mc.march(field, iso=0.0)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "m.obj")
    mc.write_obj(mesh, path)
    txt = open(path).read()
    assert txt.startswith("# Sovereign")
    assert "v " in txt and "f " in txt


# --- pipeline -------------------------------------------------------------
def test_pipeline_run_end_to_end():
    d = tempfile.mkdtemp()
    summary = pipeline.run(d, rows=20, resolution=8)
    assert summary["errors"] == []
    assert set(summary["averages"]) == {"temp", "pressure", "flow", "vibration"}
    assert summary["mesh_triangles"] > 0
    for key in ("data", "results", "gyroid_mesh", "log"):
        assert os.path.exists(summary["files"][key]), key
    # OBJ file written
    obj = open(summary["files"]["gyroid_mesh"]).read()
    assert "v " in obj and "f " in obj


# --- viz ------------------------------------------------------------------
def test_gyroid_payload_shape():
    mesh = mc.Mesh()
    mesh.vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    mesh.triangles = [(0, 1, 2)]
    p = hr.build_gyroid_payload(mesh, {"x": 1.5}, {"x": "1.7056"})
    assert p["kind"] == "gyroid"
    assert len(p["vertices"]) == 3 and len(p["triangles"]) == 1
    assert p["averages"] == {"x": 1.5}


def test_tesseract_payload():
    p = hr.build_tesseract_payload()
    assert p["kind"] == "tesseract"
    assert len(p["vertices"]) == 16
    assert len(p["edges"]) == 32  # 4D hypercube has 32 edges


def test_render_html_is_document():
    p = hr.build_tesseract_payload()
    html = hr.render_html(p)
    assert html.startswith("<!DOCTYPE html>")
    assert "tesseract" in html


def test_manifold_payloads_carry_pipeline_data():
    nodes = mf.build_node_payloads(
        {"temp": 25.0, "pressure": 1013.0},
        {"vertices": 100, "triangles": 50}, resolution=8)
    assert len(nodes) == 16
    assert all("Radix averages" in n["payload"] for n in nodes)
    assert all(n["id"].startswith("NODE_") for n in nodes)


def test_manifold_html_contains_nodes():
    nodes = mf.build_node_payloads(
        {"temp": 25.0}, {"vertices": 10, "triangles": 5}, resolution=6)
    d = tempfile.mkdtemp()
    path = mf.write_manifold_html(nodes, os.path.join(d, "m.html"))
    html = open(path).read()
    assert "NODE_00" in html
    assert "Radix averages" in html


# --- runner ---------------------------------------------------------------
def _approx(expected, rel=1e-6):
    if pytest is not None:
        return pytest.approx(expected, rel=rel)
    class _A:
        def __init__(s, e, r): s.e, s.r = e, r
        def __eq__(s, o): return math.isclose(float(o), float(s.e), rel_tol=s.r)
    return _A(expected, rel)


if __name__ == "__main__":
    module = sys.modules[__name__]
    funcs = [v for k, v in sorted(vars(module).items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print("PASS " + fn.__name__)
        except Exception as exc:
            failed += 1
            print("FAIL " + fn.__name__ + ": " + repr(exc))
    print("\n%d/%d passed" % (len(funcs) - failed, len(funcs)))
    sys.exit(1 if failed else 0)
