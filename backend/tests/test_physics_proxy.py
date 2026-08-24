import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from backend.segmentation import physics_proxy


def _write_gaussian_ply(path: Path, positions: np.ndarray, scale: float = 0.02) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ("opacity", "f4"),
    ]
    vertices = np.zeros(len(positions), dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    for name in ("scale_0", "scale_1", "scale_2"):
        vertices[name] = np.log(scale)
    vertices["rot_0"] = 1.0
    vertices["opacity"] = 5.0
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def _assert_closed(vertices: np.ndarray, faces: np.ndarray) -> None:
    assert len(vertices) >= 4
    assert len(faces) >= 4
    assert physics_proxy._watertight(faces)


def test_obb_cylinder_and_convex_hull_are_closed():
    rng = np.random.default_rng(4)
    box = rng.uniform([-1.0, -0.3, -0.5], [1.0, 0.3, 0.5], size=(500, 3))
    vertices, faces, geometry = physics_proxy._build_obb(box)
    _assert_closed(vertices, faces)
    assert len(geometry["half_extents"]) == 3

    angles = rng.uniform(0, 2 * np.pi, 800)
    radii = np.sqrt(rng.uniform(0, 1, 800)) * 0.25
    cylinder = np.column_stack([rng.uniform(-1, 1, 800), np.cos(angles) * radii, np.sin(angles) * radii])
    vertices, faces, geometry = physics_proxy._build_cylinder(cylinder)
    _assert_closed(vertices, faces)
    assert geometry["height"] > geometry["radius"]

    vertices, faces, geometry = physics_proxy._build_convex_hull(box)
    _assert_closed(vertices, faces)
    assert geometry["volume"] > 0


def test_support_plane_selects_upper_horizontal_surface():
    rng = np.random.default_rng(2)
    top = np.column_stack(
        [rng.uniform(-1, 1, 900), rng.normal(0.8, 0.002, 900), rng.uniform(-0.6, 0.6, 900)]
    )
    bottom = np.column_stack(
        [rng.uniform(-1, 1, 400), rng.normal(0.7, 0.002, 400), rng.uniform(-0.6, 0.6, 400)]
    )
    vertices, faces, geometry = physics_proxy._build_support_plane(
        np.vstack([top, bottom]), np.array([0.0, 1.0, 0.0])
    )
    _assert_closed(vertices, faces)
    assert geometry["plane_origin"][1] == pytest.approx(0.8, abs=0.02)
    assert geometry["plane_normal"][1] > 0.95


def test_generate_physics_proxy_writes_proxy_and_report(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    positions = rng.uniform([-0.2, -0.5, -0.2], [0.2, 0.5, 0.2], size=(300, 3))
    source = tmp_path / "scene.ply"
    _write_gaussian_ply(source, positions)
    directory = tmp_path / "layer"
    directory.mkdir()
    indices = directory / "gaussian-indices-source-0.u32"
    np.arange(len(positions), dtype="<u4").tofile(indices)
    stored = (
        {
            "category": "bottle",
            "gaussian_indices": [{"source_index": 0, "file": indices.name}],
        },
        directory,
    )
    monkeypatch.setattr(physics_proxy, "get_layer_metadata", lambda *_args: stored)
    monkeypatch.setattr(physics_proxy, "get_output_path", lambda *_args: source)

    output, report = physics_proxy.generate_physics_proxy("task", "layer")

    assert output.name == "physics_proxy.ply"
    assert output.is_file()
    assert report["proxy_type"] == "cylinder"
    assert report["watertight"] is True
    assert report["physics_ready"] is True
    saved = json.loads((directory / "physics-proxy-report.json").read_text(encoding="utf-8"))
    assert saved == report
    mesh = PlyData.read(str(output))
    assert len(mesh["vertex"]) == report["vertices"]
    assert len(mesh["face"]) == report["triangles"]
