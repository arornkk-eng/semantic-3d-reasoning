import json
from pathlib import Path

import pytest

from backend.segmentation import mesh_generation


def _stored_layer(tmp_path: Path) -> tuple[dict, Path]:
    directory = tmp_path / "layer"
    directory.mkdir()
    indices = directory / "gaussian-indices-source-0.u32"
    indices.write_bytes(b"indices")
    return (
        {
            "gaussian_indices": [
                {
                    "source_index": 0,
                    "file": indices.name,
                }
            ]
        },
        directory,
    )


def test_generate_layer_mesh_writes_visual_collision_and_report(tmp_path, monkeypatch):
    stored = _stored_layer(tmp_path)
    source = tmp_path / "scene.ply"
    source.write_bytes(b"ply")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(mesh_generation, "get_layer_metadata", lambda *_args: stored)
    monkeypatch.setattr(mesh_generation, "get_output_path", lambda *_args: source)
    monkeypatch.setattr(mesh_generation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("OPEN3D_PYTHON", str(python))

    def fake_run(command, **_kwargs):
        visual = Path(command[4])
        collision = Path(command[command.index("--collision-output") + 1])
        visual.write_bytes(b"visual")
        collision.write_bytes(b"collision")
        report = {
            "engine": "open3d-tsdf",
            "geometry_source": "synthetic-depth-from-gaussian-centers",
            "safe_for_collision": False,
            "vertices": 100,
            "triangles": 200,
            "collision_vertices": 30,
            "collision_triangles": 50,
        }
        return type("Completed", (), {"returncode": 0, "stdout": json.dumps(report), "stderr": ""})()

    monkeypatch.setattr(mesh_generation.subprocess, "run", fake_run)

    output, report = mesh_generation.generate_layer_mesh("task", "layer")

    assert output.name == "visual_mesh.ply"
    assert (stored[1] / "collision_mesh.ply").read_bytes() == b"collision"
    saved = json.loads((stored[1] / "mesh-report.json").read_text(encoding="utf-8"))
    assert saved["safe_for_collision"] is False
    assert saved["visual_mesh_file"] == "visual_mesh.ply"
    assert saved["collision_mesh_file"] == "collision_mesh.ply"
    assert report == saved


def test_generate_layer_mesh_rejects_missing_collision_output(tmp_path, monkeypatch):
    stored = _stored_layer(tmp_path)
    source = tmp_path / "scene.ply"
    source.write_bytes(b"ply")
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")
    monkeypatch.setattr(mesh_generation, "get_layer_metadata", lambda *_args: stored)
    monkeypatch.setattr(mesh_generation, "get_output_path", lambda *_args: source)
    monkeypatch.setattr(mesh_generation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("OPEN3D_PYTHON", str(python))

    def fake_run(command, **_kwargs):
        Path(command[4]).write_bytes(b"visual")
        return type("Completed", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()

    monkeypatch.setattr(mesh_generation.subprocess, "run", fake_run)

    with pytest.raises(mesh_generation.LayerMeshError, match="碰撞候选"):
        mesh_generation.generate_layer_mesh("task", "layer")
