import json

from backend.core.schemas import SceneSnapshotCreateRequest
from backend.storage import layer_store, scene_snapshot_store


def _request(layer_id: str = "layer1") -> SceneSnapshotCreateRequest:
    return SceneSnapshotCreateRequest.model_validate(
        {
            "camera": {
                "view_matrix": [0.0] * 16,
                "projection_matrix": [0.0] * 16,
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0],
                "focal_point": [0.0, 0.0, 0.0],
                "azim": 0.0,
                "elevation": 0.0,
                "distance": 1.0,
                "fov": 75.0,
            },
            "objects": [
                {
                    "layer_id": layer_id,
                    "name": "瓶子1",
                    "category": "bottle",
                    "center_camera": [0.0, 0.0, -2.0],
                    "bounds_min_camera": [-0.1, -0.2, -2.1],
                    "bounds_max_camera": [0.1, 0.2, -1.9],
                }
            ],
            "relations": [],
            "functions": {"bottle": ["盛装液体"]},
            "description": ["瓶子通常用于盛装液体。"],
        }
    )


def test_snapshot_auto_name_rename_and_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr(scene_snapshot_store, "LAYER_DIR", tmp_path)
    first = scene_snapshot_store.create_snapshot("task1", _request())
    second = scene_snapshot_store.create_snapshot("task1", _request())

    assert first["name"] == "视角分析1"
    assert second["name"] == "视角分析2"
    renamed = scene_snapshot_store.rename_snapshot("task1", first["snapshot_id"], "入口视角")
    assert renamed["name"] == "入口视角"
    scene_snapshot_store.delete_snapshot("task1", second["snapshot_id"])
    assert [item["name"] for item in scene_snapshot_store.list_snapshots("task1")] == ["入口视角"]


def test_layer_rename_and_snapshot_cascade_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_store, "LAYER_DIR", tmp_path)
    monkeypatch.setattr(scene_snapshot_store, "LAYER_DIR", tmp_path)
    directory = tmp_path / "task1" / "layer1"
    directory.mkdir(parents=True)
    metadata = {"layer_id": "layer1", "task_id": "task1", "name": "瓶子1"}
    (directory / "layer.json").write_text(json.dumps(metadata), encoding="utf-8")
    scene_snapshot_store.create_snapshot("task1", _request("layer1"))

    assert layer_store.rename_layer("task1", "layer1", "主瓶")["name"] == "主瓶"
    assert scene_snapshot_store.delete_snapshots_using_layer("task1", "layer1") == 1
    layer_store.delete_layer("task1", "layer1")
    assert not directory.exists()
    assert scene_snapshot_store.list_snapshots("task1") == []
