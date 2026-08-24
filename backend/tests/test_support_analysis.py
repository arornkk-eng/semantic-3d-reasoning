import numpy as np
import pytest

pytest.importorskip("pybullet")

from backend.physics import support_analysis
from backend.segmentation.physics_proxy import _box_mesh


def _body(layer_id: str, name: str, category: str, center: list[float], half: list[float]):
    center_array = np.asarray(center)
    half_array = np.asarray(half)
    vertices, faces = _box_mesh(center_array, np.eye(3), half_array)
    base_position = vertices.mean(axis=0)
    return support_analysis.ProxyBody(
        layer_id=layer_id,
        name=name,
        category=category,
        vertices=vertices,
        faces=faces,
        report={
            "proxy_type": "obb",
            "physics_ready": True,
            "geometry": {
                "center": center_array.tolist(),
                "axes": np.eye(3).tolist(),
                "half_extents": half_array.tolist(),
            },
        },
        local_vertices=vertices - base_position,
        base_position=base_position,
        extent=float(np.max(np.ptp(vertices, axis=0))),
    )


def test_pybullet_counterfactual_detects_box_on_table(monkeypatch):
    table = _body("table1", "桌子1", "table", [0.0, -0.05, 0.0], [0.6, 0.05, 0.6])
    bottle = _body("bottle1", "水瓶1", "bottle", [0.0, 0.2, 0.0], [0.1, 0.2, 0.1])
    layers = [
        {"layer_id": "table1", "name": "桌子1", "category": "table"},
        {"layer_id": "bottle1", "name": "水瓶1", "category": "bottle"},
    ]
    monkeypatch.setattr(support_analysis, "list_layers", lambda _task_id: layers)
    monkeypatch.setattr(
        support_analysis,
        "get_ground_calibration",
        lambda _task_id: {
            "confirmed": True,
            "normal": [0.0, 1.0, 0.0],
            "method": "layer_ransac",
            "ground_layer_id": "table1",
        },
    )
    monkeypatch.setattr(
        support_analysis,
        "_prepare_bodies",
        lambda *_args: ([table, bottle], np.array([0.0, 1.0, 0.0]), 1.0, np.zeros(3)),
    )
    monkeypatch.setattr(support_analysis, "save_support_analysis", lambda _task_id, result: result)

    result = support_analysis.analyze_support_relations("task1")

    assert result["engine"] == "pybullet-3.2.7"
    assert result["subject_states"][0]["stable"] is True, result
    assert len(result["relations"]) == 1
    relation = result["relations"][0]
    assert relation["subject"] == "bottle1"
    assert relation["predicate"] == "supported_by"
    assert relation["object"] == "table1"
    assert relation["evidence"]["contact_ratio"] > 0.9
    assert relation["evidence"]["fall_distance_after_removal"] > 0.02


def test_no_relation_for_floating_object(monkeypatch):
    table = _body("table1", "桌子1", "table", [0.0, -0.05, 0.0], [0.6, 0.05, 0.6])
    bottle = _body("bottle1", "水瓶1", "bottle", [2.0, 0.5, 0.0], [0.1, 0.2, 0.1])
    layers = [
        {"layer_id": "table1", "name": "桌子1", "category": "table"},
        {"layer_id": "bottle1", "name": "水瓶1", "category": "bottle"},
    ]
    monkeypatch.setattr(support_analysis, "list_layers", lambda _task_id: layers)
    monkeypatch.setattr(
        support_analysis,
        "get_ground_calibration",
        lambda _task_id: {
            "confirmed": True,
            "normal": [0.0, 1.0, 0.0],
            "method": "layer_ransac",
            "ground_layer_id": "table1",
        },
    )
    monkeypatch.setattr(
        support_analysis,
        "_prepare_bodies",
        lambda *_args: ([table, bottle], np.array([0.0, 1.0, 0.0]), 1.0, np.zeros(3)),
    )
    monkeypatch.setattr(support_analysis, "save_support_analysis", lambda _task_id, result: result)

    result = support_analysis.analyze_support_relations("task1")

    assert result["relations"] == []


def test_analysis_requires_confirmed_ground(monkeypatch):
    monkeypatch.setattr(
        support_analysis,
        "list_layers",
        lambda _task_id: [
            {"layer_id": "ground1", "name": "地面", "category": "floor"},
            {"layer_id": "box1", "name": "箱子", "category": "box"},
        ],
    )
    monkeypatch.setattr(
        support_analysis,
        "get_ground_calibration",
        lambda _task_id: {"confirmed": False, "normal": [0.0, 1.0, 0.0]},
    )

    with pytest.raises(support_analysis.SupportAnalysisError, match="确认法线"):
        support_analysis.analyze_support_relations("task1")
