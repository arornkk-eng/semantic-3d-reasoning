import asyncio
import hashlib
import json
import struct
import time

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from backend.api import segmentation as api
from backend.core import gpu_coordinator, schemas
from backend.core.schemas import (
    LayerCreateRequest,
    SemanticConfirmRequest,
    SemanticGaussianIndexSet,
    SemanticInstanceResponse,
)
from backend.segmentation.semantic_service import (
    SCORE_THRESHOLD,
    TARGETS,
    _decode_depth_map,
    _match_instances,
    _multiview_overlap,
    _target_instances,
    _TargetInstance,
    _validate_capture_dimensions,
)
from backend.segmentation.service import SegmentationSession, SegmentationViewMask, _encode_mask
from backend.storage import layer_store


def test_gpu_exclusion():
    assert gpu_coordinator.try_begin_segmentation("session-a")
    assert not gpu_coordinator.try_begin_reconstruction()
    gpu_coordinator.finish_segmentation("session-a")
    assert gpu_coordinator.try_begin_reconstruction()
    assert not gpu_coordinator.try_begin_segmentation("session-b")
    gpu_coordinator.finish_reconstruction()
    assert gpu_coordinator.status() == "idle"


def test_mask_encoding_roundtrip_shape_and_bbox():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:5] = True
    png, rle, bbox = _encode_mask(mask)
    assert png.startswith(b"\x89PNG")
    assert rle["size"] == [4, 5]
    assert sum(rle["counts"]) == 20
    assert bbox == [2, 1, 4, 2]


def test_parse_metadata(monkeypatch):
    monkeypatch.setattr(api, "get_task_meta", lambda _task_id: {"status": "completed"})
    raw = json.dumps(
        {
            "task_id": "abc123",
            "source_ply": "scene.ply",
            "viewport_width": 800,
            "viewport_height": 600,
            "capture_width": 640,
            "capture_height": 480,
            "view_matrix": list(range(16)),
            "projection_matrix": list(range(16)),
        }
    )
    parsed = api._parse_metadata(raw)
    assert parsed["viewport_width"] == 800
    assert parsed["view_matrix"][15] == 15.0


def test_layer_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_store, "LAYER_DIR", tmp_path)
    scene_ply = tmp_path / "scene.ply"
    scene_ply.write_bytes(b"ply\nsemantic-source\n")
    monkeypatch.setattr(
        layer_store,
        "get_output_path",
        lambda task_id, filename: scene_ply
        if task_id == "abc123" and filename == "scene.ply"
        else None,
    )
    mask = np.ones((2, 2), dtype=bool)
    png, rle, bbox = _encode_mask(mask)
    session = SegmentationSession(
        session_id="session",
        task_id="abc123",
        source_ply="scene.ply",
        width=2,
        height=2,
        viewport_width=800,
        viewport_height=600,
        view_matrix=[0.0] * 16,
        projection_matrix=[0.0] * 16,
        created_at=time.monotonic(),
        touched_at=time.monotonic(),
        points=[{"x": 0.5, "y": 0.5, "label": 1}],
        mask_png=png,
        mask_rle=rle,
        score=0.9,
        bbox=bbox,
        depth_coverage=0.8,
        view_support=2,
        view_count=3,
        depth_f32=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32).tobytes(),
    )
    index_set = SemanticGaussianIndexSet(
        instance_id="session",
        source_index=0,
        source_vertex_count=4,
        indices=[0, 2, 3],
    )
    layer = layer_store.create_layer(session, "Bottle", [index_set])
    assert layer["name"] == "Bottle"
    assert layer["depth_coverage"] == 0.8
    assert layer["view_support"] == 2
    assert layer["view_count"] == 3
    assert (tmp_path / "abc123" / layer["layer_id"] / "depth.f32").is_file()
    index_file = tmp_path / "abc123" / layer["layer_id"] / "gaussian-indices-source-0.u32"
    assert index_file.read_bytes() == struct.pack("<3I", 0, 2, 3)
    assert layer["gaussian_indices"][0]["encoding"] == "uint32-le"
    assert layer["gaussian_indices"][0]["count"] == 3
    assert layer["gaussian_indices"][0]["vertex_count"] == 4
    assert layer["gaussian_indices"][0]["url"].endswith("/gaussian-indices/0")
    assert layer["source_ply_sha256"] == hashlib.sha256(scene_ply.read_bytes()).hexdigest()
    assert layer_store.get_gaussian_indices_path("abc123", layer["layer_id"], 0) == index_file
    assert len(layer_store.list_layers("abc123")) == 1
    assert layer_store.get_mask_path("abc123", layer["layer_id"]).is_file()


def test_semantic_confirm_request_remains_backward_compatible():
    request = SemanticConfirmRequest(instance_ids=["chair-1"])
    assert request.gaussian_index_sets == []


def test_layer_create_request_remains_backward_compatible():
    request = LayerCreateRequest(session_id="sam-session", name="Object")
    assert request.gaussian_index_sets == []


def test_layer_create_request_accepts_matching_index_sets():
    request = LayerCreateRequest(
        session_id="sam-session",
        name="Object",
        gaussian_index_sets=[
            {
                "instance_id": "sam-session",
                "source_index": 0,
                "source_vertex_count": 4,
                "indices": [0, 2],
            }
        ],
    )
    assert request.gaussian_index_sets[0].indices == [0, 2]


def test_layer_create_request_rejects_different_session():
    with pytest.raises(ValidationError, match="不属于当前分割会话"):
        LayerCreateRequest(
            session_id="sam-session",
            name="Object",
            gaussian_index_sets=[
                {
                    "instance_id": "other-session",
                    "source_index": 0,
                    "source_vertex_count": 4,
                    "indices": [0],
                }
            ],
        )


def test_manual_layer_confirm_passes_index_sets(monkeypatch):
    mask = np.ones((1, 1), dtype=bool)
    png, rle, bbox = _encode_mask(mask)
    session = SegmentationSession(
        session_id="sam-session",
        task_id="abc123",
        source_ply="scene.ply",
        width=1,
        height=1,
        viewport_width=1,
        viewport_height=1,
        view_matrix=[0.0] * 16,
        projection_matrix=[0.0] * 16,
        created_at=time.monotonic(),
        touched_at=time.monotonic(),
        points=[],
        mask_png=png,
        mask_rle=rle,
        bbox=bbox,
    )
    request = LayerCreateRequest(
        session_id="sam-session",
        name="Object",
        gaussian_index_sets=[
            {
                "instance_id": "sam-session",
                "source_index": 0,
                "source_vertex_count": 4,
                "indices": [1, 3],
            }
        ],
    )
    captured = {}
    monkeypatch.setattr(api.segmentation_service, "get", lambda _session_id: session)
    monkeypatch.setattr(api.segmentation_service, "close", lambda session_id: None)
    monkeypatch.setattr(
        api,
        "get_task_meta",
        lambda _task_id: {"output": {"num_gaussians": 4}},
    )

    def fake_create_layer(received_session, name, gaussian_index_sets=None):
        captured["session"] = received_session
        captured["name"] = name
        captured["index_sets"] = gaussian_index_sets
        return {"layer_id": "layer-1"}

    monkeypatch.setattr(api, "create_layer", fake_create_layer)

    layer = asyncio.run(api.confirm_segmentation_layer("abc123", request))

    assert layer["layer_id"] == "layer-1"
    assert captured["session"] is session
    assert captured["name"] == "Object"
    assert captured["index_sets"][0].indices == [1, 3]


@pytest.mark.parametrize(
    "payload",
    [
        {
            "instance_id": "chair-1",
            "source_index": 0,
            "source_vertex_count": 3,
            "indices": [0, 0],
        },
        {
            "instance_id": "chair-1",
            "source_index": 0,
            "source_vertex_count": 3,
            "indices": [3],
        },
        {
            "instance_id": "chair-1",
            "source_index": -1,
            "source_vertex_count": 3,
            "indices": [0],
        },
        {
            "instance_id": "chair-1",
            "source_index": 0,
            "source_vertex_count": 3,
            "indices": ["0"],
        },
    ],
)
def test_gaussian_index_set_strict_validation(payload):
    with pytest.raises(ValidationError):
        SemanticGaussianIndexSet.model_validate(payload)


def test_confirm_request_rejects_index_set_for_unselected_instance():
    with pytest.raises(ValidationError, match="不属于已选实例"):
        SemanticConfirmRequest(
            instance_ids=["chair-1"],
            gaussian_index_sets=[
                {
                    "instance_id": "chair-2",
                    "source_index": 0,
                    "source_vertex_count": 3,
                    "indices": [0],
                }
            ],
        )


def test_confirm_request_enforces_total_index_limit(monkeypatch):
    monkeypatch.setattr(schemas, "MAX_GAUSSIAN_INDICES_TOTAL", 2)
    with pytest.raises(ValidationError, match="总数超出限制"):
        SemanticConfirmRequest(
            instance_ids=["chair-1"],
            gaussian_index_sets=[
                {
                    "instance_id": "chair-1",
                    "source_index": 0,
                    "source_vertex_count": 3,
                    "indices": [0, 1],
                },
                {
                    "instance_id": "chair-1",
                    "source_index": 1,
                    "source_vertex_count": 3,
                    "indices": [2],
                },
            ],
        )


def test_layer_store_rejects_index_set_for_different_session(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_store, "LAYER_DIR", tmp_path)
    mask = np.ones((1, 1), dtype=bool)
    png, rle, bbox = _encode_mask(mask)
    session = SegmentationSession(
        session_id="chair-1",
        task_id="abc123",
        source_ply="scene.ply",
        width=1,
        height=1,
        viewport_width=1,
        viewport_height=1,
        view_matrix=[0.0] * 16,
        projection_matrix=[0.0] * 16,
        created_at=time.monotonic(),
        touched_at=time.monotonic(),
        points=[],
        mask_png=png,
        mask_rle=rle,
        bbox=bbox,
    )
    with pytest.raises(ValueError, match="实例不匹配"):
        layer_store.create_layer(
            session,
            "Chair",
            [
                SemanticGaussianIndexSet(
                    instance_id="chair-2",
                    source_index=0,
                    source_vertex_count=1,
                    indices=[0],
                )
            ],
        )


def test_invalid_task_id_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(layer_store, "LAYER_DIR", tmp_path)
    with pytest.raises(ValueError):
        layer_store.list_layers("../escape")


def test_fixed_semantic_categories():
    assert TARGETS == {
        "cup": ("cup", "杯子"),
        "chair": ("chair", "椅子"),
        "bottle": ("bottle", "瓶子"),
    }
    assert SCORE_THRESHOLD == 0.40


def _column_major(matrix: np.ndarray) -> list[float]:
    return matrix.flatten(order="F").tolist()


def test_multiview_overlap_reprojects_perspective_depth():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 1] = 0.4
    projection = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.2, -2.2],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    view = {
        "near": 1.0,
        "far": 11.0,
        "projection": "perspective",
        "view_matrix": _column_major(np.eye(4)),
        "projection_matrix": _column_major(projection),
    }

    assert _multiview_overlap(mask, depth, view, mask, view) == 1.0
    assert _multiview_overlap(mask, depth, view, np.zeros_like(mask), view) == 0.0


def _target(category: str, mask: np.ndarray) -> _TargetInstance:
    return _TargetInstance(category, category, 0.9, mask, [0, 0, 1, 1])


def test_target_instances_keep_same_category_masks_separate():
    masks = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    masks[0, 0, 0, 0] = 1
    masks[1, 0, 1, 1] = 1
    prediction = {
        "labels": torch.tensor([0, 0]),
        "scores": torch.tensor([0.9, 0.8]),
        "boxes": torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
        "masks": masks,
    }

    instances = _target_instances(prediction, ["chair"])

    assert len(instances) == 2
    assert instances[0].mask[0, 0]
    assert not instances[0].mask[1, 1]
    assert instances[1].mask[1, 1]


def test_instance_matching_is_one_to_one():
    mask = np.ones((2, 2), dtype=bool)
    centers = [_target("chair", mask), _target("chair", mask)]
    targets = [_target("chair", mask)]

    matches = _match_instances(centers, targets)

    assert matches == {0: (0, 1.0)}


def test_depth_map_requires_exact_size_and_normalized_values():
    valid = np.asarray([0.0, 0.5, np.nan, 1.0], dtype="<f4").tobytes()
    assert _decode_depth_map(valid, 2, 2, 0).shape == (2, 2)
    with pytest.raises(ValueError, match="expected 16"):
        _decode_depth_map(valid[:-1], 2, 2, 0)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        _decode_depth_map(np.asarray([0.0, 1.1], dtype="<f4").tobytes(), 2, 1, 0)
    with pytest.raises(ValueError, match="infinity"):
        _decode_depth_map(np.asarray([0.0, np.inf], dtype="<f4").tobytes(), 2, 1, 0)


def test_capture_dimensions_must_match_decoded_image():
    image = Image.new("RGB", (4, 3))
    _validate_capture_dimensions(image, {"capture_width": 4, "capture_height": 3})
    with pytest.raises(ValueError, match="do not match"):
        _validate_capture_dimensions(image, {"capture_width": 3, "capture_height": 4})


def test_semantic_depth_count_must_match_images():
    api._validate_semantic_upload_counts(3, 3)
    api._validate_semantic_upload_counts(1, 0)
    with pytest.raises(HTTPException) as error:
        api._validate_semantic_upload_counts(3, 1)
    assert error.value.status_code == 400


def test_semantic_response_keeps_mask_url_and_adds_view_masks():
    legacy_response = SemanticInstanceResponse(
        instance_id="chair-1",
        category="chair",
        category_zh="椅子",
        instance_index=1,
        score=0.9,
        bbox=[0, 0, 1, 1],
        mask_url="/mask",
    )
    assert legacy_response.mask_url == "/mask"
    assert legacy_response.view_masks == []

    payload = legacy_response.model_dump()
    payload["view_masks"] = [{"view_index": 0, "mask_url": "/views/0/mask"}]
    validated = SemanticInstanceResponse.model_validate(payload)
    assert validated.view_masks[0].view_index == 0
    assert validated.view_masks[0].mask_url == "/views/0/mask"

    png, _, _ = _encode_mask(np.ones((1, 1), dtype=bool))
    session = SegmentationSession(
        session_id="chair-1",
        task_id="task",
        source_ply="scene.ply",
        width=1,
        height=1,
        viewport_width=1,
        viewport_height=1,
        view_matrix=[0.0] * 16,
        projection_matrix=[0.0] * 16,
        created_at=time.monotonic(),
        touched_at=time.monotonic(),
        mask_png=png,
        view_masks=[SegmentationViewMask(view_index=0, mask_png=png)],
    )
    assert session.view_masks[0].view_index == 0


def test_multiview_overlap_reprojects_orthographic_depth():
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True
    depth = np.full((3, 3), np.nan, dtype=np.float32)
    depth[1, 1] = 0.4
    projection = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, -0.2, -1.2],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    view = {
        "near": 1.0,
        "far": 11.0,
        "projection": "orthographic",
        "view_matrix": _column_major(np.eye(4)),
        "projection_matrix": _column_major(projection),
    }

    assert _multiview_overlap(mask, depth, view, mask, view) == 1.0
