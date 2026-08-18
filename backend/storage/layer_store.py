"""Persistent 2D segmentation layers."""

import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from backend.core.config import LAYER_DIR
from backend.core.schemas import SemanticGaussianIndexSet
from backend.segmentation.service import SegmentationSession
from backend.storage.file_manager import get_output_path


def _task_dir(task_id: str) -> Path:
    if not task_id.isalnum() or len(task_id) > 64:
        raise ValueError("无效任务 ID")
    return LAYER_DIR / task_id


def hash_task_scene_ply(task_id: str) -> tuple[str | None, str]:
    path = get_output_path(task_id, "scene.ply")
    if path is None or not path.is_file():
        return None, "unavailable: task scene.ply not found in server output directory"
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return None, f"unavailable: scene.ply could not be read ({type(exc).__name__})"
    return digest.hexdigest(), "verified: task output scene.ply"


def _validated_index_sets(
    session: SegmentationSession,
    index_sets: list[SemanticGaussianIndexSet | dict] | None,
) -> list[SemanticGaussianIndexSet]:
    validated = [
        item
        if isinstance(item, SemanticGaussianIndexSet)
        else SemanticGaussianIndexSet.model_validate(item)
        for item in index_sets or []
    ]
    if any(item.instance_id != session.session_id for item in validated):
        raise ValueError("Gaussian index set 与分割实例不匹配")
    return validated


def create_layer(
    session: SegmentationSession,
    name: str,
    gaussian_index_sets: list[SemanticGaussianIndexSet | dict] | None = None,
    source_ply_fingerprint: tuple[str | None, str] | None = None,
) -> dict:
    if session.mask_png is None or session.mask_rle is None or session.points is None:
        raise ValueError("分割会话尚无 mask")
    index_sets = _validated_index_sets(session, gaussian_index_sets)
    layer_id = uuid.uuid4().hex[:12]
    directory = _task_dir(session.task_id) / layer_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "mask.png").write_bytes(session.mask_png)
    if session.depth_f32 is not None:
        (directory / "depth.f32").write_bytes(session.depth_f32)
    gaussian_indices = []
    for index_set in index_sets:
        filename = f"gaussian-indices-source-{index_set.source_index}.u32"
        payload = np.asarray(index_set.indices, dtype="<u4").tobytes(order="C")
        (directory / filename).write_bytes(payload)
        gaussian_indices.append(
            {
                "instance_id": index_set.instance_id,
                "source_index": index_set.source_index,
                "encoding": "uint32-le",
                "count": len(index_set.indices),
                "vertex_count": index_set.source_vertex_count,
                "file": filename,
                "url": (
                    f"/api/tasks/{session.task_id}/layers/{layer_id}/"
                    f"gaussian-indices/{index_set.source_index}"
                ),
            }
        )
    source_ply_sha256, source_ply_sha256_status = source_ply_fingerprint or hash_task_scene_ply(
        session.task_id
    )
    created_at = datetime.now(UTC).isoformat()
    data = {
        "layer_id": layer_id,
        "task_id": session.task_id,
        "name": name,
        "source_ply": session.source_ply,
        "camera_view_matrix": session.view_matrix,
        "camera_projection_matrix": session.projection_matrix,
        "viewport_width": session.viewport_width,
        "viewport_height": session.viewport_height,
        "image_width": session.width,
        "image_height": session.height,
        "points": session.points,
        "mask_rle": session.mask_rle,
        "score": session.score,
        "bbox": session.bbox,
        "category": session.category,
        "category_zh": session.category_zh,
        "instance_index": session.instance_index,
        "depth_coverage": session.depth_coverage,
        "view_support": session.view_support,
        "view_count": session.view_count,
        "near": session.near,
        "far": session.far,
        "projection": session.projection,
        "auxiliary_views": session.auxiliary_views,
        "depth_file": "depth.f32" if session.depth_f32 is not None else None,
        "depth_format": "float32-normalized-linear-view-z"
        if session.depth_f32 is not None
        else None,
        "created_at": created_at,
        "mask_url": f"/api/tasks/{session.task_id}/layers/{layer_id}/mask",
        "gaussian_indices": gaussian_indices,
        "source_ply_sha256": source_ply_sha256,
        "source_ply_sha256_status": source_ply_sha256_status,
        "observation_count": 1,
        "observations": [],
    }
    (directory / "layer.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return data


def list_layers(task_id: str) -> list[dict]:
    directory = _task_dir(task_id)
    if not directory.exists():
        return []
    result = []
    for path in sorted(directory.glob("*/layer.json")):
        result.append(json.loads(path.read_text(encoding="utf-8")))
    return result


def rename_layer(task_id: str, layer_id: str, name: str) -> dict:
    path = _layer_metadata_path(task_id, layer_id)
    if path is None:
        raise KeyError(layer_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["name"] = name.strip()
    _write_json(path, data)
    return data


def merge_layer_observation(
    task_id: str,
    layer_id: str,
    session: SegmentationSession,
    gaussian_index_sets: list[SemanticGaussianIndexSet],
) -> tuple[dict, int, int]:
    """Union one new camera observation into an existing semantic layer."""
    path = _layer_metadata_path(task_id, layer_id)
    if path is None:
        raise KeyError(layer_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("category") and session.category and data["category"] != session.category:
        raise ValueError("识别类别与目标图层不一致")

    directory = path.parent
    entries = {item["source_index"]: item for item in data.get("gaussian_indices", [])}
    added_count = 0
    total_count = 0
    added_by_source: list[dict] = []
    for index_set in gaussian_index_sets:
        entry = entries.get(index_set.source_index)
        if entry and entry.get("vertex_count") != index_set.source_vertex_count:
            raise ValueError("目标图层与当前 Gaussian 资源版本不一致")
        existing: set[int] = set()
        if entry:
            existing_path = directory / entry["file"]
            if existing_path.is_file():
                existing = set(np.fromfile(existing_path, dtype="<u4").tolist())
        incoming = set(index_set.indices)
        added = sorted(incoming - existing)
        merged = sorted(existing | incoming)
        filename = (
            entry["file"] if entry else f"gaussian-indices-source-{index_set.source_index}.u32"
        )
        np.asarray(merged, dtype="<u4").tofile(directory / filename)
        if entry is None:
            entry = {
                "instance_id": data.get("layer_id", layer_id),
                "source_index": index_set.source_index,
                "encoding": "uint32-le",
                "vertex_count": index_set.source_vertex_count,
                "file": filename,
                "url": f"/api/tasks/{task_id}/layers/{layer_id}/gaussian-indices/{index_set.source_index}",
            }
            data.setdefault("gaussian_indices", []).append(entry)
            entries[index_set.source_index] = entry
        entry["count"] = len(merged)
        added_count += len(added)
        total_count += len(merged)
        added_by_source.append({"source_index": index_set.source_index, "indices": added})

    observation_number = int(data.get("observation_count", 1)) + 1
    observation_mask = f"observation-{observation_number}-mask.png"
    (directory / observation_mask).write_bytes(session.mask_png)
    data.setdefault("observations", []).append(
        {
            "observation_index": observation_number,
            "created_at": datetime.now(UTC).isoformat(),
            "category": session.category,
            "score": session.score,
            "camera_view_matrix": session.view_matrix,
            "camera_projection_matrix": session.projection_matrix,
            "viewport_width": session.viewport_width,
            "viewport_height": session.viewport_height,
            "mask_file": observation_mask,
            "added_gaussian_indices": added_by_source,
        }
    )
    data["observation_count"] = observation_number
    data["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(path, data)
    return data, added_count, total_count


def delete_layer(task_id: str, layer_id: str) -> None:
    path = _layer_metadata_path(task_id, layer_id)
    if path is None:
        raise KeyError(layer_id)
    directory = path.parent.resolve()
    if directory.parent != _task_dir(task_id).resolve():
        raise ValueError("图层目录越界")
    shutil.rmtree(directory)


def _layer_metadata_path(task_id: str, layer_id: str) -> Path | None:
    if not layer_id.isalnum() or len(layer_id) > 32:
        return None
    path = _task_dir(task_id) / layer_id / "layer.json"
    return path if path.is_file() else None


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def get_mask_path(task_id: str, layer_id: str) -> Path | None:
    if not layer_id.isalnum() or len(layer_id) > 32:
        return None
    path = _task_dir(task_id) / layer_id / "mask.png"
    return path if path.is_file() else None


def get_gaussian_indices_path(
    task_id: str,
    layer_id: str,
    source_index: int,
) -> Path | None:
    if not layer_id.isalnum() or len(layer_id) > 32 or source_index < 0:
        return None
    directory = _task_dir(task_id) / layer_id
    metadata_path = directory / "layer.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entry = next(
        (
            item
            for item in metadata.get("gaussian_indices", [])
            if item.get("source_index") == source_index
        ),
        None,
    )
    if entry is None or not isinstance(entry.get("file"), str):
        return None
    path = (directory / entry["file"]).resolve()
    if path.parent != directory.resolve():
        return None
    return path if path.is_file() else None


def get_layer_metadata(task_id: str, layer_id: str) -> tuple[dict, Path] | None:
    """Return validated layer metadata and its private storage directory."""
    path = _layer_metadata_path(task_id, layer_id)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data, path.parent
