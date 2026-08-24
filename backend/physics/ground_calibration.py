"""Manual ground selection with geometric plane fitting and explicit normal confirmation."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from scipy.spatial.transform import Rotation

from backend.segmentation.physics_proxy import PhysicsProxyError, _load_gaussians
from backend.storage.file_manager import get_output_path
from backend.storage.ground_calibration_store import (
    get_ground_calibration,
    save_ground_calibration,
)
from backend.storage.layer_store import get_layer_metadata


class GroundCalibrationError(RuntimeError):
    pass


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        raise GroundCalibrationError("地面法线无效")
    return vector / length


def _world_transform(origin: np.ndarray, normal: np.ndarray) -> list[list[float]]:
    rotation, _ = Rotation.align_vectors([[0.0, 1.0, 0.0]], [normal])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = -matrix[:3, :3] @ origin
    return matrix.tolist()


def _orient_deterministically(normal: np.ndarray) -> np.ndarray:
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0:
        normal *= -1
    return normal


def _fit_plane_ransac(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if len(points) < 20:
        raise GroundCalibrationError("地面图层有效 Gaussian 少于 20 个")
    points = points[
        np.linspace(0, len(points) - 1, min(len(points), 20_000), dtype=np.int64)
    ]
    extent = float(np.max(np.ptp(points, axis=0)))
    if not np.isfinite(extent) or extent < 1e-7:
        raise GroundCalibrationError("地面图层空间范围过小")
    threshold = max(extent * 0.01, 1e-7)
    rng = np.random.default_rng(0)
    best: np.ndarray | None = None
    for _ in range(600):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal /= length
        distances = np.abs((points - sample[0]) @ normal)
        inliers = distances <= threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or int(best.sum()) < 20:
        raise GroundCalibrationError("地面图层未拟合出有效平面")

    plane_points = points[best]
    origin = plane_points.mean(axis=0)
    _, _, vh = np.linalg.svd(plane_points - origin, full_matrices=False)
    normal = _orient_deterministically(_normalize(vh[-1]))
    distances = np.abs((points - origin) @ normal)
    inliers = distances <= threshold
    plane_points = points[inliers]
    origin = plane_points.mean(axis=0)
    residuals = (plane_points - origin) @ normal
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    return origin, normal, plane_points, float(inliers.mean()), rmse


def _plane_boundary(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> dict:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 0.0, 1.0])
    axis_u = _normalize(np.cross(normal, helper))
    axis_v = _normalize(np.cross(normal, axis_u))
    coordinates = np.column_stack([(points - origin) @ axis_u, (points - origin) @ axis_v])
    lower, upper = np.percentile(coordinates, [1.0, 99.0], axis=0)
    return {
        "axis_u": axis_u.tolist(),
        "axis_v": axis_v.tolist(),
        "min": lower.tolist(),
        "max": upper.tolist(),
    }


def _base_result(
    task_id: str,
    method: str,
    origin: np.ndarray,
    normal: np.ndarray,
    inlier_ratio: float,
    fit_error: float,
    ground_layer_id: str | None,
    boundary: dict | None,
    points: list[list[float]] | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "method": method,
        "ground_layer_id": ground_layer_id,
        "origin": origin.tolist(),
        "normal": normal.tolist(),
        "flipped": False,
        "inlier_ratio": inlier_ratio,
        "fit_error": fit_error,
        "boundary": boundary,
        "points": points or [],
        "confirmed": False,
        "world_up": [0.0, 1.0, 0.0],
        "gravity": [0.0, -9.81, 0.0],
        "world_from_scene": _world_transform(origin, normal),
        "updated_at": datetime.now(UTC).isoformat(),
    }


def calibrate_from_layer(task_id: str, layer_id: str) -> dict:
    stored = get_layer_metadata(task_id, layer_id)
    if stored is None:
        raise GroundCalibrationError("地面图层不存在")
    layer, directory = stored
    source = get_output_path(task_id, "scene.ply")
    if source is None or not source.is_file():
        raise GroundCalibrationError("任务 scene.ply 不存在")
    entries = layer.get("gaussian_indices", [])
    if len(entries) != 1 or entries[0].get("source_index") != 0:
        raise GroundCalibrationError("当前地面标定仅支持 scene.ply 单一 Gaussian 源")
    indices = (directory / str(entries[0].get("file", ""))).resolve()
    if indices.parent != directory.resolve() or not indices.is_file():
        raise GroundCalibrationError("地面图层 Gaussian 索引不存在")
    try:
        positions, _, _ = _load_gaussians(source, indices)
    except PhysicsProxyError as exc:
        raise GroundCalibrationError(str(exc)) from exc
    origin, normal, inliers, ratio, rmse = _fit_plane_ransac(positions)
    result = _base_result(
        task_id,
        "layer_ransac",
        origin,
        normal,
        ratio,
        rmse,
        layer_id,
        _plane_boundary(inliers, origin, normal),
    )
    return save_ground_calibration(task_id, result)


def calibrate_from_points(task_id: str, points: list[list[float]]) -> dict:
    values = np.asarray(points, dtype=np.float64)
    if values.shape != (3, 3) or not np.isfinite(values).all():
        raise GroundCalibrationError("三点地面坐标无效")
    edge_a, edge_b = values[1] - values[0], values[2] - values[0]
    scale = max(float(np.linalg.norm(edge_a)), float(np.linalg.norm(edge_b)))
    cross = np.cross(edge_a, edge_b)
    if scale < 1e-7 or float(np.linalg.norm(cross)) < scale * scale * 0.01:
        raise GroundCalibrationError("三个地面点过近或接近共线")
    normal = _orient_deterministically(_normalize(cross))
    origin = values.mean(axis=0)
    result = _base_result(
        task_id,
        "three_points",
        origin,
        normal,
        1.0,
        0.0,
        None,
        _plane_boundary(values, origin, normal),
        points=values.tolist(),
    )
    return save_ground_calibration(task_id, result)


def flip_ground_normal(task_id: str) -> dict:
    calibration = get_ground_calibration(task_id)
    if calibration is None:
        raise GroundCalibrationError("尚未创建地面标定")
    normal = -_normalize(np.asarray(calibration["normal"], dtype=np.float64))
    origin = np.asarray(calibration["origin"], dtype=np.float64)
    calibration["normal"] = normal.tolist()
    calibration["flipped"] = not bool(calibration.get("flipped", False))
    calibration["confirmed"] = False
    calibration["world_from_scene"] = _world_transform(origin, normal)
    calibration["updated_at"] = datetime.now(UTC).isoformat()
    return save_ground_calibration(task_id, calibration)


def confirm_ground_calibration(task_id: str) -> dict:
    calibration = get_ground_calibration(task_id)
    if calibration is None:
        raise GroundCalibrationError("尚未创建地面标定")
    calibration["confirmed"] = True
    calibration["confirmed_at"] = datetime.now(UTC).isoformat()
    calibration["updated_at"] = calibration["confirmed_at"]
    return save_ground_calibration(task_id, calibration)
