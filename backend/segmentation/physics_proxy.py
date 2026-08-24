"""Build closed rigid-body collision proxies directly from indexed 3D Gaussians."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, QhullError

from backend.segmentation.object_cleaning import ObjectCleaningError, clean_object_gaussians
from backend.storage.file_manager import get_output_path
from backend.storage.layer_store import get_layer_metadata

ProxyType = Literal["auto", "obb", "cylinder", "convex_hull", "support_plane"]

_CYLINDER_CATEGORIES = {"bottle", "can", "cup", "vase"}
_OBB_CATEGORIES = {
    "book",
    "box",
    "laptop",
    "microwave",
    "monitor",
    "oven",
    "refrigerator",
    "television",
    "tv",
}
_SUPPORT_CATEGORIES = {"counter", "desk", "dining table", "shelf", "table"}
_MAX_FIT_GAUSSIANS = 20_000
_MAX_HULL_GAUSSIANS = 6_000


class PhysicsProxyError(RuntimeError):
    pass


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        raise PhysicsProxyError("方向向量无效")
    return vector / length


def _sample_rows(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return values[indices]


def _rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.where(norms > 1e-9, norms, 1.0)
    w, x, y, z = quaternions.T
    result = np.empty((len(quaternions), 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - z * w)
    result[:, 0, 2] = 2 * (x * z + y * w)
    result[:, 1, 0] = 2 * (x * y + z * w)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - x * w)
    result[:, 2, 0] = 2 * (x * z - y * w)
    result[:, 2, 1] = 2 * (y * z + x * w)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def _load_gaussians(source: Path, indices_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(str(source))
    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "scale_0", "scale_1", "scale_2"}
    if not required.issubset(names):
        raise PhysicsProxyError("scene.ply 缺少 Gaussian 位置或尺度字段")
    indices = np.fromfile(indices_path, dtype="<u4").astype(np.int64)
    if not len(indices):
        raise PhysicsProxyError("图层没有 Gaussian")
    if indices.max(initial=-1) >= len(vertex):
        raise PhysicsProxyError("图层 Gaussian 索引超出 scene.ply")

    positions = np.column_stack([vertex[name][indices] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    scales = np.exp(
        np.clip(
            np.column_stack([vertex[name][indices] for name in ("scale_0", "scale_1", "scale_2")]),
            -20.0,
            5.0,
        )
    ).astype(np.float64)
    if {"rot_0", "rot_1", "rot_2", "rot_3"}.issubset(names):
        rotations = np.column_stack(
            [vertex[name][indices] for name in ("rot_0", "rot_1", "rot_2", "rot_3")]
        ).astype(np.float64)
    else:
        rotations = np.zeros((len(indices), 4), dtype=np.float64)
        rotations[:, 0] = 1.0

    valid = np.isfinite(positions).all(axis=1) & np.isfinite(scales).all(axis=1)
    if "opacity" in names:
        opacity = np.asarray(vertex["opacity"][indices], dtype=np.float64)
        alpha = 1.0 / (1.0 + np.exp(-np.clip(opacity, -30.0, 30.0)))
        valid &= alpha >= 0.05
    positions, scales, rotations = positions[valid], scales[valid], rotations[valid]
    if len(positions) < 20:
        raise PhysicsProxyError("有效 Gaussian 少于 20 个")

    return positions, scales, rotations


def _observation_anchor(layer: dict, directory: Path) -> np.ndarray | None:
    width = int(layer.get("image_width") or 0)
    height = int(layer.get("image_height") or 0)
    near, far = layer.get("near"), layer.get("far")
    if width <= 0 or height <= 0 or near is None or far is None:
        return None
    depth_path, mask_path = directory / "depth.f32", directory / "mask.png"
    if not depth_path.is_file() or not mask_path.is_file():
        return None
    depth = np.fromfile(depth_path, dtype="<f4")
    if len(depth) != width * height:
        return None
    depth = depth.reshape((height, width))
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    if mask.shape != depth.shape:
        return None
    valid = mask & np.isfinite(depth) & (depth >= 0.0) & (depth <= 1.0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 20:
        return None
    if len(xs) > 4096:
        chosen = np.linspace(0, len(xs) - 1, 4096, dtype=np.int64)
        xs, ys = xs[chosen], ys[chosen]
    linear_depth = float(near) + depth[ys, xs] * (float(far) - float(near))
    projection = np.asarray(layer.get("camera_projection_matrix"), dtype=np.float64).reshape(
        4, 4, order="F"
    )
    nx = 2 * (xs + 0.5) / width - 1
    ny = 1 - 2 * (ys + 0.5) / height
    if layer.get("projection") == "orthographic":
        camera_x = (nx - projection[0, 3]) / projection[0, 0]
        camera_y = (ny - projection[1, 3]) / projection[1, 1]
    else:
        camera_x = nx * linear_depth / projection[0, 0]
        camera_y = ny * linear_depth / projection[1, 1]
    camera = np.stack([camera_x, camera_y, -linear_depth, np.ones_like(linear_depth)])
    view = np.asarray(layer.get("camera_view_matrix"), dtype=np.float64).reshape(
        4, 4, order="F"
    )
    world = np.linalg.inv(view) @ camera
    points = (world[:3] / world[3]).T
    points = points[np.isfinite(points).all(axis=1)]
    return np.median(points, axis=0) if len(points) >= 20 else None


def _surface_points(
    positions: np.ndarray, scales: np.ndarray, rotations: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(positions) > maximum:
        selected = np.linspace(0, len(positions) - 1, maximum, dtype=np.int64)
        positions, scales, rotations = positions[selected], scales[selected], rotations[selected]
    extent = float(np.max(np.ptp(positions, axis=0)))
    scale_cap = max(extent * 0.05, float(np.percentile(scales, 95)))
    scales = np.clip(scales, 1e-8, scale_cap)
    matrices = _rotation_matrices(rotations)
    offsets = matrices * scales[:, None, :]
    endpoints = np.concatenate(
        [positions[:, None, :] + offsets.transpose(0, 2, 1), positions[:, None, :] - offsets.transpose(0, 2, 1)],
        axis=1,
    ).reshape((-1, 3))
    return positions, endpoints


def _pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    covariance = np.cov((points - center).T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    frame = vectors[:, order]
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return center, frame


def _box_mesh(center: np.ndarray, frame: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signs = np.array(
        [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    vertices = center + (signs * half) @ frame.T
    faces = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _build_obb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    origin, frame = _pca_frame(points)
    local = (points - origin) @ frame
    lower, upper = local.min(axis=0), local.max(axis=0)
    center = origin + ((lower + upper) * 0.5) @ frame.T
    half = np.maximum((upper - lower) * 0.5, 1e-7)
    vertices, faces = _box_mesh(center, frame, half)
    return vertices, faces, {
        "center": center.tolist(),
        "axes": frame.T.tolist(),
        "half_extents": half.tolist(),
    }


def _build_cylinder(
    points: np.ndarray, up: np.ndarray, sections: int = 32
) -> tuple[np.ndarray, np.ndarray, dict]:
    axis = _normalize(up)
    origin = np.median(points, axis=0)
    axial = (points - origin) @ axis
    low, high = np.percentile(axial, [1.0, 99.0]).tolist()
    center = origin + axis * ((low + high) * 0.5)
    radial = points - center - np.outer((points - center) @ axis, axis)
    radius = max(float(np.percentile(np.linalg.norm(radial, axis=1), 99.0)), 1e-7)
    height = max(float(high - low), 1e-7)
    basis_a, basis_b = _plane_basis(axis)
    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    ring = np.array([math.cos(a) * basis_a + math.sin(a) * basis_b for a in angles]) * radius
    bottom_center = center - axis * height * 0.5
    top_center = center + axis * height * 0.5
    vertices = np.vstack([bottom_center + ring, top_center + ring, bottom_center, top_center])
    faces: list[list[int]] = []
    for index in range(sections):
        nxt = (index + 1) % sections
        faces.extend(
            [
                [index, nxt, sections + nxt],
                [index, sections + nxt, sections + index],
                [2 * sections, nxt, index],
                [2 * sections + 1, sections + index, sections + nxt],
            ]
        )
    return vertices, np.asarray(faces, dtype=np.int32), {
        "center": center.tolist(),
        "axis": axis.tolist(),
        "radius": radius,
        "height": height,
    }


def _build_convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    unique = np.unique(np.round(points, decimals=8), axis=0)
    try:
        hull = ConvexHull(unique)
    except QhullError as exc:
        raise PhysicsProxyError("Gaussian 几何无法生成三维凸包") from exc
    used = np.asarray(hull.vertices, dtype=np.int64)
    remap = np.full(len(unique), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    faces = remap[np.asarray(hull.simplices, dtype=np.int64)].astype(np.int32)
    vertices = unique[used]
    return vertices, faces, {"volume": float(hull.volume), "area": float(hull.area)}


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 0.0, 1.0])
    first = _normalize(np.cross(normal, helper))
    return first, _normalize(np.cross(normal, first))


def _build_support_plane(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    points = _sample_rows(points, 12_000)
    extent = max(float(np.max(np.ptp(points, axis=0))), 1e-7)
    threshold = extent * 0.012
    cosine_limit = math.cos(math.radians(30.0))
    rng = np.random.default_rng(0)
    candidates: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    for _ in range(400):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            continue
        normal /= length
        if np.dot(normal, up) < 0:
            normal *= -1
        if float(np.dot(normal, up)) < cosine_limit:
            continue
        distances = np.abs((points - sample[0]) @ normal)
        inliers = distances <= threshold
        count = int(inliers.sum())
        if count < 20:
            continue
        height = float(np.median(points[inliers] @ up))
        candidates.append((count, height, normal.copy(), inliers))
    if not candidates:
        raise PhysicsProxyError("未找到接近水平的支撑平面")
    max_count = max(item[0] for item in candidates)
    viable = [item for item in candidates if item[0] >= max_count * 0.65]
    _, _, _, seed_inliers = max(viable, key=lambda item: item[1])
    plane_points = points[seed_inliers]
    origin = plane_points.mean(axis=0)
    _, _, vh = np.linalg.svd(plane_points - origin, full_matrices=False)
    normal = vh[-1]
    if np.dot(normal, up) < 0:
        normal *= -1
    distances = np.abs((points - origin) @ normal)
    inliers = distances <= threshold
    plane_points = points[inliers]
    origin = plane_points.mean(axis=0)
    basis_a, basis_b = _plane_basis(normal)
    coordinates = np.column_stack([(plane_points - origin) @ basis_a, (plane_points - origin) @ basis_b])
    lower, upper = np.percentile(coordinates, [1.0, 99.0], axis=0)
    half_xy = np.maximum((upper - lower) * 0.5, threshold)
    thickness = max(extent * 0.02, threshold * 2.0)
    plane_center = origin + basis_a * ((lower[0] + upper[0]) * 0.5) + basis_b * (
        (lower[1] + upper[1]) * 0.5
    )
    box_center = plane_center - normal * thickness * 0.5
    frame = np.column_stack([basis_a, basis_b, normal])
    half = np.array([half_xy[0], half_xy[1], thickness * 0.5])
    vertices, faces = _box_mesh(box_center, frame, half)
    return vertices, faces, {
        "plane_origin": plane_center.tolist(),
        "plane_normal": normal.tolist(),
        "axes": [basis_a.tolist(), basis_b.tolist(), normal.tolist()],
        "half_extents": half.tolist(),
        "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "distance_threshold": threshold,
    }


def _watertight(faces: np.ndarray) -> bool:
    edges: dict[tuple[int, int], int] = {}
    for face in faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(first), int(second))))
            edges[key] = edges.get(key, 0) + 1
    return bool(edges) and all(count == 2 for count in edges.values())


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertex_data = np.empty(len(vertices), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices.T
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "O")])
    face_data["vertex_indices"] = [np.asarray(face, dtype=np.int32) for face in faces]
    PlyData(
        [PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")],
        text=False,
    ).write(str(path))


def _resolve_proxy_type(requested: ProxyType, category: str) -> str:
    if requested != "auto":
        return requested
    normalized = category.strip().lower()
    if normalized in _SUPPORT_CATEGORIES:
        return "support_plane"
    if normalized in _CYLINDER_CATEGORIES:
        return "cylinder"
    if normalized in _OBB_CATEGORIES:
        return "obb"
    return "convex_hull"


def generate_physics_proxy(
    task_id: str,
    layer_id: str,
    proxy_type: ProxyType = "auto",
    up_axis: list[float] | tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[Path, dict]:
    stored = get_layer_metadata(task_id, layer_id)
    if stored is None:
        raise KeyError(layer_id)
    layer, directory = stored
    source = get_output_path(task_id, "scene.ply")
    if source is None or not source.is_file():
        raise PhysicsProxyError("任务 scene.ply 不存在")
    entries = layer.get("gaussian_indices", [])
    if len(entries) != 1 or entries[0].get("source_index") != 0:
        raise PhysicsProxyError("当前物理代理仅支持 scene.ply 单一 Gaussian 源")
    indices_path = (directory / str(entries[0].get("file", ""))).resolve()
    if indices_path.parent != directory.resolve() or not indices_path.is_file():
        raise PhysicsProxyError("图层 Gaussian 索引不存在")

    positions, scales, rotations = _load_gaussians(source, indices_path)
    try:
        cleaned = clean_object_gaussians(
            positions,
            scales,
            rotations,
            observation_anchor=_observation_anchor(layer, directory),
        )
    except ObjectCleaningError as exc:
        raise PhysicsProxyError(f"三维主体清理失败：{exc}") from exc
    positions, scales, rotations = cleaned.positions, cleaned.scales, cleaned.rotations
    category = str(layer.get("category") or "")
    resolved = _resolve_proxy_type(proxy_type, category)
    maximum = _MAX_HULL_GAUSSIANS if resolved == "convex_hull" else _MAX_FIT_GAUSSIANS
    centers, surface = _surface_points(positions, scales, rotations, maximum)
    fit_points = np.vstack([centers, surface])

    if resolved == "obb":
        vertices, faces, geometry = _build_obb(fit_points)
    elif resolved == "cylinder":
        vertices, faces, geometry = _build_cylinder(
            fit_points, _normalize(np.asarray(up_axis, dtype=np.float64))
        )
    elif resolved == "convex_hull":
        vertices, faces, geometry = _build_convex_hull(fit_points)
    elif resolved == "support_plane":
        vertices, faces, geometry = _build_support_plane(centers, _normalize(np.asarray(up_axis)))
    else:
        raise PhysicsProxyError(f"不支持的物理代理类型：{resolved}")

    watertight = _watertight(faces)
    physics_ready = watertight and len(vertices) >= 4 and len(faces) >= 4
    if not physics_ready:
        raise PhysicsProxyError("生成的物理代理未通过闭合检查")
    output = directory / "physics_proxy.ply"
    _write_mesh(output, vertices, faces)
    report = {
        "engine": "gaussian-physics-proxy",
        "proxy_type": resolved,
        "requested_proxy_type": proxy_type,
        "category": category,
        "geometry_source": "gaussian-centers-scales-rotations",
        "source_gaussians": cleaned.report["source_count"],
        "used_gaussians": len(centers),
        "object_cleaning": cleaned.report,
        "vertices": len(vertices),
        "triangles": len(faces),
        "watertight": watertight,
        "physics_ready": physics_ready,
        "proxy_file": output.name,
        "geometry": geometry,
        "generated_at": datetime.now(UTC).isoformat(),
        "report_version": 1,
    }
    report_path = directory / "physics-proxy-report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return output, report
