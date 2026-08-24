"""Infer static rigid-body support relations with PyBullet counterfactual simulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from backend.segmentation.physics_proxy import PhysicsProxyError, generate_physics_proxy
from backend.storage.ground_calibration_store import get_ground_calibration
from backend.storage.layer_store import list_layers
from backend.storage.physics_relation_store import save_support_analysis

_SUPPORT_SURFACE_CATEGORIES = {"counter", "desk", "dining table", "floor", "shelf", "table"}
_BASELINE_STEPS = 480
_STAT_STEPS = 120
_REMOVAL_STEPS = 240


class SupportAnalysisError(RuntimeError):
    pass


@dataclass
class ProxyBody:
    layer_id: str
    name: str
    category: str
    vertices: np.ndarray
    faces: np.ndarray
    report: dict
    local_vertices: np.ndarray | None = None
    base_position: np.ndarray | None = None
    extent: float = 0.0
    normalization_scale: float = 1.0
    world_origin: np.ndarray | None = None


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        raise SupportAnalysisError("重力方向无效")
    return vector / length


def _load_proxy_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = PlyData.read(str(path))
    vertex = mesh["vertex"]
    vertices = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    if "face" not in mesh:
        raise SupportAnalysisError(f"物理代理缺少三角面：{path.name}")
    faces = np.asarray([item[0] for item in mesh["face"]], dtype=np.int32)
    if len(vertices) < 4 or len(faces) < 4 or not np.isfinite(vertices).all():
        raise SupportAnalysisError(f"物理代理几何无效：{path.name}")
    return vertices, faces


def _prepare_bodies(
    task_id: str,
    layers: list[dict],
    requested_up: np.ndarray,
    ground_layer_id: str | None,
) -> tuple[list[ProxyBody], np.ndarray, float, np.ndarray]:
    bodies: list[ProxyBody] = []
    for layer in layers:
        layer_id = str(layer["layer_id"])
        try:
            proxy_path, report = generate_physics_proxy(
                task_id,
                layer_id,
                "support_plane" if layer_id == ground_layer_id else "auto",
                requested_up.tolist(),
            )
        except (KeyError, PhysicsProxyError) as exc:
            raise SupportAnalysisError(f"{layer.get('name', layer_id)}：{exc}") from exc
        if not report.get("physics_ready"):
            raise SupportAnalysisError(f"{layer.get('name', layer_id)}：物理代理未通过检查")
        vertices, faces = _load_proxy_mesh(proxy_path)
        bodies.append(
            ProxyBody(
                layer_id=layer_id,
                name=str(layer.get("name") or layer_id),
                category=str(layer.get("category") or ""),
                vertices=vertices,
                faces=faces,
                report=report,
            )
        )
    all_vertices = np.vstack([body.vertices for body in bodies])
    lower, upper = all_vertices.min(axis=0), all_vertices.max(axis=0)
    scene_extent = float(np.max(upper - lower))
    if not np.isfinite(scene_extent) or scene_extent < 1e-7:
        raise SupportAnalysisError("场景空间范围无效")
    world_origin = (lower + upper) * 0.5
    normalization_scale = 2.0 / scene_extent
    for body in bodies:
        normalized = (body.vertices - world_origin) * normalization_scale
        base_position = normalized.mean(axis=0)
        body.local_vertices = normalized - base_position
        body.base_position = base_position
        body.extent = max(float(np.max(np.ptp(normalized, axis=0))), 1e-5)
        body.normalization_scale = normalization_scale
        body.world_origin = world_origin
    return bodies, requested_up, normalization_scale, world_origin


def _pybullet():
    try:
        import pybullet as bullet
    except ImportError as exc:
        raise SupportAnalysisError("PyBullet 未安装，请安装 requirements.txt") from exc
    return bullet


def _create_body(bullet, body: ProxyBody, dynamic: bool, client: int) -> int:
    assert body.local_vertices is not None and body.base_position is not None
    proxy_type = body.report.get("proxy_type")
    geometry = body.report.get("geometry", {})
    orientation = [0.0, 0.0, 0.0, 1.0]
    base_position = body.base_position
    if proxy_type in {"obb", "support_plane"} and geometry.get("half_extents"):
        half_extents = np.asarray(geometry["half_extents"], dtype=np.float64)
        axes = np.asarray(geometry["axes"], dtype=np.float64)
        if proxy_type == "support_plane":
            plane_origin = np.asarray(geometry["plane_origin"], dtype=np.float64)
            normal = np.asarray(geometry["plane_normal"], dtype=np.float64)
            center = plane_origin - normal * half_extents[2]
        else:
            center = np.asarray(geometry["center"], dtype=np.float64)
        world_origin = body.world_origin if body.world_origin is not None else np.zeros(3)
        base_position = (center - world_origin) * body.normalization_scale
        orientation = Rotation.from_matrix(axes.T).as_quat().tolist()
        shape = bullet.createCollisionShape(
            bullet.GEOM_BOX,
            halfExtents=(half_extents * body.normalization_scale).tolist(),
            physicsClientId=client,
        )
    elif proxy_type == "cylinder" and geometry.get("axis"):
        center = np.asarray(geometry["center"], dtype=np.float64)
        axis = _normalize(np.asarray(geometry["axis"], dtype=np.float64))
        world_origin = body.world_origin if body.world_origin is not None else np.zeros(3)
        base_position = (center - world_origin) * body.normalization_scale
        rotation, _ = Rotation.align_vectors([axis], [[0.0, 0.0, 1.0]])
        orientation = rotation.as_quat().tolist()
        shape = bullet.createCollisionShape(
            bullet.GEOM_CYLINDER,
            radius=float(geometry["radius"]) * body.normalization_scale,
            height=float(geometry["height"]) * body.normalization_scale,
            physicsClientId=client,
        )
    else:
        shape = bullet.createCollisionShape(
            bullet.GEOM_MESH,
            vertices=body.local_vertices.tolist(),
            indices=body.faces.reshape(-1).tolist(),
            physicsClientId=client,
        )
    if shape < 0:
        raise SupportAnalysisError(f"无法创建碰撞体：{body.name}")
    body_id = bullet.createMultiBody(
        baseMass=1.0 if dynamic else 0.0,
        baseCollisionShapeIndex=shape,
        basePosition=base_position.tolist(),
        baseOrientation=orientation,
        physicsClientId=client,
    )
    bullet.changeDynamics(
        body_id,
        -1,
        lateralFriction=0.5,
        restitution=0.0,
        linearDamping=0.04,
        angularDamping=0.04,
        physicsClientId=client,
    )
    return body_id


def _connect_world(bullet, up: np.ndarray) -> int:
    client = bullet.connect(bullet.DIRECT)
    if client < 0:
        raise SupportAnalysisError("无法创建 PyBullet DIRECT 物理世界")
    bullet.setGravity(*(-up * 9.81), physicsClientId=client)
    bullet.setTimeStep(1.0 / 240.0, physicsClientId=client)
    bullet.setPhysicsEngineParameter(
        numSolverIterations=100,
        contactBreakingThreshold=0.002,
        physicsClientId=client,
    )
    return client


def _baseline(
    bodies: list[ProxyBody], subject: ProxyBody, up: np.ndarray
) -> tuple[dict, dict[str, dict]]:
    bullet = _pybullet()
    client = _connect_world(bullet, up)
    try:
        ids: dict[str, int] = {}
        reverse: dict[int, str] = {}
        for body in bodies:
            body_id = _create_body(bullet, body, body.layer_id == subject.layer_id, client)
            ids[body.layer_id] = body_id
            reverse[body_id] = body.layer_id
        subject_id = ids[subject.layer_id]
        initial_position, _ = bullet.getBasePositionAndOrientation(subject_id, physicsClientId=client)
        positions: list[np.ndarray] = []
        linear_speeds: list[float] = []
        angular_speeds: list[float] = []
        evidence = {
            body.layer_id: {"contact_frames": 0, "upward_force_sum": 0.0, "upward_contact_frames": 0}
            for body in bodies
            if body.layer_id != subject.layer_id
        }
        for step in range(_BASELINE_STEPS):
            bullet.stepSimulation(physicsClientId=client)
            if step < _BASELINE_STEPS - _STAT_STEPS:
                continue
            position, _ = bullet.getBasePositionAndOrientation(subject_id, physicsClientId=client)
            linear, angular = bullet.getBaseVelocity(subject_id, physicsClientId=client)
            positions.append(np.asarray(position))
            linear_speeds.append(float(np.linalg.norm(linear)))
            angular_speeds.append(float(np.linalg.norm(angular)))
            frame_forces: dict[str, float] = {}
            frame_upward: set[str] = set()
            for contact in bullet.getContactPoints(bodyA=subject_id, physicsClientId=client):
                supporter_id = reverse.get(int(contact[2]))
                if supporter_id is None or supporter_id == subject.layer_id:
                    continue
                normal = np.asarray(contact[7], dtype=np.float64)
                upward = float(np.dot(normal, up))
                normal_force = max(float(contact[9]), 0.0)
                if normal_force <= 0:
                    continue
                frame_forces[supporter_id] = frame_forces.get(supporter_id, 0.0) + normal_force * max(
                    upward, 0.0
                )
                if upward > 0.5:
                    frame_upward.add(supporter_id)
            for supporter_id, force in frame_forces.items():
                evidence[supporter_id]["contact_frames"] += 1
                evidence[supporter_id]["upward_force_sum"] += force
                if supporter_id in frame_upward:
                    evidence[supporter_id]["upward_contact_frames"] += 1

        final_position, final_orientation = bullet.getBasePositionAndOrientation(
            subject_id, physicsClientId=client
        )
        recent_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
        stable = (
            float(np.mean(linear_speeds)) < 0.01
            and float(np.mean(angular_speeds)) < 0.05
            and recent_displacement < max(0.005, subject.extent * 0.01)
        )
        for item in evidence.values():
            item["contact_ratio"] = item.pop("contact_frames") / _STAT_STEPS
            item["upward_contact_ratio"] = item.pop("upward_contact_frames") / _STAT_STEPS
            item["mean_upward_force"] = item.pop("upward_force_sum") / _STAT_STEPS
            item["weight"] = 9.81
            item["upward_force_ratio"] = item["mean_upward_force"] / 9.81
        state = {
            "stable": stable,
            "initial_position": list(initial_position),
            "settled_position": list(final_position),
            "settled_orientation": list(final_orientation),
            "mean_linear_speed": float(np.mean(linear_speeds)),
            "mean_angular_speed": float(np.mean(angular_speeds)),
            "recent_displacement": recent_displacement,
        }
        return state, evidence
    finally:
        bullet.disconnect(physicsClientId=client)


def _fall_after_removal(
    bodies: list[ProxyBody],
    subject: ProxyBody,
    removed_layer_id: str,
    up: np.ndarray,
    settled_position: list[float],
    settled_orientation: list[float],
) -> float:
    bullet = _pybullet()
    client = _connect_world(bullet, up)
    try:
        subject_id = -1
        for body in bodies:
            if body.layer_id == removed_layer_id:
                continue
            body_id = _create_body(bullet, body, body.layer_id == subject.layer_id, client)
            if body.layer_id == subject.layer_id:
                subject_id = body_id
                bullet.resetBasePositionAndOrientation(
                    body_id,
                    settled_position,
                    settled_orientation,
                    physicsClientId=client,
                )
        if subject_id < 0:
            raise SupportAnalysisError("反事实场景缺少待分析物体")
        start, _ = bullet.getBasePositionAndOrientation(subject_id, physicsClientId=client)
        for _ in range(_REMOVAL_STEPS):
            bullet.stepSimulation(physicsClientId=client)
        end, _ = bullet.getBasePositionAndOrientation(subject_id, physicsClientId=client)
        return max(float(np.dot(np.asarray(start) - np.asarray(end), up)), 0.0)
    finally:
        bullet.disconnect(physicsClientId=client)


def _confidence(contact_ratio: float, force_ratio: float, fall_distance: float, threshold: float) -> float:
    values = [
        np.clip(contact_ratio, 0.0, 1.0),
        np.clip(force_ratio, 0.0, 1.0),
        np.clip(fall_distance / max(threshold * 3.0, 1e-8), 0.0, 1.0),
    ]
    return float(np.mean(values))


def analyze_support_relations(
    task_id: str,
    subject_layer_ids: list[str] | None = None,
    supporter_layer_ids: list[str] | None = None,
) -> dict:
    layers = list_layers(task_id)
    if len(layers) < 2:
        raise SupportAnalysisError("至少需要两个语义图层")
    known = {str(layer["layer_id"]): layer for layer in layers}
    requested_subjects = set(subject_layer_ids or [])
    requested_supporters = set(supporter_layer_ids or [])
    unknown = (requested_subjects | requested_supporters) - set(known)
    if unknown:
        raise SupportAnalysisError(f"图层不存在：{sorted(unknown)[0]}")

    calibration = get_ground_calibration(task_id)
    if calibration is None or not calibration.get("confirmed"):
        raise SupportAnalysisError("请先选择地面、确认法线并完成世界标定")
    requested_up = _normalize(np.asarray(calibration["normal"], dtype=np.float64))
    ground_layer_id = calibration.get("ground_layer_id")
    bodies, calibrated_up, normalization_scale, world_origin = _prepare_bodies(
        task_id, layers, requested_up, ground_layer_id
    )
    body_by_id = {body.layer_id: body for body in bodies}
    if requested_subjects:
        subjects = [body_by_id[layer_id] for layer_id in requested_subjects]
    else:
        subjects = [
            body
            for body in bodies
            if body.layer_id != ground_layer_id
            and body.category.strip().lower() not in _SUPPORT_SURFACE_CATEGORIES
        ]
    if not subjects:
        raise SupportAnalysisError("没有可分析的动态刚体图层")

    relations: list[dict] = []
    subject_states: list[dict] = []
    for subject in subjects:
        state, evidence_by_supporter = _baseline(bodies, subject, calibrated_up)
        subject_states.append({"layer_id": subject.layer_id, **state})
        for supporter_id, evidence in evidence_by_supporter.items():
            if requested_supporters and supporter_id not in requested_supporters:
                continue
            if evidence["contact_ratio"] < 0.8 or evidence["upward_contact_ratio"] < 0.8:
                continue
            if evidence["upward_force_ratio"] < 0.2:
                continue
            fall_distance = _fall_after_removal(
                bodies,
                subject,
                supporter_id,
                calibrated_up,
                state["settled_position"],
                state["settled_orientation"],
            )
            fall_threshold = max(0.02, subject.extent * 0.05)
            supported = bool(state["stable"] and fall_distance > fall_threshold)
            if not supported:
                continue
            evidence.update(
                {
                    "stable": True,
                    "fall_distance_after_removal": fall_distance,
                    "fall_threshold": fall_threshold,
                }
            )
            relations.append(
                {
                    "subject": subject.layer_id,
                    "predicate": "supported_by",
                    "object": supporter_id,
                    "confidence": _confidence(
                        evidence["contact_ratio"],
                        evidence["upward_force_ratio"],
                        fall_distance,
                        fall_threshold,
                    ),
                    "evidence": evidence,
                }
            )

    result = {
        "task_id": task_id,
        "engine": "pybullet-3.2.7",
        "generated_at": datetime.now(UTC).isoformat(),
        "gravity": {
            "up_axis": calibrated_up.tolist(),
            "vector": (-calibrated_up * 9.81).tolist(),
            "source": "confirmed_ground_calibration",
            "calibration_method": calibration.get("method"),
            "ground_layer_id": ground_layer_id,
        },
        "normalization": {
            "scale": normalization_scale,
            "world_origin": world_origin.tolist(),
        },
        "objects": [
            {
                "layer_id": body.layer_id,
                "name": body.name,
                "category": body.category,
                "proxy_type": body.report.get("proxy_type"),
                "physics_ready": body.report.get("physics_ready", False),
            }
            for body in bodies
        ],
        "subject_states": subject_states,
        "relations": relations,
    }
    return save_support_analysis(task_id, result)
