"""Robustly isolate the observed 3D body from contaminated semantic Gaussians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


class ObjectCleaningError(RuntimeError):
    pass


@dataclass(frozen=True)
class CleanedGaussians:
    positions: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    report: dict


def _extent(points: np.ndarray, lower: float = 0.0, upper: float = 100.0) -> np.ndarray:
    bounds = np.percentile(points, [lower, upper], axis=0)
    return np.maximum(bounds[1] - bounds[0], 0.0)


def _distance_to_bounds(point: np.ndarray, points: np.ndarray) -> float:
    lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.linalg.norm(delta))


def clean_object_gaussians(
    positions: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
    observation_anchor: np.ndarray | None = None,
) -> CleanedGaussians:
    """Select a deterministic dense component, preferring the observed depth anchor."""
    count = len(positions)
    if count < 20:
        raise ObjectCleaningError("有效 Gaussian 少于 20 个")

    sample_indices = np.linspace(0, count - 1, min(count, 20_000), dtype=np.int64)
    sample = positions[sample_indices]
    neighbor_count = min(8, len(sample))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(sample)
    distances = neighbors.kneighbors(return_distance=True)[0][:, -1]
    positive = distances[np.isfinite(distances) & (distances > 1e-9)]
    full_extent = _extent(positions)
    scene_span = max(float(np.max(full_extent)), 1e-7)
    if not len(positive):
        raise ObjectCleaningError("Gaussian 缺少可分辨的三维邻域")
    epsilon = max(float(np.percentile(positive, 70.0)) * 1.8, scene_span * 0.001)
    min_samples = max(6, min(16, len(sample) // 500))
    labels = DBSCAN(eps=epsilon, min_samples=min_samples, n_jobs=-1).fit_predict(sample)

    candidates: list[tuple[tuple[float, float, float, int], int, np.ndarray]] = []
    for label in sorted(set(labels) - {-1}):
        member_sample = np.flatnonzero(labels == label)
        if len(member_sample) < 20:
            continue
        member_points = sample[member_sample]
        robust_extent = _extent(member_points, 1.0, 99.0)
        diagonal = max(float(np.linalg.norm(robust_extent)), epsilon)
        local_distance = float(np.median(distances[member_sample]))
        if observation_anchor is not None and np.isfinite(observation_anchor).all():
            anchor_distance = _distance_to_bounds(observation_anchor, member_points) / diagonal
        else:
            anchor_distance = 0.0
        rank = (
            anchor_distance,
            local_distance / max(epsilon, 1e-9),
            -float(np.log1p(len(member_sample))),
            int(label),
        )
        candidates.append((rank, label, member_sample))
    if not candidates:
        raise ObjectCleaningError("未找到包含至少 20 个 Gaussian 的连续主体簇")

    _, selected_label, selected_sample = min(candidates, key=lambda item: item[0])
    seed_points = sample[selected_sample]
    seed_lower, seed_upper = np.percentile(seed_points, [0.5, 99.5], axis=0)
    margin = max(epsilon * 1.5, scene_span * 0.002)
    selected = np.all(
        (positions >= seed_lower - margin) & (positions <= seed_upper + margin), axis=1
    )
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) < 20:
        raise ObjectCleaningError("主体簇清理后 Gaussian 少于 20 个")

    selected_points = positions[selected_indices]
    trim_lower, trim_upper = np.percentile(selected_points, [1.0, 99.0], axis=0)
    central = np.all((selected_points >= trim_lower) & (selected_points <= trim_upper), axis=1)
    if int(central.sum()) >= 20:
        selected_indices = selected_indices[central]
        selected_points = positions[selected_indices]

    retained_ratio = float(len(selected_indices) / count)
    clean_extent = _extent(selected_points, 1.0, 99.0)
    full_diagonal = max(float(np.linalg.norm(full_extent)), 1e-9)
    clean_diagonal = max(float(np.linalg.norm(clean_extent)), 1e-9)
    warnings: list[dict] = []
    if retained_ratio < 0.5:
        warnings.append(
            {
                "code": "SEMANTIC_GAUSSIAN_CONTAMINATION",
                "severity": "warning",
                "message": "多数语义 Gaussian 不属于选中的连续主体簇",
            }
        )
    if full_diagonal / clean_diagonal > 3.0:
        warnings.append(
            {
                "code": "OBJECT_EXTENT_OUTLIERS",
                "severity": "warning",
                "message": "原始三维范围被远离主体的 Gaussian 显著放大",
            }
        )
    report = {
        "schema_version": "1.0",
        "status": "warning" if warnings else "pass",
        "source_count": count,
        "retained_count": len(selected_indices),
        "retained_ratio": retained_ratio,
        "cluster_count": len(candidates),
        "selected_cluster": int(selected_label),
        "epsilon": epsilon,
        "min_samples": min_samples,
        "anchor_used": observation_anchor is not None,
        "extent_full": full_extent.tolist(),
        "extent_robust": clean_extent.tolist(),
        "extent_reduction_ratio": full_diagonal / clean_diagonal,
        "warnings": warnings,
    }
    return CleanedGaussians(
        positions=positions[selected_indices],
        scales=scales[selected_indices],
        rotations=rotations[selected_indices],
        report=report,
    )
