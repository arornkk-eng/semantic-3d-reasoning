"""Open3D-style geometric refinement for projected Gaussian selections."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


def refine_gaussian_selection(
    geometry: np.ndarray,
    seed_indices: list[int],
    candidate_indices: list[int],
    scene_radius: float,
) -> dict[str, object]:
    started = time.perf_counter()
    count = geometry.shape[0]
    seeds = np.unique(np.asarray(seed_indices, dtype=np.int64))
    candidates = np.unique(np.asarray(candidate_indices, dtype=np.int64))
    if geometry.shape != (count, 8) or not np.isfinite(geometry).all():
        raise ValueError("geometry 必须是有限的 N×8 float32 数据")
    if not np.isfinite(scene_radius) or scene_radius <= 0:
        raise ValueError("scene_radius 必须大于 0")
    if seeds.size == 0:
        raise ValueError("精细补全需要至少一个种子")
    if np.any(seeds < 0) or np.any(seeds >= count) or np.any(candidates < 0) or np.any(candidates >= count):
        raise ValueError("Gaussian 索引越界")

    positions = geometry[:, :3].astype(np.float64, copy=False)
    colors = np.clip(geometry[:, 3:6].astype(np.float64, copy=False), 0, 1)
    scales = geometry[:, 6].astype(np.float64, copy=False)
    opacity = geometry[:, 7].astype(np.float64, copy=False)
    candidates = candidates[(opacity[candidates] >= 0.05) & (scales[candidates] > 0)]
    candidates = np.setdiff1d(candidates, seeds, assume_unique=True)
    nodes = np.union1d(seeds, candidates)
    if candidates.size == 0:
        return _result(seeds, seeds.size, started)

    seed_scale = float(np.median(scales[seeds][scales[seeds] > 0]))
    min_radius = max(scene_radius * 0.001, seed_scale * 2)
    max_radius = max(min_radius, min(scene_radius * 0.015, seed_scale * 10))
    tree = cKDTree(positions[nodes])
    _, local_neighbors = tree.query(
        positions[seeds],
        k=min(64, len(nodes)),
        distance_upper_bound=max_radius * 2,
    )
    local_neighbors = np.asarray(local_neighbors).reshape(-1)
    local_neighbors = local_neighbors[local_neighbors < len(nodes)]
    nearby_candidates = np.intersect1d(nodes[np.unique(local_neighbors)], candidates)
    candidates = nearby_candidates
    nodes = np.union1d(seeds, candidates)
    if candidates.size == 0:
        return _result(seeds, seeds.size, started)
    tree = cKDTree(positions[nodes])
    node_lookup = {int(index): offset for offset, index in enumerate(nodes)}
    _, neighbor_table = tree.query(
        positions[nodes], k=min(32, len(nodes)), distance_upper_bound=max_radius
    )
    neighbor_table = np.asarray(neighbor_table).reshape((len(nodes), -1))
    normals = _estimate_normals(positions[nodes], neighbor_table)
    accepted = set(map(int, seeds))
    pending = set(map(int, candidates))
    growth_cap = max(1, int(np.ceil(seeds.size * 0.25)))
    added = 0

    for _ in range(2):
        scored: list[tuple[float, int]] = []
        for index in pending:
            local = node_lookup[index]
            support = 0.0
            support_count = 0
            for neighbor_local in neighbor_table[local]:
                if neighbor_local >= len(nodes):
                    continue
                neighbor = int(nodes[neighbor_local])
                if neighbor not in accepted or neighbor == index:
                    continue
                scale_ratio = max(scales[index], scales[neighbor]) / max(
                    min(scales[index], scales[neighbor]), 1e-9
                )
                if scale_ratio > 4 or np.linalg.norm(colors[index] - colors[neighbor]) > 0.35:
                    continue
                dynamic_radius = max(
                    min_radius,
                    min(max_radius, 3 * (scales[index] + scales[neighbor])),
                )
                if np.linalg.norm(positions[index] - positions[neighbor]) > dynamic_radius:
                    continue
                normal_score = abs(float(np.dot(normals[local], normals[neighbor_local])))
                if normal_score < 0.45:
                    continue
                support += normal_score
                support_count += 1
            if support_count >= 2 and support >= 0.9:
                scored.append((support, index))
        if not scored or added >= growth_cap:
            break
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [index for _, index in scored[: growth_cap - added]]
        accepted.update(chosen)
        pending.difference_update(chosen)
        added += len(chosen)

    return _result(np.asarray(sorted(accepted), dtype=np.int64), seeds.size, started)


def _estimate_normals(points: np.ndarray, neighbor_table: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(points)
    for index, point in enumerate(points):
        neighbors = neighbor_table[index]
        neighbors = neighbors[neighbors < len(points)][:16]
        if len(neighbors) < 3:
            normals[index] = (0, 0, 1)
            continue
        local = points[neighbors] - np.mean(points[neighbors], axis=0)
        covariance = local.T @ local
        normal = np.linalg.eigh(covariance)[1][:, 0]
        length = np.linalg.norm(normal)
        normals[index] = normal / length if length > 1e-9 else (0, 0, 1)
    return normals


def _result(indices: np.ndarray, seed_count: int, started: float) -> dict[str, object]:
    return {
        "indices": indices.astype(int).tolist(),
        "seed_count": int(seed_count),
        "added_count": int(len(indices) - seed_count),
        "engine": "open3d-style-scipy",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
    }
