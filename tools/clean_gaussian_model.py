"""Clean a Gaussian PLY with robust outlier removal and edge-aware smoothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def _robust_outlier_mask(points: np.ndarray, neighbors: int) -> tuple[np.ndarray, dict]:
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(neighbors + 1, len(points)), workers=-1)
    mean_distance = distances[:, 1:].mean(axis=1)
    median = float(np.median(mean_distance))
    mad = float(np.median(np.abs(mean_distance - median)))
    robust_sigma = max(1.4826 * mad, np.finfo(np.float32).eps)
    threshold = median + 3.5 * robust_sigma

    extent = np.ptp(points, axis=0)
    voxel_size = max(float(np.linalg.norm(extent)) / 256.0, np.finfo(np.float32).eps)
    cells = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    local_density = counts[inverse]

    # Statistical isolation is primary. Empty voxel support strengthens the
    # decision only in the sparse tail and avoids deleting thin valid edges.
    keep = mean_distance <= threshold
    sparse_cutoff = median + 2.5 * robust_sigma
    keep &= ~((local_density == 1) & (mean_distance > sparse_cutoff))
    return keep, {
        "knn_median": median,
        "knn_mad": mad,
        "knn_threshold": threshold,
        "voxel_size": voxel_size,
    }


def _estimate_normals(points: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.empty_like(points)
    chunk = 4096
    for start in range(0, len(points), chunk):
        ids = indices[start : start + chunk]
        neighbors = points[ids]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered)
        _, vectors = np.linalg.eigh(covariance)
        normals[start : start + chunk] = vectors[:, :, 0]
    return normals


def _edge_aware_smooth(points: np.ndarray, neighbors: int) -> tuple[np.ndarray, dict]:
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(neighbors + 1, len(points)), workers=-1)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    normals = _estimate_normals(points, indices)
    sigma_spatial = max(float(np.median(distances[:, -1])) * 0.5, 1e-8)
    sigma_normal = 0.18
    result = points.copy()
    chunk = 2048
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        ids = indices[start:stop]
        delta = points[ids] - points[start:stop, None, :]
        spatial = np.exp(-(distances[start:stop] ** 2) / (2.0 * sigma_spatial**2))
        alignment = np.abs(np.einsum("nkj,nj->nk", normals[ids], normals[start:stop]))
        normal_weight = np.exp(-((1.0 - alignment) ** 2) / (2.0 * sigma_normal**2))
        weights = spatial * normal_weight
        signed = np.einsum("nkj,nj->nk", delta, normals[start:stop])
        displacement = (weights * signed).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-8)
        result[start:stop] += 0.35 * displacement[:, None] * normals[start:stop]
    return result, {
        "neighbors": int(indices.shape[1]),
        "sigma_spatial": sigma_spatial,
        "sigma_normal": sigma_normal,
        "strength": 0.35,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--gaussians", type=Path)
    parser.add_argument("--neighbors", type=int, default=24)
    args = parser.parse_args()

    ply = PlyData.read(args.input_ply)
    vertex = ply["vertex"].data
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    keep, outlier_report = _robust_outlier_mask(points, args.neighbors)
    kept_indices = np.flatnonzero(keep)
    clean_points, smooth_report = _edge_aware_smooth(points[keep], args.neighbors)

    clean_vertex = vertex[keep].copy()
    clean_vertex["x"] = clean_points[:, 0]
    clean_vertex["y"] = clean_points[:, 1]
    clean_vertex["z"] = clean_points[:, 2]
    output_ply = args.output_prefix.with_suffix(".ply")
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(clean_vertex, "vertex")], text=False).write(output_ply)

    output_tensor = None
    if args.gaussians:
        tensors = torch.load(args.gaussians, map_location="cpu", weights_only=True)
        index_tensor = torch.from_numpy(kept_indices.astype(np.int64))
        cleaned = {
            key: value.index_select(0, index_tensor) if torch.is_tensor(value) else value
            for key, value in tensors.items()
        }
        cleaned["means"] = torch.from_numpy(clean_points).to(dtype=tensors["means"].dtype)
        output_tensor = args.output_prefix.with_name(args.output_prefix.name + "-gaussians.pt")
        torch.save(cleaned, output_tensor)

    report = {
        "source": str(args.input_ply),
        "input_gaussians": len(points),
        "output_gaussians": len(clean_points),
        "removed_outliers": int((~keep).sum()),
        "removed_ratio": float((~keep).mean()),
        "outlier_filter": outlier_report,
        "smoothing": smooth_report,
        "outputs": {
            "ply": str(output_ply),
            "gaussians": str(output_tensor) if output_tensor else None,
        },
    }
    report_path = args.output_prefix.with_name(args.output_prefix.name + "-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
