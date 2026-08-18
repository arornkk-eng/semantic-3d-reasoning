"""Convert indexed Gaussian centers from a 3DGS PLY into an Open3D TSDF mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    return extrinsic


def _directions(count: int) -> list[np.ndarray]:
    golden = np.pi * (3.0 - np.sqrt(5.0))
    result = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        angle = index * golden
        result.append(np.array([np.cos(angle) * radius, y, np.sin(angle) * radius]))
    return result


def _render_depth(
    points: np.ndarray,
    extrinsic: np.ndarray,
    resolution: int,
    focal: float,
    splat_pixels: int,
) -> np.ndarray:
    camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    valid = camera[:, 2] > 1e-5
    camera = camera[valid]
    u = np.rint(focal * camera[:, 0] / camera[:, 2] + resolution * 0.5).astype(np.int64)
    v = np.rint(focal * camera[:, 1] / camera[:, 2] + resolution * 0.5).astype(np.int64)
    inside = (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
    flat = np.full(resolution * resolution, np.inf, dtype=np.float32)
    np.minimum.at(flat, v[inside] * resolution + u[inside], camera[inside, 2].astype(np.float32))
    depth = flat.reshape((resolution, resolution))
    if splat_pixels > 1:
        radius = splat_pixels // 2
        padded = np.pad(depth, radius, mode="constant", constant_values=np.inf)
        depth = np.minimum.reduce([
            padded[dy : dy + resolution, dx : dx + resolution]
            for dy in range(splat_pixels)
            for dx in range(splat_pixels)
        ])
    depth[~np.isfinite(depth)] = 0.0
    return np.ascontiguousarray(depth)


def convert(
    source: Path,
    indices_path: Path,
    output: Path,
    views: int,
    resolution: int,
    collision_output: Path | None = None,
    voxel_divisor: float = 160.0,
    collision_triangles: int = 5000,
) -> dict:
    if views < 6:
        raise ValueError("views 必须至少为 6")
    if resolution < 128:
        raise ValueError("resolution 必须至少为 128")
    if not np.isfinite(voxel_divisor) or not 64 <= voxel_divisor <= 512:
        raise ValueError("voxel_divisor 必须位于 64 至 512")
    if collision_triangles < 100:
        raise ValueError("collision_triangles 必须至少为 100")
    cloud = o3d.io.read_point_cloud(str(source), remove_nan_points=True, remove_infinite_points=True)
    all_points = np.asarray(cloud.points)
    indices = np.fromfile(indices_path, dtype="<u4").astype(np.int64)
    if not len(indices):
        raise ValueError("图层没有 Gaussian")
    if indices.max(initial=-1) >= len(all_points):
        raise ValueError("图层索引超出 scene.ply 范围")
    points = np.asarray(all_points[indices], dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 20:
        raise ValueError("图层 Gaussian 少于 20 个，无法稳定生成 Mesh")

    lower, upper = np.percentile(points, [1.0, 99.0], axis=0)
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    filtered = points[inside]
    if len(filtered) >= max(20, len(points) // 2):
        points = filtered
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) * 0.5
    extent = float(np.max(upper - lower))
    if extent <= 1e-7:
        raise ValueError("图层空间范围过小")
    voxel = extent / voxel_divisor
    truncation = voxel * 5.0
    camera_distance = extent * 2.2
    focal = resolution * 0.9
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        resolution, resolution, focal, focal, resolution * 0.5, resolution * 0.5
    )
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel,
        sdf_trunc=truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    color_data = np.empty((resolution, resolution, 3), dtype=np.uint8)
    color_data[:] = (110, 170, 235)
    color = o3d.geometry.Image(color_data)
    integrated = 0
    for direction in _directions(views):
        extrinsic = _look_at(center + direction * camera_distance, center)
        depth_data = _render_depth(points, extrinsic, resolution, focal, splat_pixels=5)
        if np.count_nonzero(depth_data) < 20:
            continue
        depth = o3d.geometry.Image(depth_data)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1.0,
            depth_trunc=camera_distance * 2.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, extrinsic)
        integrated += 1
    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        minimum = max(30, int(len(mesh.triangles) * 0.002))
        remove = np.asarray(counts)[np.asarray(clusters)] < minimum
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if not len(mesh.triangles):
        raise ValueError("TSDF 未提取到有效表面")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False):
        raise RuntimeError("Mesh 文件写入失败")
    collision_vertices = 0
    collision_faces = 0
    if collision_output is not None:
        target = min(collision_triangles, len(mesh.triangles))
        collision = (
            mesh.simplify_quadric_decimation(target)
            if target < len(mesh.triangles)
            else o3d.geometry.TriangleMesh(mesh)
        )
        collision.remove_duplicated_vertices()
        collision.remove_duplicated_triangles()
        collision.remove_degenerate_triangles()
        collision.remove_unreferenced_vertices()
        collision.compute_vertex_normals()
        collision_output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(collision_output), collision, write_ascii=False):
            raise RuntimeError("碰撞候选 Mesh 文件写入失败")
        collision_vertices = len(collision.vertices)
        collision_faces = len(collision.triangles)
    return {
        "engine": "open3d-tsdf",
        "geometry_source": "synthetic-depth-from-gaussian-centers",
        "completion_level": "L2-preview",
        "safe_for_collision": False,
        "safe_for_grasp_proposal": True,
        "source_gaussians": len(indices),
        "used_gaussians": len(points),
        "views": integrated,
        "voxel_size": voxel,
        "sdf_trunc": truncation,
        "extent": extent,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "collision_vertices": collision_vertices,
        "collision_triangles": collision_faces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("indices", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--collision-output", type=Path)
    parser.add_argument("--voxel-divisor", type=float, default=160.0)
    parser.add_argument("--collision-triangles", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(convert(
        args.source,
        args.indices,
        args.output,
        args.views,
        args.resolution,
        args.collision_output,
        args.voxel_divisor,
        args.collision_triangles,
    )))


if __name__ == "__main__":
    main()
