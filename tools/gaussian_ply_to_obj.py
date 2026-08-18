"""Convert cleaned Gaussian centers to a geometric OBJ with Open3D Poisson."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("output_obj", type=Path)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--density-quantile", type=float, default=0.02)
    parser.add_argument("--target-triangles", type=int, default=150_000)
    args = parser.parse_args()

    cloud = o3d.io.read_point_cloud(str(args.input_ply), remove_nan_points=True, remove_infinite_points=True)
    if len(cloud.points) < 100:
        raise ValueError("有效 Gaussian 中心不足，无法生成 Mesh")

    extent = float(np.linalg.norm(cloud.get_axis_aligned_bounding_box().get_extent()))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=max(extent / 100.0, 1e-4), max_nn=32)
    )
    cloud.orient_normals_consistent_tangent_plane(24)
    cloud.normalize_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=args.depth, linear_fit=True
    )
    densities = np.asarray(densities)
    threshold = float(np.quantile(densities, args.density_quantile))
    mesh.remove_vertices_by_mask(densities < threshold)
    mesh = mesh.crop(cloud.get_axis_aligned_bounding_box())
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    triangle_count_before = len(mesh.triangles)
    if triangle_count_before > args.target_triangles:
        mesh = mesh.simplify_quadric_decimation(args.target_triangles)
    mesh.compute_vertex_normals()
    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(args.output_obj), mesh, write_vertex_normals=True):
        raise RuntimeError("OBJ 写入失败")

    report = {
        "source": str(args.input_ply),
        "output": str(args.output_obj),
        "source_gaussians": len(cloud.points),
        "poisson_depth": args.depth,
        "density_quantile": args.density_quantile,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "triangles_before_decimation": triangle_count_before,
        "watertight": mesh.is_watertight(),
        "edge_manifold": mesh.is_edge_manifold(allow_boundary_edges=True),
    }
    report_path = args.output_obj.with_suffix(".obj-report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
