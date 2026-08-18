"""Fuse editor RGB-D captures into a triangle mesh with Open3D TSDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image

CV_FROM_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def _matrix(values: list[float]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError("camera matrix must contain 16 values")
    return np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")


def _read_f32(path: Path, width: int, height: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = width * height
    if values.size != expected:
        raise ValueError(f"{path}: expected {expected} floats, got {values.size}")
    return values.reshape((height, width))


def _intrinsic(view: dict[str, Any]) -> o3d.camera.PinholeCameraIntrinsic:
    width = int(view["width"])
    height = int(view["height"])
    projection = _matrix(view["projection_matrix"])
    if view.get("projection", "perspective") != "perspective":
        raise ValueError("TSDF prototype currently supports perspective cameras only")
    fx = abs(projection[0, 0]) * width * 0.5
    fy = abs(projection[1, 1]) * height * 0.5
    cx = width * 0.5
    cy = height * 0.5
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def _load_frame(
    root: Path,
    view: dict[str, Any],
    coverage_threshold: float,
) -> tuple[o3d.geometry.RGBDImage, o3d.camera.PinholeCameraIntrinsic, np.ndarray]:
    width = int(view["width"])
    height = int(view["height"])
    color = np.asarray(Image.open(root / view["color"]).convert("RGB"), dtype=np.uint8)
    if color.shape[:2] != (height, width):
        raise ValueError(f"{view['color']}: color dimensions do not match metadata")

    normalized_depth = _read_f32(root / view["depth"], width, height)
    near = float(view["near"])
    far = float(view["far"])
    valid = np.isfinite(normalized_depth) & (normalized_depth >= 0.0) & (normalized_depth <= 1.0)
    if view.get("coverage"):
        valid &= _read_f32(root / view["coverage"], width, height) >= coverage_threshold
    if view.get("mask"):
        mask = np.asarray(Image.open(root / view["mask"]).convert("L"), dtype=np.uint8)
        if mask.shape != (height, width):
            raise ValueError(f"{view['mask']}: mask dimensions do not match metadata")
        valid &= mask >= int(view.get("mask_threshold", 128))

    depth = np.zeros((height, width), dtype=np.float32)
    depth[valid] = near + normalized_depth[valid] * (far - near)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0,
        depth_trunc=float(view.get("depth_trunc", far)),
        convert_rgb_to_intensity=False,
    )
    world_to_camera_gl = _matrix(view["view_matrix"])
    return rgbd, _intrinsic(view), CV_FROM_GL @ world_to_camera_gl


def fuse_bundle(
    bundle: Path,
    output: Path,
    voxel_size: float,
    sdf_trunc: float,
    coverage_threshold: float = 0.08,
    min_triangles: int = 0,
) -> dict[str, Any]:
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    views = metadata.get("views", [])
    if not views:
        raise ValueError("metadata.json contains no views")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for view in views:
        rgbd, intrinsic, extrinsic = _load_frame(bundle, view, coverage_threshold)
        volume.integrate(rgbd, intrinsic, extrinsic)

    mesh = volume.extract_triangle_mesh()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if min_triangles and len(mesh.triangles):
        clusters, counts, _ = mesh.cluster_connected_triangles()
        keep = np.asarray(counts)[np.asarray(clusters)] >= min_triangles
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_ascii=False):
        raise RuntimeError(f"failed to write {output}")
    bounds = mesh.get_axis_aligned_bounding_box()
    result = {
        "views": len(views),
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "bounds_min": bounds.min_bound.tolist() if len(mesh.vertices) else None,
        "bounds_max": bounds.max_bound.tolist() if len(mesh.vertices) else None,
        "output": str(output.resolve()),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size", type=float, required=True)
    parser.add_argument("--sdf-trunc", type=float)
    parser.add_argument("--coverage-threshold", type=float, default=0.08)
    parser.add_argument("--min-triangles", type=int, default=0)
    args = parser.parse_args()
    result = fuse_bundle(
        args.bundle,
        args.output,
        args.voxel_size,
        args.sdf_trunc or args.voxel_size * 4.0,
        args.coverage_threshold,
        args.min_triangles,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
