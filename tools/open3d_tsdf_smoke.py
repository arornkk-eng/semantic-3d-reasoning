"""Generate synthetic sphere RGB-D views and verify the TSDF pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from open3d_tsdf import CV_FROM_GL, fuse_bundle
from PIL import Image


def _look_at_cv(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def main() -> None:
    root = Path("evaluation/open3d-tsdf/synthetic")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    width = height = 160
    fx = fy = 145.0
    near, far = 0.1, 8.0
    yy, xx = np.mgrid[:height, :width]
    dirs_camera = np.stack([(xx - width / 2) / fx, (yy - height / 2) / fy, np.ones_like(xx)], axis=-1)
    views = []
    count = 20
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radius = np.sqrt(1.0 - y * y)
        angle = index * golden
        eye = 3.0 * np.array([np.cos(angle) * radius, y, np.sin(angle) * radius])
        extrinsic_cv = _look_at_cv(eye, np.zeros(3))
        rotation = extrinsic_cv[:3, :3]
        directions = dirs_camera @ rotation
        origins = np.broadcast_to(eye, directions.shape)
        a = np.sum(directions * directions, axis=-1)
        b = 2.0 * np.sum(origins * directions, axis=-1)
        discriminant = b * b - 4.0 * a * (np.sum(origins * origins, axis=-1) - 1.0)
        hit = discriminant >= 0.0
        depth = np.full((height, width), np.nan, dtype=np.float32)
        depth[hit] = ((-b[hit] - np.sqrt(discriminant[hit])) / (2.0 * a[hit])).astype(np.float32)
        points = origins + directions * np.nan_to_num(depth)[..., None]
        color = np.zeros((height, width, 3), dtype=np.uint8)
        color[hit] = np.clip((points[hit] + 1.0) * 127.5, 0, 255).astype(np.uint8)
        normalized = (depth - near) / (far - near)
        color_name = f"color-{index}.png"
        depth_name = f"depth-{index}.f32"
        Image.fromarray(color).save(root / color_name)
        normalized.astype("<f4").tofile(root / depth_name)
        projection = np.zeros((4, 4))
        projection[0, 0] = 2.0 * fx / width
        projection[1, 1] = 2.0 * fy / height
        projection[2, 2] = -(far + near) / (far - near)
        projection[2, 3] = -2.0 * far * near / (far - near)
        projection[3, 2] = -1.0
        view_gl = CV_FROM_GL @ extrinsic_cv
        views.append({
            "color": color_name,
            "depth": depth_name,
            "width": width,
            "height": height,
            "near": near,
            "far": far,
            "projection": "perspective",
            "view_matrix": view_gl.flatten(order="F").tolist(),
            "projection_matrix": projection.flatten(order="F").tolist(),
        })
    (root / "metadata.json").write_text(json.dumps({"views": views}), encoding="utf-8")
    output = root.parent / "sphere-tsdf.ply"
    result = fuse_bundle(root, output, voxel_size=0.035, sdf_trunc=0.14, min_triangles=30)
    if result["vertices"] < 1000 or result["triangles"] < 1000:
        raise RuntimeError(f"unexpectedly sparse TSDF mesh: {result}")
    extent = np.asarray(result["bounds_max"]) - np.asarray(result["bounds_min"])
    if np.any(extent < 1.7) or np.any(extent > 2.3):
        raise RuntimeError(f"unexpected TSDF bounds: {result}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
