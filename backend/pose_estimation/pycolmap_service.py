"""Recover input camera poses after ZipSplat has produced its model."""

from __future__ import annotations

import json
import logging

import numpy as np

from backend.core.config import OUTPUT_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)


def estimate_camera_poses(task_id: str) -> dict:
    """Run CPU COLMAP and save registered cameras without blocking reconstruction."""
    import pycolmap

    image_dir = UPLOAD_DIR / task_id
    output_dir = OUTPUT_DIR / task_id
    pose_dir = output_dir / "colmap"
    pose_dir.mkdir(parents=True, exist_ok=True)
    database = pose_dir / "database.db"
    sparse_dir = pose_dir / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    image_names = sorted(
        path.name
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if len(image_names) < 2:
        raise ValueError("相机位姿估计至少需要两张图片")

    pycolmap.extract_features(
        database,
        image_dir,
        image_names=image_names,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    reconstructions = pycolmap.incremental_mapping(database, image_dir, sparse_dir)
    if not reconstructions:
        raise RuntimeError("pycolmap 未能建立稀疏模型")

    reconstruction = max(reconstructions.values(), key=lambda item: item.num_reg_images())
    reconstruction.write(sparse_dir / "0")
    cameras = []
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.images[image_id]
        camera = reconstruction.cameras[image.camera_id]
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = image.cam_from_world().matrix()
        cameras.append(
            {
                "image": image.name,
                "camera_id": image.camera_id,
                "model": camera.model.name,
                "width": camera.width,
                "height": camera.height,
                "params": camera.params.tolist(),
                "world_to_camera": world_to_camera.tolist(),
                "camera_to_world": np.linalg.inv(world_to_camera).tolist(),
            }
        )

    report = {
        "status": "completed",
        "input_images": len(image_names),
        "registered_images": len(cameras),
        "points3D": reconstruction.num_points3D(),
        "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
        "cameras": cameras,
        "coordinate_system": "colmap",
        "aligned_to_zipsplat": False,
    }
    (output_dir / "cameras.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "任务 %s 位姿完成: %d/%d 张", task_id, len(cameras), len(image_names)
    )
    return report
