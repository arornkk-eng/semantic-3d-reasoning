"""Run a minimal CPU COLMAP reconstruction and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pycolmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    images = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(images) < 2:
        raise SystemExit("at least two images are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    database = args.output_dir / "database.db"
    sparse = args.output_dir / "sparse"
    sparse.mkdir()

    pycolmap.extract_features(
        database,
        args.image_dir,
        image_names=[path.name for path in images],
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.cpu,
    )
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    reconstructions = pycolmap.incremental_mapping(database, args.image_dir, sparse)

    models = []
    for model_id, reconstruction in reconstructions.items():
        model_path = sparse / str(model_id)
        reconstruction.write(model_path)
        registered = [reconstruction.images[i].name for i in reconstruction.reg_image_ids()]
        models.append({
            "model_id": model_id,
            "registered_images": len(registered),
            "registered_names": registered,
            "points3D": reconstruction.num_points3D(),
            "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
            "mean_observations_per_image": reconstruction.compute_mean_observations_per_reg_image(),
        })

    report = {
        "pycolmap_version": pycolmap.__version__,
        "input_images": len(images),
        "image_names": [path.name for path in images],
        "model_count": len(models),
        "models": models,
        "success": bool(models and max(model["registered_images"] for model in models) >= 2),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
