"""Generate a deterministic reconstruction baseline report from images and a 3DGS PLY."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from plyfile import PlyData
from scipy.spatial import cKDTree

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SH_C0 = 0.28209479177387814


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    points = np.quantile(values, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        key: round(float(value), 8)
        for key, value in zip(
            ("min", "p50", "p90", "p95", "p99", "max"), points, strict=True
        )
    }


def read_gaussians(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"]
    means = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    log_scales = np.column_stack(
        (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])
    ).astype(np.float64)
    opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    colors = np.column_stack((vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]))
    colors = np.clip((colors * SH_C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    return means, np.exp(log_scales), 1.0 / (1.0 + np.exp(-opacity)), colors


def calculate_metrics(means: np.ndarray, scales: np.ndarray, alpha: np.ndarray) -> dict:
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    nearest = cKDTree(means).query(means, k=2, workers=1)[0][:, 1]
    max_scale = scales.max(axis=1)
    return {
        "gaussian_count": len(means),
        "bbox_min": bbox_min.round(8).tolist(),
        "bbox_max": bbox_max.round(8).tolist(),
        "bbox_diagonal": round(diagonal, 8),
        "alpha": quantiles(alpha),
        "alpha_below_0_02": int(np.count_nonzero(alpha <= 0.02)),
        "scale_max_axis": quantiles(max_scale),
        "nearest_neighbor_distance": quantiles(nearest),
    }


def image_tile(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail(size)
        tile = Image.new("RGB", size, "white")
        tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        return tile


def projection(
    means: np.ndarray, colors: np.ndarray, axes: tuple[int, int], size: tuple[int, int]
) -> Image.Image:
    width, height = size
    canvas = Image.new("RGB", size, (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    sample = np.linspace(0, len(means) - 1, min(len(means), 30000), dtype=int)
    points = means[sample][:, axes]
    rgb = colors[sample]
    low, high = np.quantile(points, [0.01, 0.99], axis=0)
    span = np.maximum(high - low, 1e-9)
    xy = np.clip((points - low) / span, 0, 1)
    xy[:, 0] = 12 + xy[:, 0] * (width - 24)
    xy[:, 1] = height - 12 - xy[:, 1] * (height - 24)
    order = np.argsort(means[sample, 3 - axes[0] - axes[1]])
    for index in order:
        x, y = xy[index]
        draw.point((int(x), int(y)), fill=tuple(int(v) for v in rgb[index]))
    return canvas


def make_preview(
    image_paths: list[Path], means: np.ndarray, colors: np.ndarray, output: Path
) -> None:
    width = 1200
    canvas = Image.new("RGB", (width, 760), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), "Fixed inputs", fill="black")
    tile_width = min(360, (width - 40) // max(1, len(image_paths)))
    for index, path in enumerate(image_paths):
        tile = image_tile(path, (tile_width - 8, 260))
        canvas.paste(tile, (20 + index * tile_width, 38))
        draw.text((20 + index * tile_width, 302), path.name[:42], fill="black")
    labels_axes = (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2)))
    for index, (label, axes) in enumerate(labels_axes):
        x = 20 + index * 390
        draw.text((x, 342), f"Point cloud {label}", fill="black")
        canvas.paste(projection(means, colors, axes, (370, 370)), (x, 368))
    canvas.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_dir = repo / config["input_dir"]
    ply_path = repo / config["ply_path"]
    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No input images found: {input_dir}")
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)

    means, scales, alpha, colors = read_gaussians(ply_path)
    output_dir = repo / "evaluation" / "baselines" / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "inputs": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in images
        ],
        "output": {
            "name": ply_path.name,
            "bytes": ply_path.stat().st_size,
            "sha256": sha256(ply_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(calculate_metrics(means, scales, alpha), indent=2) + "\n", encoding="utf-8"
    )
    make_preview(images, means, colors, output_dir / "preview.png")
    print(output_dir)


if __name__ == "__main__":
    main()
