"""Compare ZipSplat post-processing variants on one unfiltered inference result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def summarize(means: np.ndarray, scales: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> dict:
    selected_means = means[mask]
    selected_scales = scales[mask]
    selected_alpha = alpha[mask]
    bbox_min = selected_means.min(axis=0)
    bbox_max = selected_means.max(axis=0)
    max_scale = selected_scales.max(axis=1)
    return {
        "count": int(mask.sum()),
        "retained_pct": round(float(mask.mean() * 100), 3),
        "bbox_diagonal": round(float(np.linalg.norm(bbox_max - bbox_min)), 8),
        "alpha_p50": round(float(np.median(selected_alpha)), 8),
        "scale_p50": round(float(np.median(max_scale)), 8),
        "scale_p99": round(float(np.quantile(max_scale, 0.99)), 8),
        "scale_max": round(float(max_scale.max()), 8),
    }


def scene_mask(means: np.ndarray, alpha: np.ndarray, tail_percent: float) -> np.ndarray:
    alpha_mask = alpha > 0.02
    indices = np.flatnonzero(alpha_mask)
    nearest = cKDTree(means[alpha_mask]).query(means[alpha_mask], k=2, workers=1)[0][:, 1]
    keep = nearest <= np.percentile(nearest, 100 - tail_percent)
    result = np.zeros(len(means), dtype=bool)
    result[indices[keep]] = True
    return result


def object_mask(means: np.ndarray, alpha: np.ndarray, top_n: int) -> tuple[np.ndarray, dict]:
    alpha_mask = alpha > 0.05
    indices = np.flatnonzero(alpha_mask)
    points = means[alpha_mask]
    scene_diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    labels = DBSCAN(eps=scene_diag * 0.03, min_samples=12, n_jobs=1).fit_predict(points)
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    selected = cluster_ids[np.argsort(counts)[::-1][:top_n]] if len(cluster_ids) else []
    keep = np.isin(labels, selected) if len(cluster_ids) else np.ones(len(points), dtype=bool)
    result = np.zeros(len(means), dtype=bool)
    result[indices[keep]] = True
    detail = {
        "clusters": len(cluster_ids),
        "noise_count": int(np.count_nonzero(labels == -1)),
        "largest_clusters": sorted((int(v) for v in counts), reverse=True)[:10],
    }
    return result, detail


def make_mask_preview(means: np.ndarray, variants: dict[str, np.ndarray], output: Path) -> None:
    panel_width, panel_height = 380, 300
    canvas = Image.new("RGB", (panel_width * 2, panel_height * 4), "white")
    low, high = np.quantile(means[:, :2], [0.01, 0.99], axis=0)
    span = np.maximum(high - low, 1e-9)
    for panel, (label, mask) in enumerate(variants.items()):
        offset_x = (panel % 2) * panel_width
        offset_y = (panel // 2) * panel_height
        draw = ImageDraw.Draw(canvas)
        draw.text((offset_x + 10, offset_y + 8), label, fill="black")
        points = means[mask, :2]
        xy = np.clip((points - low) / span, 0, 1)
        xy[:, 0] = offset_x + 10 + xy[:, 0] * (panel_width - 20)
        xy[:, 1] = offset_y + panel_height - 10 - xy[:, 1] * (panel_height - 40)
        for x, y in xy:
            draw.point((int(x), int(y)), fill=(40, 70, 100))
    canvas.save(output, optimize=True)


def infer(repo: Path, input_dir: Path) -> dict[str, torch.Tensor]:
    sys.path.insert(0, str(repo / "ZipSplat-main"))
    from zipsplat import ZipSplat, load_image

    paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No images found: {input_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for raw inference")
    model = ZipSplat(weights="zipsplat").cuda().eval()
    images = [load_image(path) for path in paths]
    with torch.no_grad():
        gaussians = model(images, compression=1.0)[0]
    return {
        "means": gaussians.means.detach().cpu(),
        "scales": gaussians.scales.detach().cpu(),
        "opacities": gaussians.opacities.detach().cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-raw", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = repo / "evaluation" / "experiments" / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_gaussians.pt"
    if args.reuse_raw:
        tensors = torch.load(raw_path, map_location="cpu", weights_only=True)
    else:
        tensors = infer(repo, repo / config["input_dir"])
        torch.save(tensors, raw_path)

    means = tensors["means"].numpy()
    scales = tensors["scales"].numpy()
    alpha = tensors["opacities"].numpy()
    all_points = np.ones(len(means), dtype=bool)
    report = {
        "source": config["name"],
        "raw": summarize(means, scales, alpha, all_points),
        "splat_scale": {},
        "scene_cleanup": {},
        "object_dbscan": {},
    }
    preview_variants = {"raw": all_points}
    for factor in (1.0, 0.4):
        stats = summarize(means, scales * factor, alpha, all_points)
        stats["relative_projected_radius"] = factor
        stats["relative_ellipsoid_volume"] = round(factor**3, 3)
        report["splat_scale"][str(factor)] = stats
    for tail in (1, 5, 10):
        mask = scene_mask(means, alpha, tail)
        report["scene_cleanup"][str(tail)] = summarize(means, scales, alpha, mask)
        preview_variants[f"scene tail {tail}%"] = mask
    alpha_only = alpha > 0.02
    report["scene_cleanup"]["alpha_only"] = summarize(
        means, scales, alpha, alpha_only
    )
    preview_variants["scene alpha only"] = alpha_only
    for top_n in (1, 3, 5):
        mask, detail = object_mask(means, alpha, top_n)
        report["object_dbscan"][str(top_n)] = {
            **summarize(means, scales, alpha, mask),
            **detail,
        }
        preview_variants[f"object top {top_n}"] = mask
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    make_mask_preview(means, preview_variants, output_dir / "mask_comparison.png")
    print(report_path)


if __name__ == "__main__":
    main()
