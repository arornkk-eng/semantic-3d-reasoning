"""Compare view count and exposure alignment with one loaded ZipSplat model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SIX_VIEW_NAMES = [
    "IMG_1679.jpg",
    "IMG_1683.jpg",
    "IMG_1681.jpg",
    "IMG_1677.jpg",
    "IMG_1678.jpg",
    "IMG_1680.jpg",
]


def align_exposure(images: list[torch.Tensor]) -> list[torch.Tensor]:
    medians = torch.stack([image.mean(dim=0).median() for image in images])
    target = medians.median()
    return [(image * (target / median.clamp_min(1e-4))).clamp(0, 1) for image, median in zip(images, medians, strict=True)]


def summarize(gaussians) -> dict:
    means = gaussians.means.detach().cpu().numpy()
    scales = gaussians.scales.detach().cpu().numpy().max(axis=1)
    alpha = gaussians.opacities.detach().cpu().numpy()
    bbox = means.max(axis=0) - means.min(axis=0)
    return {
        "count": int(gaussians.num_gaussians),
        "bbox_diagonal": round(float(np.linalg.norm(bbox)), 8),
        "alpha_p50": round(float(np.median(alpha)), 8),
        "scale_p50": round(float(np.median(scales)), 8),
        "scale_p99": round(float(np.quantile(scales, 0.99)), 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "ZipSplat-main"))
    from zipsplat import ZipSplat, load_image

    paths = sorted(args.input_dir.glob("*.jpg"))
    by_name = {path.name: path for path in paths}
    selected = [by_name[name] for name in SIX_VIEW_NAMES]
    variants = {"6_views": selected, "9_views": paths}
    model = ZipSplat(weights="zipsplat").cuda().eval()
    report = {}
    with torch.no_grad():
        for label, variant_paths in variants.items():
            images = [load_image(path) for path in variant_paths]
            for exposure in (False, True):
                model_images = align_exposure(images) if exposure else images
                gaussians = model(model_images, compression=1.0)[0]
                key = f"{label}_{'aligned' if exposure else 'original'}"
                report[key] = {
                    "images": [path.name for path in variant_paths],
                    **summarize(gaussians),
                }
                del gaussians
                torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
