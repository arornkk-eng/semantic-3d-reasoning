"""Run the production reconstruction path with outputs redirected to evaluation/candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.core.config import (  # noqa: E402
    DEFAULT_NUM_VIEWS,
    SCENE_ALPHA_THRESHOLD,
    SCENE_OUTLIER_PERCENTILE,
    SPLAT_SCALE_FACTOR,
)
from backend.zipsplat_engine import runner  # noqa: E402
from backend.zipsplat_engine.splat_converter import ply_to_splat  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--mode", choices=("object", "scene"), default="scene")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    repo = args.repo.resolve()
    candidate_root = repo / "evaluation" / "candidates"
    runner.OUTPUT_DIR = candidate_root
    result = runner.run_reconstruction(args.task_id, mode=args.mode)
    output_dir = candidate_root / args.task_id
    conversion = ply_to_splat(output_dir / "scene.ply", output_dir / "scene.splat")
    report = {
        "task_id": args.task_id,
        "mode": args.mode,
        "parameters": {
            "num_views": DEFAULT_NUM_VIEWS,
            "scene_alpha_threshold": SCENE_ALPHA_THRESHOLD,
            "scene_outlier_percentile": SCENE_OUTLIER_PERCENTILE,
            "splat_scale_factor": SPLAT_SCALE_FACTOR,
        },
        "reconstruction": result,
        "conversion": conversion,
    }
    (output_dir / "run_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
