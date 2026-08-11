"""Run one real SAM 2.1 Tiny point-prompt prediction for environment verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.segmentation.service import segmentation_service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metadata = {
        "task_id": "smoke",
        "source_ply": "scene.ply",
        "viewport_width": 1000,
        "viewport_height": 1000,
        "view_matrix": [0.0] * 16,
        "projection_matrix": [0.0] * 16,
    }
    session = segmentation_service.create(args.image.read_bytes(), metadata)
    try:
        result = segmentation_service.predict(
            session.session_id,
            [
                {"x": 0.5, "y": 0.5, "label": 1},
                {"x": 0.05, "y": 0.05, "label": 0},
            ],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(result.mask_png or b"")
        print(json.dumps({"score": result.score, "bbox": result.bbox}))
    finally:
        segmentation_service.close(session.session_id)


if __name__ == "__main__":
    main()
