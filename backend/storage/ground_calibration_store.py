"""Persistent user-confirmed ground and gravity calibration."""

import json
from pathlib import Path

from backend.core.config import LAYER_DIR


def _path(task_id: str) -> Path:
    if not task_id.isalnum() or len(task_id) > 64:
        raise ValueError("无效任务 ID")
    return LAYER_DIR / task_id / "_physics" / "ground-calibration.json"


def save_ground_calibration(task_id: str, calibration: dict) -> dict:
    path = _path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return calibration


def get_ground_calibration(task_id: str) -> dict | None:
    path = _path(task_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def delete_ground_calibration(task_id: str) -> bool:
    path = _path(task_id)
    if not path.is_file():
        return False
    path.unlink()
    return True
