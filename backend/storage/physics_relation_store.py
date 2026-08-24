"""Persistent world-space physical relation analysis."""

import json
from pathlib import Path

from backend.core.config import LAYER_DIR


def _directory(task_id: str) -> Path:
    if not task_id.isalnum() or len(task_id) > 64:
        raise ValueError("无效任务 ID")
    return LAYER_DIR / task_id / "_physics"


def save_support_analysis(task_id: str, result: dict) -> dict:
    directory = _directory(task_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "support-analysis.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def get_support_analysis(task_id: str) -> dict | None:
    path = _directory(task_id) / "support-analysis.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
