"""Persistent camera-relative scene-understanding snapshots."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from backend.core.config import LAYER_DIR
from backend.core.schemas import SceneSnapshotCreateRequest


def _directory(task_id: str) -> Path:
    if not task_id.isalnum() or len(task_id) > 64:
        raise ValueError("无效任务 ID")
    return LAYER_DIR / task_id / "_scene-snapshots"


def _path(task_id: str, snapshot_id: str) -> Path:
    if not snapshot_id.isalnum() or len(snapshot_id) > 32:
        raise ValueError("无效快照 ID")
    return _directory(task_id) / f"{snapshot_id}.json"


def list_snapshots(task_id: str) -> list[dict]:
    directory = _directory(task_id)
    if not directory.exists():
        return []
    return sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("*.json")),
        key=lambda item: item["sequence"],
    )


def create_snapshot(task_id: str, request: SceneSnapshotCreateRequest) -> dict:
    directory = _directory(task_id)
    directory.mkdir(parents=True, exist_ok=True)
    existing = list_snapshots(task_id)
    sequence = max((item["sequence"] for item in existing), default=0) + 1
    data = {
        "snapshot_id": uuid.uuid4().hex[:12],
        "task_id": task_id,
        "name": f"视角分析{sequence}",
        "sequence": sequence,
        "created_at": datetime.now(UTC).isoformat(),
        **request.model_dump(),
    }
    _write(_path(task_id, data["snapshot_id"]), data)
    return data


def rename_snapshot(task_id: str, snapshot_id: str, name: str) -> dict:
    path = _path(task_id, snapshot_id)
    if not path.is_file():
        raise KeyError(snapshot_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["name"] = name.strip()
    _write(path, data)
    return data


def delete_snapshot(task_id: str, snapshot_id: str) -> None:
    path = _path(task_id, snapshot_id)
    if not path.is_file():
        raise KeyError(snapshot_id)
    path.unlink()


def snapshots_using_layer(task_id: str, layer_id: str) -> list[dict]:
    return [
        item for item in list_snapshots(task_id)
        if any(obj["layer_id"] == layer_id for obj in item["objects"])
    ]


def delete_snapshots_using_layer(task_id: str, layer_id: str) -> int:
    matches = snapshots_using_layer(task_id, layer_id)
    for item in matches:
        delete_snapshot(task_id, item["snapshot_id"])
    return len(matches)


def _write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
