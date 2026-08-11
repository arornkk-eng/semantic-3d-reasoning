"""TaskQueue 取消 / 恢复逻辑的单元测试。

这些测试把任务元数据目录重定向到临时目录，避免污染真实的 data/。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.core import queue_manager
from backend.core.queue_manager import TaskQueue
from backend.storage import file_manager


@pytest.fixture
def tmp_task_dir(tmp_path, monkeypatch):
    """将任务元数据目录重定向到临时目录。"""
    monkeypatch.setattr(queue_manager, "TASK_DIR", tmp_path)
    monkeypatch.setattr(file_manager, "TASK_DIR", tmp_path)
    return tmp_path


def _write_meta(task_id: str, status: str, tmp_dir: Path) -> None:
    meta = {
        "task_id": task_id,
        "status": status,
        "type": "image",
        "mode": "object",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "input": {"file_count": 3, "filenames": ["a.jpg"]},
        "output": None,
        "error": None,
    }
    (tmp_dir / f"{task_id}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def test_enqueue_dequeue(tmp_task_dir):
    q = TaskQueue()
    q.enqueue("t1")
    assert q.size() == 1
    assert q.dequeue(timeout=1.0) == "t1"
    assert q.size() == 0


def test_dequeue_empty_returns_none(tmp_task_dir):
    q = TaskQueue()
    assert q.dequeue(timeout=0.2) is None


def test_cancel_waiting_task(tmp_task_dir):
    _write_meta("t1", "waiting", tmp_task_dir)
    q = TaskQueue()
    q.enqueue("t1")
    assert q.cancel("t1") is True
    # 取消后出队应跳过该任务
    assert q.dequeue(timeout=0.2) is None


def test_cancel_non_waiting_returns_false(tmp_task_dir):
    _write_meta("t1", "running", tmp_task_dir)
    q = TaskQueue()
    assert q.cancel("t1") is False


def test_recover_pending(tmp_task_dir):
    _write_meta("waiting1", "waiting", tmp_task_dir)
    _write_meta("running1", "running", tmp_task_dir)
    _write_meta("done1", "completed", tmp_task_dir)
    q = TaskQueue()
    recovered = q.recover_pending()
    assert recovered == 2
    # running 应被重置为 waiting 并重新入队
    running_meta = json.loads((tmp_task_dir / "running1.json").read_text(encoding="utf-8"))
    assert running_meta["status"] == "waiting"
