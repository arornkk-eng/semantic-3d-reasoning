"""file_manager 原子写入、读回、路径穿越防护的测试。"""

from __future__ import annotations

import pytest

from backend.core import config as core_config
from backend.storage import file_manager


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    task_dir = tmp_path / "tasks"
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    for d in (task_dir, upload_dir, output_dir):
        d.mkdir()
    monkeypatch.setattr(core_config, "TASK_DIR", task_dir)
    monkeypatch.setattr(core_config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(core_config, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(file_manager, "TASK_DIR", task_dir)
    monkeypatch.setattr(file_manager, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(file_manager, "OUTPUT_DIR", output_dir)
    return {"task": task_dir, "upload": upload_dir, "output": output_dir}


def test_save_and_get_roundtrip(tmp_dirs):
    file_manager.save_task_meta("abc", {"task_id": "abc", "status": "waiting"})
    loaded = file_manager.get_task_meta("abc")
    assert loaded["task_id"] == "abc"
    assert loaded["status"] == "waiting"
    assert "updated_at" in loaded  # save_task_meta 应自动写入时间戳


def test_atomic_write_leaves_no_tmp(tmp_dirs):
    file_manager.save_task_meta("abc", {"task_id": "abc"})
    assert not list(tmp_dirs["task"].glob("*.tmp"))


def test_get_missing_returns_none(tmp_dirs):
    assert file_manager.get_task_meta("nope") is None


def test_output_path_traversal_blocked(tmp_dirs):
    out = tmp_dirs["output"] / "abc"
    out.mkdir()
    (out / "scene.ply").write_text("x", encoding="utf-8")

    # 正常文件可访问
    assert file_manager.get_output_path("abc", "scene.ply") is not None
    # 相对路径穿越被拒绝
    assert file_manager.get_output_path("abc", "../secret.txt") is None
    # 绝对路径穿越被拒绝
    assert file_manager.get_output_path("abc", "/etc/passwd") is None


def test_list_all_task_ids(tmp_dirs):
    # task_id 为 12 位 hex（与运行时 uuid.hex[:12] 一致）
    (tmp_dirs["task"] / "a1b2c3d4e5f6.json").write_text("{}", encoding="utf-8")
    (tmp_dirs["upload"] / "f6e5d4c3b2a1").mkdir()
    ids = file_manager.list_all_task_ids()
    assert "a1b2c3d4e5f6" in ids
    assert "f6e5d4c3b2a1" in ids
