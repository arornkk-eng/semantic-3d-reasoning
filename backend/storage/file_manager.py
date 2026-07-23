"""文件管理：上传保存、任务元数据读写、输出文件管理。"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from backend.core.config import UPLOAD_DIR, TASK_DIR, OUTPUT_DIR


def _tz_now() -> str:
    """返回 ISO8601 格式的 UTC 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ---- 上传 ----

def save_uploads(task_id: str, files: list[UploadFile]) -> Path:
    """将上传的文件保存到 data/uploads/{task_id}/，返回保存目录。"""
    upload_dir = UPLOAD_DIR / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        file_path = upload_dir / f.filename
        with open(file_path, "wb") as out:
            f.file.seek(0)
            shutil.copyfileobj(f.file, out)
    return upload_dir


# ---- 任务元数据 ----

def get_task_meta(task_id: str) -> Optional[dict]:
    """读取任务元数据，不存在则返回 None。"""
    meta_path = TASK_DIR / f"{task_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_task_meta(task_id: str, meta: dict) -> None:
    """原子写入任务元数据（先 tmp 再 rename）。"""
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = _tz_now()
    meta_path = TASK_DIR / f"{task_id}.json"
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)


def create_task_meta(task_id: str, filenames: list[str]) -> dict:
    """创建新任务的初始元数据。"""
    now = _tz_now()
    meta = {
        "task_id": task_id,
        "status": "waiting",
        "created_at": now,
        "updated_at": now,
        "input": {
            "file_count": len(filenames),
            "filenames": filenames,
        },
        "output": None,
        "error": None,
    }
    save_task_meta(task_id, meta)
    return meta


def list_all_task_metas() -> list[dict]:
    """列出所有任务元数据（按创建时间降序）。"""
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    metas = []
    for p in sorted(TASK_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            metas.append(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return metas


# ---- 输出 ----

def get_output_dir(task_id: str) -> Path:
    """返回任务输出目录路径。"""
    return OUTPUT_DIR / task_id


def list_outputs(task_id: str) -> list[dict]:
    """列出任务的输出文件。"""
    output_dir = OUTPUT_DIR / task_id
    if not output_dir.exists():
        return []
    files = []
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "url": f"/api/result/{task_id}/{p.name}",
            })
    return files


def get_output_path(task_id: str, filename: str) -> Optional[Path]:
    """返回输出文件的完整路径，不存在则返回 None。"""
    p = OUTPUT_DIR / task_id / filename
    # 安全检查：防止路径穿越
    p = p.resolve()
    if not str(p).startswith(str((OUTPUT_DIR / task_id).resolve())):
        return None
    return p if p.exists() else None


# ---- 清理 ----

def cleanup_uploads(task_id: str) -> None:
    """删除任务的上传文件。"""
    upload_dir = UPLOAD_DIR / task_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


def delete_task(task_id: str) -> bool:
    """删除任务及其所有关联文件（元数据 + 上传 + 输出）。

    返回 True 表示删除了至少一个文件/目录，
    返回 False 表示任务不存在。
    """
    deleted = False

    # 删除任务元数据
    meta_path = TASK_DIR / f"{task_id}.json"
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    # 删除上传目录
    upload_dir = UPLOAD_DIR / task_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
        deleted = True

    # 删除输出目录
    output_dir = OUTPUT_DIR / task_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
        deleted = True

    return deleted
