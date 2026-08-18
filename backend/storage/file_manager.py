"""文件管理：上传保存、任务元数据读写、输出文件管理。"""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from backend.core.config import OUTPUT_DIR, TASK_DIR, UPLOAD_DIR

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _safe_id(value: str, label: str = "ID") -> str:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} contains invalid characters")
    return value


def _safe_child(root: Path, *parts: str) -> Path:
    root = root.resolve()
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes storage root") from exc
    return path


def _save_upload_files(destination: Path, files: list[UploadFile]) -> Path:
    seen: set[str] = set()
    for upload in files:
        filename = upload.filename
        if not filename or Path(filename).name != filename or "\\" in filename:
            raise ValueError("upload filename must be a plain filename")
        key = filename.casefold()
        if key in seen:
            raise ValueError(f"duplicate upload filename: {filename}")
        seen.add(key)
    destination.mkdir(parents=True, exist_ok=True)
    for upload in files:
        filename = upload.filename
        assert filename is not None
        with _safe_child(destination, filename).open("wb") as output:
            upload.file.seek(0)
            shutil.copyfileobj(upload.file, output)
    return destination


def _tz_now() -> str:
    """返回 ISO8601 格式的 UTC 时间字符串。"""
    return datetime.now(UTC).isoformat()


# ---- 上传 ----


def save_uploads(task_id: str, files: list[UploadFile]) -> Path:
    """将上传的文件保存到 data/uploads/{task_id}/，返回保存目录。"""
    return _save_upload_files(_safe_child(UPLOAD_DIR, _safe_id(task_id, "task ID")), files)


def save_videos(task_id: str, files: list[UploadFile]) -> Path:
    """将上传的视频保存到 data/uploads/{task_id}/videos/，返回保存目录。"""
    return _save_upload_files(
        _safe_child(UPLOAD_DIR, _safe_id(task_id, "task ID"), "videos"), files
    )


# ---- 任务元数据 ----


def list_all_task_ids() -> list[str]:
    """列出服务端所有已知任务 ID。"""
    ids = set()
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        if d.exists():
            for p in d.iterdir():
                if p.is_dir() and len(p.name) == 12:  # 12-char hex task IDs
                    ids.add(p.name)
    if TASK_DIR.exists():
        for p in TASK_DIR.glob("*.json"):
            ids.add(p.stem)
    return sorted(ids, reverse=True)


def get_task_meta(task_id: str) -> dict | None:
    """读取任务元数据，不存在则返回 None。"""
    meta_path = _safe_child(TASK_DIR, f"{_safe_id(task_id, 'task ID')}.json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_task_meta(task_id: str, meta: dict) -> None:
    """原子写入任务元数据（先 tmp 再 rename）。"""
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    meta["updated_at"] = _tz_now()
    meta_path = _safe_child(TASK_DIR, f"{_safe_id(task_id, 'task ID')}.json")
    tmp_path = meta_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)


def create_task_meta(task_id: str, filenames: list[str], mode: str = "object") -> dict:
    """创建新任务的初始元数据。

    Args:
        task_id: 任务 ID
        filenames: 上传的文件名列表
        mode: 重建模式 — \"object\"（物体）或 \"scene\"（场景）
    """
    now = _tz_now()
    meta = {
        "task_id": task_id,
        "status": "waiting",
        "type": "image",
        "mode": mode,
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


def create_video_task_meta(
    task_id: str,
    filenames: list[str],
    mode: str = "scene",
    max_frames: int = 40,
    sample_interval: float = 1.0,
) -> dict:
    """创建视频重建任务的初始元数据。

    Args:
        task_id: 任务 ID
        filenames: 视频文件名列表
        mode: 重建模式（默认 scene）
        max_frames: 最终保留的最大帧数
        sample_interval: 采样间隔（秒）
    """
    now = _tz_now()
    meta = {
        "task_id": task_id,
        "status": "waiting",
        "type": "video",
        "mode": mode,
        "created_at": now,
        "updated_at": now,
        "input": {
            "video_count": len(filenames),
            "filenames": filenames,
        },
        "video_config": {
            "max_frames": max_frames,
            "sample_interval": sample_interval,
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
    return _safe_child(OUTPUT_DIR, _safe_id(task_id, "task ID"))


def list_outputs(task_id: str) -> list[dict]:
    """列出任务的输出文件。"""
    output_dir = get_output_dir(task_id)
    if not output_dir.exists():
        return []
    files = []
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            files.append(
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "url": f"/api/result/{task_id}/{p.name}",
                }
            )
    return files


def get_output_path(task_id: str, filename: str) -> Path | None:
    """返回输出文件的完整路径，不存在则返回 None。"""
    try:
        p = _safe_child(get_output_dir(task_id), filename)
    except ValueError:
        return None
    return p if p.exists() else None


# ---- 清理 ----


def cleanup_uploads(task_id: str) -> None:
    """删除任务的上传文件。"""
    upload_dir = _safe_child(UPLOAD_DIR, _safe_id(task_id, "task ID"))
    if upload_dir.exists():
        shutil.rmtree(upload_dir)


def delete_task(task_id: str) -> bool:
    """删除任务及其所有关联文件（元数据 + 上传 + 输出）。

    返回 True 表示删除了至少一个文件/目录，
    返回 False 表示任务不存在。
    """
    deleted = False

    # 删除任务元数据
    safe_task_id = _safe_id(task_id, "task ID")
    meta_path = _safe_child(TASK_DIR, f"{safe_task_id}.json")
    if meta_path.exists():
        meta_path.unlink()
        deleted = True

    # 删除上传目录
    upload_dir = _safe_child(UPLOAD_DIR, safe_task_id)
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
        deleted = True

    # 删除输出目录
    output_dir = _safe_child(OUTPUT_DIR, safe_task_id)
    if output_dir.exists():
        shutil.rmtree(output_dir)
        deleted = True

    return deleted
