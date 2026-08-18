"""POST /api/upload — 图片上传端点。
POST /api/upload-video — 视频上传 + 智能帧提取端点。
"""

import logging
import uuid

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.core.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_FILE_COUNT,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_VIDEO_COUNT,
    MAX_VIDEO_SIZE_BYTES,
)
from backend.core.queue_manager import TaskQueue
from backend.core.schemas import UploadResponse, VideoUploadResponse
from backend.storage.file_manager import (
    create_task_meta,
    create_video_task_meta,
    save_uploads,
    save_videos,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_queue(request: Request) -> TaskQueue:
    """从 app.state 获取任务队列。"""
    return request.app.state.task_queue


@router.post("/upload", response_model=UploadResponse)
async def upload_images(
    request: Request,
    files: list[UploadFile] = File(..., description="图片文件（多选）"),
    mode: str = Form("object", description="重建模式: object（物体） / scene（场景）"),
):
    """上传多张图片，创建重建任务，返回 task_id。

    限制：总大小 ≤ 100MB，格式限 jpg/png/bmp/webp，最多 12 张。
    """
    # ---- 校验 mode ----
    if mode not in ("object", "scene"):
        raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}（支持: object, scene）")

    # ---- 校验 ----
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一张图片")

    if len(files) > MAX_FILE_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"最多上传 {MAX_FILE_COUNT} 张图片，当前 {len(files)} 张",
        )

    # 校验扩展名 + 读取总大小
    total_size = 0
    filenames: list[str] = []
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        filenames.append(f.filename)
        ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件格式: {f.filename}（支持: {', '.join(sorted(ALLOWED_EXTENSIONS))})",
            )
        # 获取文件大小
        f.file.seek(0, 2)  # seek to end
        size = f.file.tell()
        f.file.seek(0)
        total_size += size

    if total_size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件总大小 {total_size / 1024 / 1024:.1f} MB 超出限制（{MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB）",
        )

    # ---- 保存 ----
    task_id = uuid.uuid4().hex[:12]  # 12 位短 ID，方便 URL 使用
    try:
        save_uploads(task_id, files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    create_task_meta(task_id, filenames, mode=mode)

    # ---- 入队 ----
    queue = get_queue(request)
    queue.enqueue(task_id)

    logger.info(f"上传完成: {task_id}, {len(files)} 个文件, {total_size / 1024:.1f} KB")

    return {
        "task_id": task_id,
        "status": "waiting",
        "file_count": len(files),
        "queue_position": queue.size(),
    }


@router.post("/upload-video", response_model=VideoUploadResponse)
async def upload_videos(
    request: Request,
    videos: list[UploadFile] = File(..., description="MP4 视频文件（多选）"),
    mode: str = Form("scene", description="重建模式（视频默认 scene）"),
    max_frames: int = Form(25, description="最终保留的最大帧数"),
    sample_interval: float = Form(1.0, description="采样间隔（秒）"),
):
    """上传多段 MP4 视频，创建视频重建任务。

    工作流：上传 → 保存视频 → Worker 提取帧 → 质量筛选 → 多样性选择 → 3D 重建。
    限制：单段视频 ≤ 500MB，格式限 mp4/avi/mov/mkv，最多 10 段。
    """
    # ---- 校验 mode ----
    if mode not in ("object", "scene"):
        raise HTTPException(status_code=400, detail=f"不支持的模式: {mode}（支持: object, scene）")

    # ---- 校验 ----
    if not videos:
        raise HTTPException(status_code=400, detail="请至少上传一段视频")

    if len(videos) > MAX_VIDEO_COUNT:
        raise HTTPException(
            status_code=400,
            detail=f"最多上传 {MAX_VIDEO_COUNT} 段视频，当前 {len(videos)} 段",
        )

    # ---- 校验扩展名 + 大小 ----
    total_size = 0
    video_filenames: list[str] = []
    for f in videos:
        if not f.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        video_filenames.append(f.filename)
        ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的视频格式: {f.filename}（支持: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))})",
            )
        f.file.seek(0, 2)
        size = f.file.tell()
        f.file.seek(0)
        total_size += size
        if size > MAX_VIDEO_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"单段视频 {f.filename} 大小 {size / 1024 / 1024:.1f} MB 超出限制"
                f"（{MAX_VIDEO_SIZE_BYTES / 1024 / 1024:.0f} MB）",
            )

    # ---- 校验参数 ----
    if max_frames < 5 or max_frames > MAX_FILE_COUNT:
        raise HTTPException(
            status_code=400, detail=f"max_frames 应在 5~{MAX_FILE_COUNT} 之间"
        )
    if sample_interval < 0.1 or sample_interval > 10.0:
        raise HTTPException(status_code=400, detail="sample_interval 应在 0.1~10.0 秒之间")

    # ---- 保存视频 ----
    task_id = uuid.uuid4().hex[:12]
    try:
        save_videos(task_id, videos)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    create_video_task_meta(
        task_id,
        video_filenames,
        mode=mode,
        max_frames=max_frames,
        sample_interval=sample_interval,
    )

    # ---- 入队 ----
    queue = get_queue(request)
    queue.enqueue(task_id)

    logger.info(
        f"视频上传完成: {task_id}, {len(videos)} 段视频, "
        f"总大小 {total_size / 1024 / 1024:.1f} MB, "
        f"max_frames={max_frames}, interval={sample_interval}s"
    )

    return {
        "task_id": task_id,
        "status": "waiting",
        "video_count": len(videos),
        "queue_position": queue.size(),
    }
