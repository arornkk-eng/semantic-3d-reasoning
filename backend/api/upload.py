"""POST /api/upload — 图片上传端点。"""

import uuid
import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.core.config import ALLOWED_EXTENSIONS, MAX_FILE_COUNT, MAX_UPLOAD_SIZE_BYTES
from backend.core.queue_manager import TaskQueue
from backend.storage.file_manager import create_task_meta, save_uploads

logger = logging.getLogger(__name__)
router = APIRouter()


def get_queue(request: Request) -> TaskQueue:
    """从 app.state 获取任务队列。"""
    return request.app.state.task_queue


@router.post("/upload")
async def upload_images(
    request: Request,
    files: list[UploadFile] = File(..., description="图片文件（多选）"),
):
    """上传多张图片，创建重建任务，返回 task_id。

    限制：单文件 ≤ 50MB 总大小，格式限 jpg/png/bmp/webp，最多 50 张。
    """
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
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
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
    filenames = [f.filename for f in files]
    save_uploads(task_id, files)
    create_task_meta(task_id, filenames)

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
