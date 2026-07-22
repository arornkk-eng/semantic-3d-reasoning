"""GET /api/task/{task_id} — 任务状态查询端点。"""

from fastapi import APIRouter, HTTPException

from backend.storage.file_manager import get_task_meta

router = APIRouter()


@router.get("/task/{task_id}")
async def get_task(task_id: str):
    """查询任务状态。

    返回: task_id, status, created_at, updated_at,
          input (file_count, filenames),
          output (ply, video, num_gaussians) — 仅当 completed,
          error — 仅当 failed。
    """
    meta = get_task_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return meta
