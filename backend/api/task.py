"""任务状态查询 / 取消 / 删除端点。"""

from fastapi import APIRouter, HTTPException, Request

from backend.storage.file_manager import delete_task, get_task_meta

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


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str, request: Request):
    """取消任务：等待中 → 移出队列；运行中 → 终止子进程释放 GPU。

    已完成的、已失败的、已取消的任务不可再次取消。
    """
    meta = get_task_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    status = meta["status"]

    if status == "completed":
        raise HTTPException(status_code=400, detail="任务已完成，无法取消")
    if status == "failed":
        raise HTTPException(status_code=400, detail="任务已失败，无需取消")
    if status == "cancelled":
        raise HTTPException(status_code=400, detail="任务已经取消了")

    if status == "waiting":
        # 从队列中标记取消
        queue = request.app.state.task_queue
        queue.cancel(task_id)

    elif status == "running":
        # 终止正在运行的子进程
        from backend.core.worker import cancel_current

        if not cancel_current(task_id):
            raise HTTPException(
                status_code=409,
                detail="无法终止：任务进程已退出或任务 ID 不匹配",
            )

    return {"task_id": task_id, "cancelled": True}


@router.delete("/task/{task_id}")
async def delete_task_endpoint(task_id: str):
    """删除任务及其所有关联文件（上传图片、输出文件、元数据）。

    返回: 删除成功确认。
    """
    meta = get_task_meta(task_id)
    if meta is not None and meta.get("status") in {"waiting", "running"}:
        raise HTTPException(status_code=409, detail="请先取消运行中或排队中的任务")
    deleted = delete_task(task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return {"task_id": task_id, "deleted": True}
