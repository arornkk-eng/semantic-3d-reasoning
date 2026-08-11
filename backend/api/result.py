"""GET /api/result/{task_id} — 结果查询与下载端点。"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.storage.file_manager import get_output_path, get_task_meta, list_outputs

router = APIRouter()


@router.get("/result/{task_id}")
async def get_result(task_id: str):
    """列出任务的所有输出文件。

    返回: task_id, status, files [{name, size, url}]。
    """
    meta = get_task_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    if meta["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态: {meta['status']}",
        )

    files = list_outputs(task_id)
    return {
        "task_id": task_id,
        "status": meta["status"],
        "files": files,
    }


@router.get("/result/{task_id}/{filename:path}")
async def download_file(task_id: str, filename: str):
    """下载单个输出文件。

    支持: scene.ply, turntable_360.mp4。
    """
    file_path = get_output_path(task_id, filename)
    if file_path is None:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    media_type = "application/octet-stream" if filename.endswith(".ply") else None

    resp = FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp
