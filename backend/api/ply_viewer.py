"""临时 PLY 上传端点：用户上传 PLY 文件 → 返回查看器 URL。"""

import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.core.config import DATA_DIR
from backend.core.schemas import PlyUploadResponse

router = APIRouter()

# 临时 PLY 存储目录
VIEWER_DIR = DATA_DIR / "viewer"
VIEWER_DIR.mkdir(parents=True, exist_ok=True)

# 文件大小上限：2 GB。写入采用流式分块，不会整体载入内存。
MAX_PLY_SIZE = 2 * 1024 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024  # 1 MB 分块


async def stream_to_disk(file: UploadFile, dest, max_bytes: int) -> int:
    """将上传文件流式写入 dest，边写边计数，超过 max_bytes 立即中止并清理。

    返回写入的总字节数。避免 `await file.read()` 将大文件整体读入内存导致 OOM。
    """
    total = 0
    with dest.open("wb") as out:
        while True:
            chunk = await file.read(_CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"文件过大（超过 {max_bytes / 1024 / 1024 / 1024:.0f} GB），已拒绝",
                )
            out.write(chunk)
    return total


@router.post("/view-ply", response_model=PlyUploadResponse)
async def upload_ply_for_viewing(file: UploadFile):
    """上传 PLY 文件用于临时查看。

    接收单个 .ply 文件，流式保存到临时目录，返回查看器 URL。
    文件保留 24 小时后由后台清理（或重启时清理）。

    Returns:
        PlyUploadResponse {ply_id, filename, size, url}
    """
    if not file.filename or not file.filename.lower().endswith(".ply"):
        raise HTTPException(
            status_code=400,
            detail="仅支持 .ply 格式的 3D 高斯点云文件",
        )

    ply_id = uuid.uuid4().hex[:12]
    ply_path = VIEWER_DIR / f"{ply_id}.ply"
    size = await stream_to_disk(file, ply_path, MAX_PLY_SIZE)
    if size == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    return PlyUploadResponse(
        ply_id=ply_id,
        filename=file.filename,
        size=size,
        # 带 .ply 后缀:SuperSplat 编辑器按 URL 扩展名识别格式,无后缀无法自动加载
        url=f"/api/ply/{ply_id}.ply",
    )


@router.get("/ply/{ply_id}")
async def get_ply_file(ply_id: str):
    """获取临时 PLY 文件。

    兼容 /ply/{id} 与 /ply/{id}.ply 两种形式 —— 编辑器(SuperSplat)按 URL
    扩展名识别格式,必须带 .ply 后缀才能自动加载。
    """
    # 剥掉可选的 .ply 后缀
    if ply_id.lower().endswith(".ply"):
        ply_id = ply_id[:-4]
    # 安全检查：只允许 hex 字符
    if not ply_id.isalnum() or len(ply_id) > 20:
        raise HTTPException(status_code=404, detail="无效的 PLY ID")

    ply_path = VIEWER_DIR / f"{ply_id}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="PLY 文件不存在或已过期")

    return FileResponse(
        path=str(ply_path),
        filename=f"{ply_id}.ply",
        media_type="application/octet-stream",
    )
