"""临时 PLY 上传端点：用户上传 PLY 文件 → 返回查看器 URL。"""

import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.core.config import DATA_DIR

router = APIRouter()

# 临时 PLY 存储目录
VIEWER_DIR = DATA_DIR / "viewer"
VIEWER_DIR.mkdir(parents=True, exist_ok=True)

# 文件大小限制：300 MB
MAX_PLY_SIZE = 300 * 1024 * 1024


@router.post("/view-ply")
async def upload_ply_for_viewing(file: UploadFile):
    """上传 PLY 文件用于临时查看。

    接收单个 .ply 文件，保存到临时目录，返回查看器 URL。
    文件保留 24 小时后由后台清理（或重启时清理）。

    Returns:
        {ply_id, url, viewer_url}
    """
    if not file.filename or not file.filename.lower().endswith(".ply"):
        raise HTTPException(
            status_code=400,
            detail="仅支持 .ply 格式的 3D 高斯点云文件",
        )

    # 读取文件内容（检查大小）
    content = await file.read()
    if len(content) > MAX_PLY_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大（{len(content) / 1024 / 1024:.1f} MB），最大支持 300 MB",
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件为空")

    # 保存到临时目录
    ply_id = uuid.uuid4().hex[:12]
    ply_path = VIEWER_DIR / f"{ply_id}.ply"
    ply_path.write_bytes(content)

    return {
        "ply_id": ply_id,
        "filename": file.filename,
        "size": len(content),
        "url": f"/api/ply/{ply_id}",
    }


@router.get("/ply/{ply_id}")
async def get_ply_file(ply_id: str):
    """获取临时 PLY 文件。"""
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


@router.post("/discover-ply/{ply_id}")
async def discover_ply_objects(ply_id: str, body: dict | None = None):
    """对临时 PLY 文件运行 3D 聚类发现（不需要原始帧）。

    Body (可选):
        {"n_clusters": 5, "n_samples": 5000}
    """
    import asyncio

    if not ply_id.isalnum() or len(ply_id) > 20:
        raise HTTPException(status_code=404, detail="无效的 PLY ID")

    ply_path = VIEWER_DIR / f"{ply_id}.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="PLY 文件不存在或已过期")

    if body is None:
        body = {}

    n_clusters = int(body.get("n_clusters", 5))
    n_samples = int(body.get("n_samples", 5000))

    def _run():
        from backend.recognition.cluster_3d import discover_objects
        output_path = VIEWER_DIR / f"{ply_id}_clusters.json"
        return discover_objects(
            ply_path=ply_path,
            output_path=output_path,
            n_clusters=n_clusters,
            n_samples=n_samples,
        )

    try:
        loop = asyncio.get_running_loop()
        clusters = await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聚类失败: {str(e)}")

    return {
        "ply_id": ply_id,
        "n_clusters_found": len(clusters),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "count": c["count"],
                "ratio": c.get("ratio", 0),
                "center_3d": c["center_3d"],
                "bbox_3d": c["bbox_3d"],
                "dominant_color_rgb": c["dominant_color_rgb"],
                "suggested_label": c.get("suggested_label", "object"),
                "label_confidence": c.get("label_confidence", 0),
            }
            for c in clusters
        ],
    }


@router.get("/discover-ply/{ply_id}")
async def get_ply_clusters(ply_id: str):
    """获取已计算的 PLY 聚类结果。"""
    import json

    if not ply_id.isalnum() or len(ply_id) > 20:
        raise HTTPException(status_code=404, detail="无效的 PLY ID")

    clusters_path = VIEWER_DIR / f"{ply_id}_clusters.json"
    if not clusters_path.exists():
        return {"ply_id": ply_id, "clusters": []}

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    return {
        "ply_id": ply_id,
        "method": clusters.get("method"),
        "n_clusters_found": clusters.get("n_clusters_found", 0),
        "total_vertices": clusters.get("total_vertices", 0),
        "clusters": clusters.get("clusters", []),
    }
