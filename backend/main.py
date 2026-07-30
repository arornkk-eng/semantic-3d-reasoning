"""ZipSplat-Demo FastAPI 后端入口。

提供图片上传、任务队列、结果下载的 REST API。
通过 run_backend.bat 启动（需要 MSVC + CUDA 环境）。
"""

import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import ply_viewer, result, task, upload
from backend.recognition.router import router as recognition_router
from backend.core.config import OUTPUT_DIR, TASK_DIR, UPLOAD_DIR
from backend.core.queue_manager import TaskQueue
from backend.core.worker import start_worker

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- FastAPI 应用 ----
app = FastAPI(
    title="ZipSplat-Demo",
    description="本地 AI 3D 重建 Web Demo — 多视角图片 → 高斯点云 + 360° 视频",
    version="0.1.0",
)

# CORS（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 全局状态 ----
task_queue = TaskQueue()
app.state.task_queue = task_queue


@app.on_event("startup")
async def startup():
    """应用启动：创建目录 → 恢复任务 → 启动 Worker。"""
    # 确保数据目录存在
    for d in [TASK_DIR, UPLOAD_DIR, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录就绪: {d}")

    # 恢复未完成任务
    recovered = task_queue.recover_pending()
    if recovered:
        logger.info(f"已恢复 {recovered} 个未完成任务")

    # 启动后台 Worker 线程
    start_worker(task_queue)
    logger.info("ZipSplat-Demo 后端已就绪")


@app.get("/api/health")
async def health():
    """健康检查 + GPU 信息。"""
    gpu_info = {}
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                "cuda_version": torch.version.cuda,
            }
        else:
            gpu_info = {"gpu_name": "N/A (CPU only)"}
    except Exception:
        gpu_info = {"gpu_name": "检测失败"}

    return {
        "status": "ok",
        **gpu_info,
        "queue_size": task_queue.size(),
    }


@app.get("/api/tasks")
async def list_tasks():
    """列出服务端所有任务 ID。"""
    from backend.storage.file_manager import list_all_task_ids
    return {"task_ids": list_all_task_ids()}


# ---- 注册路由 ----
app.include_router(upload.router, prefix="/api")
app.include_router(task.router, prefix="/api")
app.include_router(result.router, prefix="/api")
app.include_router(recognition_router, prefix="/api")
app.include_router(ply_viewer.router, prefix="/api")
