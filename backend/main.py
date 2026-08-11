"""ZipSplat-Demo FastAPI 后端入口。

提供图片上传、任务队列、结果下载的 REST API。
通过 run_backend.bat 启动（需要 MSVC + CUDA 环境）。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import ply_viewer, result, segmentation, task, upload
from backend.core.config import LAYER_DIR, OUTPUT_DIR, TASK_DIR, UPLOAD_DIR, resolve_cors
from backend.core.queue_manager import TaskQueue
from backend.core.schemas import HealthResponse
from backend.core.worker import start_worker

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---- 生命周期管理（取代已弃用的 @app.on_event）----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子：建目录 → 恢复任务 → 启动 Worker。"""
    for d in [TASK_DIR, UPLOAD_DIR, OUTPUT_DIR, LAYER_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"数据目录就绪: {d}")

    recovered = task_queue.recover_pending()
    if recovered:
        logger.info(f"已恢复 {recovered} 个未完成任务")

    start_worker(task_queue)
    logger.info("ZipSplat-Demo 后端已就绪")
    yield
    # 关闭钩子：Worker 为守护线程，进程退出即终止，无需额外清理


# ---- FastAPI 应用 ----
app = FastAPI(
    title="ZipSplat-Demo",
    description="本地 AI 3D 重建 Web Demo — 多视角图片 → 高斯点云 + 360° 视频",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS：显式来源白名单（来自配置/环境变量），绝不开放 "*" + 凭证
cors_origins, cors_credentials, cors_origin_regex = resolve_cors()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=cors_credentials,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# ---- 全局状态 ----
task_queue = TaskQueue()
app.state.task_queue = task_queue


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """健康检查 + GPU 信息。"""
    gpu_info: dict[str, str] = {}
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
app.include_router(ply_viewer.router, prefix="/api")
app.include_router(segmentation.router, prefix="/api")
