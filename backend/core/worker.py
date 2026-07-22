"""后台工作线程：循环从队列取任务 → 执行重建 → 更新状态。"""

import logging
import threading
import traceback

from backend.core.queue_manager import TaskQueue
from backend.storage.file_manager import get_task_meta, save_task_meta
from backend.zipsplat_engine.runner import run_reconstruction

logger = logging.getLogger(__name__)


def start_worker(queue: TaskQueue) -> None:
    """启动后台 worker 线程（daemon，随主进程结束）。"""
    # 使用 threading.Thread 而非 asyncio，因为子进程调用是同步阻塞的
    t = threading.Thread(
        target=_worker_loop,
        args=(queue,),
        daemon=True,
        name="zipsplat-worker",
    )
    t.start()
    logger.info("Worker 线程已启动")


def _worker_loop(queue: TaskQueue) -> None:
    """Worker 主循环。"""
    while True:
        task_id = queue.dequeue(timeout=1.0)
        if task_id is None:
            continue  # 队列空闲，继续等待

        logger.info(f"开始处理任务: {task_id}")

        # 1. 更新状态为 running
        meta = get_task_meta(task_id)
        if meta is None:
            logger.error(f"任务不存在: {task_id}")
            continue
        meta["status"] = "running"
        save_task_meta(task_id, meta)

        # 2. 执行重建
        try:
            result = run_reconstruction(task_id)
            # 3. 成功 → completed
            meta = get_task_meta(task_id)
            meta["status"] = "completed"
            meta["output"] = {
                "ply": "scene.ply",
                "ply_size": result["ply_size"],
                "num_gaussians": result["num_gaussians"],
            }
            save_task_meta(task_id, meta)
            logger.info(f"任务完成: {task_id}, 高斯球: {result['num_gaussians']:,}")

        except Exception as e:
            # 4. 失败 → failed
            meta = get_task_meta(task_id)
            meta["status"] = "failed"
            meta["error"] = _format_error(e)
            save_task_meta(task_id, meta)
            logger.error(f"任务失败: {task_id}, 错误: {_format_error(e)}")


def _format_error(e: Exception) -> str:
    """格式化异常信息，保留最后 1000 字符。"""
    tb = traceback.format_exception(type(e), e, e.__traceback__)
    return "".join(tb)[-2000:]
