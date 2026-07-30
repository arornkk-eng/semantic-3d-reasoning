"""后台 Worker：循环从队列取任务 → 子进程执行重建 → 更新状态。

使用 multiprocessing.Process 隔离任务执行，支持运行时终止。
子进程终止时 GPU 显存由 CUDA 驱动自动回收。
"""

import logging
import multiprocessing
import threading
import traceback

from backend.core.queue_manager import TaskQueue
from backend.storage.file_manager import get_task_meta, save_task_meta

logger = logging.getLogger(__name__)

# ---- 模块级状态（用于跨线程取消） ----
_current_task_id: str | None = None
_current_proc: multiprocessing.Process | None = None
_cancel_flag = False
_lock = threading.Lock()


def start_worker(queue: TaskQueue) -> None:
    """启动后台 worker 线程（daemon，随主进程结束）。"""
    t = threading.Thread(
        target=_worker_loop,
        args=(queue,),
        daemon=True,
        name="zipsplat-worker",
    )
    t.start()
    logger.info("Worker 线程已启动")


def cancel_current(task_id: str) -> bool:
    """终止当前正在运行的任务（子进程）。

    Args:
        task_id: 要取消的任务 ID（必须与当前运行的任务匹配）

    Returns:
        True  成功终止子进程
        False 任务 ID 不匹配，或没有正在运行的任务
    """
    global _cancel_flag, _current_proc, _current_task_id

    with _lock:
        if _current_task_id != task_id:
            logger.warning(f"取消请求不匹配: 请求={task_id}, 当前运行={_current_task_id}")
            return False

        if _current_proc is None or not _current_proc.is_alive():
            logger.warning(f"没有运行中的子进程: task_id={task_id}")
            return False

        _cancel_flag = True
        proc = _current_proc

    # 在锁外终止进程，避免死锁
    proc.terminate()

    # 等待进程退出（可配置超时）
    proc.join(timeout=10)
    if proc.is_alive():
        logger.warning(f"子进程未响应 terminate，执行 kill: {task_id}")
        proc.kill()
        proc.join(timeout=5)

    logger.info(f"已终止运行中的任务: {task_id}")
    return True


# ---- 内部实现 ----

def _worker_loop(queue: TaskQueue) -> None:
    """Worker 主循环：取任务 → 子进程执行 → 更新状态。"""
    global _current_task_id, _current_proc, _cancel_flag

    while True:
        task_id = queue.dequeue(timeout=1.0)
        if task_id is None:
            continue  # 队列空闲

        logger.info(f"开始处理任务: {task_id}")

        # 1. 更新状态为 running + 时间预估
        meta = get_task_meta(task_id)
        if meta is None:
            logger.error(f"任务不存在: {task_id}")
            continue
        meta["status"] = "running"
        meta["started_at"] = meta.get("updated_at", "")  # worker 开始处理的时间
        # 预估耗时：图片任务 ~2min，视频任务 ~6min
        is_video = meta.get("type") == "video"
        frame_count = meta.get("video_config", {}).get("max_frames", 25) if is_video else meta.get("input", {}).get("file_count", 10)
        meta["estimated_seconds"] = 300 if is_video else 120  # 25帧≈5分钟
        meta["stage"] = "extracting" if is_video else "reconstructing"
        meta["progress_pct"] = 0
        save_task_meta(task_id, meta)

        # 2. 在子进程中执行重建（隔离 GPU 上下文）
        with _lock:
            _current_task_id = task_id
            _cancel_flag = False
            _current_proc = multiprocessing.Process(
                target=_run_in_subprocess,
                args=(task_id,),
                daemon=True,
            )

        _current_proc.start()
        _current_proc.join()  # 阻塞直到子进程退出（正常/异常/被终止）

        # 3. 根据退出原因更新状态
        with _lock:
            was_cancelled = _cancel_flag
            _cancel_flag = False

        if was_cancelled:
            meta = get_task_meta(task_id)
            if meta and meta["status"] == "running":
                meta["status"] = "cancelled"
                save_task_meta(task_id, meta)
            logger.info(f"任务已取消: {task_id}")

        else:
            # 正常退出 — 元数据已由子进程更新为 completed/failed
            meta = get_task_meta(task_id)
            if meta is None:
                logger.error(f"任务元数据丢失: {task_id}")
            elif meta["status"] == "completed":
                logger.info(f"任务完成: {task_id}, "
                            f"高斯球: {meta.get('output', {}).get('num_gaussians', '?')}")
            elif meta["status"] == "failed":
                logger.error(f"任务失败: {task_id}, 错误: {meta.get('error', '?')[:120]}")
            else:
                # 子进程崩溃或异常退出（元数据未被更新）
                meta["status"] = "failed"
                meta["error"] = "Worker 进程异常退出（可能显存不足）"
                save_task_meta(task_id, meta)
                logger.error(f"子进程异常退出: {task_id}")

        with _lock:
            _current_task_id = None
            _current_proc = None


def _run_in_subprocess(task_id: str) -> None:
    """子进程入口：执行重建并直接更新元数据。

    此函数在独立的子进程中运行。子进程终止时，
    CUDA 驱动自动回收该进程持有的所有 GPU 显存。
    """
    # 子进程独立日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s(sub): %(message)s",
        datefmt="%H:%M:%S",
    )
    sub_logger = logging.getLogger(__name__)

    try:
        from backend.zipsplat_engine.runner import run_reconstruction
        from backend.storage.file_manager import get_task_meta as _get, save_task_meta as _save

        meta = _get(task_id)
        mode = meta.get("mode", "object") if meta else "object"
        task_type = meta.get("type", "image") if meta else "image"

        # ---- 视频任务：先提取帧 ----
        if task_type == "video":
            from pathlib import Path
            from backend.core.config import UPLOAD_DIR
            from backend.video_processor.extractor import extract_and_select

            video_config = meta.get("video_config", {})
            max_frames = video_config.get("max_frames", 25)
            sample_interval = video_config.get("sample_interval", 1.0)

            video_dir = UPLOAD_DIR / task_id / "videos"
            if not video_dir.exists():
                raise FileNotFoundError(f"视频目录不存在: {video_dir}")

            video_paths = sorted(video_dir.iterdir())
            sub_logger.info(
                f"视频任务: {len(video_paths)} 段视频, "
                f"max_frames={max_frames}, interval={sample_interval}s"
            )

            # 输出帧到 uploads/{task_id}/（与图片任务一致，方便 run_reconstruction 读取）
            frame_dir = UPLOAD_DIR / task_id
            extract_result = extract_and_select(
                video_paths=video_paths,
                output_dir=frame_dir,
                max_frames=max_frames,
                sample_interval=sample_interval,
            )

            sub_logger.info(
                f"帧提取完成: 原始 {extract_result['total_extracted']} → "
                f"质量通过 {extract_result['quality_passed']} → "
                f"选中 {extract_result['selected']}"
            )

            # 更新元数据：抽帧完成，进入重建阶段
            meta = _get(task_id)
            if meta:
                meta["stage"] = "reconstructing"
                meta["progress_pct"] = 20  # 抽帧约占 20% 总进度
                meta["video_result"] = {
                    "total_extracted": extract_result["total_extracted"],
                    "quality_passed": extract_result["quality_passed"],
                    "selected": extract_result["selected"],
                }
                _save(task_id, meta)

            # 如果没选出任何帧，直接失败
            if extract_result["selected"] == 0:
                raise RuntimeError("未能从视频中选出任何合格帧，请检查视频质量或降低阈值")

        # ---- 执行重建 ----
        sub_logger.info(f"重建模式: {mode}")
        result = run_reconstruction(task_id, mode=mode)

        # 成功 → 更新元数据为 completed
        meta = _get(task_id)
        if meta:
            meta["status"] = "completed"
            meta["output"] = {
                "ply": "scene.ply",
                "ply_size": result["ply_size"],
                "num_gaussians": result["num_gaussians"],
            }
            meta["error"] = None
            _save(task_id, meta)
            sub_logger.info(f"重建成功: {task_id}, 高斯球={result['num_gaussians']:,}")

    except Exception as e:
        sub_logger.error(f"重建失败: {task_id}, 错误: {e}")

        try:
            from backend.storage.file_manager import get_task_meta as _get, save_task_meta as _save

            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            meta = _get(task_id)
            if meta:
                meta["status"] = "failed"
                meta["error"] = tb[-2000:]
                _save(task_id, meta)
        except Exception as meta_err:
            sub_logger.error(f"更新失败状态时出错: {meta_err}")
