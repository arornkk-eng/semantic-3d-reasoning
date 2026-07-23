"""任务队列：python queue.Queue 包装，支持取消和持久化恢复。"""

import logging
import queue
import threading
from typing import Optional

from backend.core.config import TASK_DIR
from backend.storage.file_manager import get_task_meta, list_all_task_metas, save_task_meta

logger = logging.getLogger(__name__)


class TaskQueue:
    """单 GPU 串行任务队列，支持按 ID 取消等待中的任务。"""

    def __init__(self):
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    def enqueue(self, task_id: str) -> None:
        """将任务加入队列。"""
        self._queue.put(task_id)
        logger.info(f"任务入队: {task_id}, 当前队列长度: {self._queue.qsize()}")

    def dequeue(self, timeout: float = 1.0) -> Optional[str]:
        """阻塞地从队列取出一个任务。跳过已取消的任务。超时返回 None。"""
        while True:
            try:
                task_id = self._queue.get(timeout=timeout)
            except queue.Empty:
                return None

            with self._lock:
                if task_id in self._cancelled:
                    self._cancelled.discard(task_id)
                    logger.info(f"跳过已取消的任务: {task_id}")
                    continue

            logger.info(f"任务出队: {task_id}, 剩余: {self._queue.qsize()}")
            return task_id

    def cancel(self, task_id: str) -> bool:
        """取消等待中的任务：标记为 cancelled 并从队列逻辑中移除。

        Returns:
            True  任务在等待队列中，已取消
            False 任务不在等待队列中（可能已开始执行或已完成）
        """
        meta = get_task_meta(task_id)
        if meta is None or meta["status"] != "waiting":
            return False

        meta["status"] = "cancelled"
        save_task_meta(task_id, meta)

        with self._lock:
            self._cancelled.add(task_id)

        logger.info(f"任务已取消: {task_id}")
        return True

    def size(self) -> int:
        """返回当前队列长度。"""
        return self._queue.qsize()

    def recover_pending(self) -> int:
        """启动时恢复未完成的任务（waiting/running → 重新入队）。

        返回恢复的任务数量。
        """
        TASK_DIR.mkdir(parents=True, exist_ok=True)
        recovered = 0
        for meta in list_all_task_metas():
            task_id = meta["task_id"]
            status = meta["status"]
            if status in ("waiting", "running"):
                # 将 running 重置为 waiting 并重新入队
                if status == "running":
                    meta["status"] = "waiting"
                    save_task_meta(task_id, meta)
                self._queue.put(task_id)
                recovered += 1
                logger.info(f"恢复任务: {task_id} (原状态: {status})")
        if recovered:
            logger.info(f"共恢复 {recovered} 个未完成任务")
        return recovered
