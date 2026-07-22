import { useState, useEffect, useRef, useCallback } from "react";
import { getTask } from "../api/client";
import type { TaskMeta } from "../types";

const POLL_INTERVAL = 2000; // 2 秒
const MAX_POLL_TIME = 15 * 60 * 1000; // 15 分钟超时

/**
 * 跟踪单个任务的实时状态。
 * 任务到达终态（completed/failed）后自动停止轮询，或超过 15 分钟后停止。
 */
export function useTask(taskId: string | null) {
  const [task, setTask] = useState<TaskMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const startTimeRef = useRef<number>(0);

  const isTerminal = (t: TaskMeta | null) =>
    t?.status === "completed" || t?.status === "failed";

  const tick = useCallback(async () => {
    if (!taskId) return;
    try {
      const result = await getTask(taskId);
      setTask(result);
      setError(null);
      setLoading(false);

      if (!isTerminal(result) && Date.now() - startTimeRef.current < MAX_POLL_TIME) {
        timerRef.current = setTimeout(tick, POLL_INTERVAL);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "未知错误");
      setLoading(false);
      // 出错也继续轮询
      timerRef.current = setTimeout(tick, POLL_INTERVAL);
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId) {
      setTask(null);
      setError(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    setTask(null);
    setError(null);
    startTimeRef.current = Date.now();

    tick();

    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [taskId, tick]);

  return { task, error, loading };
}
