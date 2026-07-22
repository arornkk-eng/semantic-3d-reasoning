import { useState, useEffect, useRef } from "react";

/**
 * 通用轮询 hook。
 *
 * @param fetcher  异步数据获取函数
 * @param interval 轮询间隔（毫秒）
 * @param enabled  是否启用轮询（false 时停止）
 * @returns { data, error, loading }
 */
export function usePolling<T>(
  fetcher: () => Promise<T>,
  interval: number,
  enabled: boolean
): { data: T | null; error: string | null; loading: boolean } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const savedFetcher = useRef(fetcher);
  savedFetcher.current = fetcher;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const tick = async () => {
      try {
        const result = await savedFetcher.current();
        if (!cancelled) {
          setData(result);
          setError(null);
          setLoading(false);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "未知错误");
          setLoading(false);
        }
      }
      if (!cancelled) {
        timer = setTimeout(tick, interval);
      }
    };

    tick();

    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [interval, enabled]);

  return { data, error, loading };
}
