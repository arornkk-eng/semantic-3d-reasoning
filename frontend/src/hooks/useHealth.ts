import { useCallback } from "react";
import { usePolling } from "./usePolling";
import { getHealth } from "../api/client";
import type { HealthResponse } from "../types";

const HEALTH_INTERVAL = 30_000; // 30 秒

export function useHealth() {
  const fetcher = useCallback(() => getHealth(), []);
  const { data, error } = usePolling<HealthResponse>(
    fetcher,
    HEALTH_INTERVAL,
    true
  );
  return { health: data, error };
}
