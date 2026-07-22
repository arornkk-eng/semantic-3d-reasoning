import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | null;
  error: string | null;
}

export default function HealthBadge({ health, error }: Props) {
  const connected = health?.status === "ok";
  const gpuName = health?.gpu_name ?? "—";
  const gpuMem = health?.gpu_memory_total ?? "—";
  const queue = health?.queue_size ?? 0;

  return (
    <div
      className={`flex items-center gap-3 px-4 py-2 text-xs transition-colors ${
        connected
          ? "bg-gray-900 text-gray-300"
          : "bg-red-900 text-red-200"
      }`}
    >
      {/* 连接状态 */}
      <span className="flex items-center gap-1.5">
        <span
          className={`inline-block w-2 h-2 rounded-full ${
            connected ? "bg-green-400" : "bg-red-400 animate-pulse"
          }`}
        />
        {connected && !error ? "后端已连接" : "后端断开"}
      </span>

      {connected && (
        <>
          <span className="text-gray-600">|</span>
          <span title="GPU">🖥 {gpuName}</span>
          <span className="text-gray-600">|</span>
          <span>显存: {gpuMem}</span>
          <span className="text-gray-600">|</span>
          <span>队列: {queue} 个任务</span>
        </>
      )}

      {error && (
        <span className="text-red-400 ml-auto" title={error}>
          错误: {error.length > 40 ? error.slice(0, 40) + "…" : error}
        </span>
      )}
    </div>
  );
}
