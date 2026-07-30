import { useState } from "react";
import { useTask } from "../hooks/useTask";
import { cancelTask, deleteTask } from "../api/client";
import StatusBadge from "./StatusBadge";
import ResultPanel from "./ResultPanel";

interface Props {
  taskId: string;
  onDelete: (taskId: string) => void;
}

/** 单个任务卡片：自动轮询状态，展示进度和结果。支持取消运行中/等待中的任务。 */
export default function TaskCard({ taskId, onDelete }: Props) {
  const { task, error } = useTask(taskId);
  const [deleting, setDeleting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const handleCancel = async () => {
    setCancelling(true);
    setCancelError(null);
    try {
      await cancelTask(taskId);
    } catch (e: any) {
      setCancelError(e.message);
      setCancelling(false);
    }
  };

  const handleDelete = async () => {
    setDeleting(true);
    try {
      await deleteTask(taskId);
      onDelete(taskId);
    } catch (e: any) {
      alert(`删除失败: ${e.message}`);
      setDeleting(false);
      setShowConfirm(false);
    }
  };

  if (error && !task) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-center justify-between">
        <p className="text-sm text-red-700">
          加载任务 {taskId.slice(0, 8)}… 失败: {error}
        </p>
        <button
          onClick={() => onDelete(taskId)}
          className="text-xs text-red-500 hover:text-red-700 underline shrink-0 ml-2"
        >
          移除
        </button>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-1/3 mb-2" />
        <div className="h-3 bg-gray-200 rounded w-1/2" />
      </div>
    );
  }

  const shortId = task.task_id.slice(0, 8);
  const timeStr = new Date(task.created_at).toLocaleString("zh-CN");
  const isActive = task.status === "waiting" || task.status === "running";
  const isVideo = task.type === "video";

  // ---- 进度计算 ----
  const getProgress = (): { pct: number; label: string; remaining: string } => {
    // 已完成 → 100%
    if (task.status === "completed") return { pct: 100, label: "完成", remaining: "" };
    if (task.status === "failed") return { pct: 100, label: "失败", remaining: "" };
    if (task.status === "cancelled") return { pct: 100, label: "已取消", remaining: "" };

    // 后端有明确的 progress_pct
    if (task.progress_pct !== undefined && task.progress_pct > 0) {
      const stageLabel = task.stage === "extracting" ? "抽帧中" : "重建中";
      return { pct: task.progress_pct, label: stageLabel, remaining: "…" };
    }

    // 等待中 → 0%
    if (task.status === "waiting") {
      return { pct: 0, label: "排队中", remaining: "等待 worker…" };
    }

    // 运行中：基于时间估算
    const estimated = task.estimated_seconds || (isVideo ? 360 : 120);
    const startedAt = task.started_at ? new Date(task.started_at).getTime() : new Date(task.created_at).getTime();
    const elapsed = (Date.now() - startedAt) / 1000;
    const pct = Math.min(Math.round((elapsed / estimated) * 100), 95);
    const stageLabel = task.stage === "extracting" ? "抽帧中" : "重建中";
    const remainSec = Math.max(0, estimated - elapsed);
    const remaining = remainSec > 60
      ? `约 ${Math.ceil(remainSec / 60)} 分钟`
      : `约 ${Math.ceil(remainSec)} 秒`;

    return { pct, label: stageLabel, remaining };
  };

  const progress = getProgress();
  const progressColor =
    task.status === "failed" ? "bg-red-500" :
    task.status === "cancelled" ? "bg-amber-500" :
    task.status === "completed" ? "bg-green-500" :
    "bg-blue-500";

  return (
    <div className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow transition-shadow relative group">
      {/* 右上角操作按钮 */}
      <div className="absolute top-2 right-2">
        <div className="flex items-center gap-1">
          {/* 取消按钮 — 仅活跃任务可见 */}
          {isActive && !cancelling && (
            <button
              onClick={handleCancel}
              disabled={cancelling}
              title="终止任务"
              className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-amber-500 p-1 rounded"
            >
              <StopIcon />
            </button>
          )}

          {/* 取消中 */}
          {cancelling && (
            <span className="text-xs text-amber-600 animate-pulse">终止中…</span>
          )}

          {/* 删除按钮 */}
          {showConfirm ? (
            <div className="flex items-center gap-1">
              <span className="text-xs text-gray-500">确认?</span>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="text-xs px-2 py-0.5 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
              >
                {deleting ? "…" : "删除"}
              </button>
              <button
                onClick={() => setShowConfirm(false)}
                disabled={deleting}
                className="text-xs px-2 py-0.5 bg-gray-200 text-gray-600 rounded hover:bg-gray-300"
              >
                否
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowConfirm(true)}
              title="删除任务"
              className="opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-red-500 p-1 rounded"
            >
              <TrashIcon />
            </button>
          )}
        </div>
      </div>

      {/* 头部 */}
      <div className="flex items-center justify-between mb-2 pr-20">
        <div className="flex items-center gap-2">
          <code className="text-sm font-mono text-gray-500">{shortId}…</code>
          <StatusBadge status={task.status} />
          {isVideo && <span className="text-xs bg-purple-100 text-purple-600 px-1.5 py-0.5 rounded">视频</span>}
        </div>
        <span className="text-xs text-gray-400">{timeStr}</span>
      </div>

      {/* 详情 */}
      <div className="text-xs text-gray-500 mb-2">
        {isVideo
          ? `${task.video_config?.max_frames || task.video_result?.selected || "?"} 帧 (${task.input?.video_count || task.input?.file_count || "?"} 段视频)`
          : `${task.input?.file_count || "?"} 张图片`}
        {task.mode === "scene" ? " · 场景模式" : " · 物体模式"}
      </div>

      {/* 进度条 — 活跃任务显示 */}
      {(isActive || task.status === "completed" || task.status === "failed" || task.status === "cancelled") && (
        <div className="mt-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-500">
              {progress.label}
              {progress.remaining && (
                <span className="ml-2 text-gray-400">({progress.remaining})</span>
              )}
            </span>
            <span className="text-xs font-mono text-gray-500">{progress.pct}%</span>
          </div>
          <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000 ease-linear ${progressColor} ${
                isActive ? "progress-bar-animated" : ""
              }`}
              style={{ width: `${progress.pct}%` }}
            />
          </div>
        </div>
      )}

      {/* 取消错误提示 */}
      {cancelError && (
        <div className="mt-2 p-2 bg-amber-50 border border-amber-100 rounded text-xs text-amber-700">
          取消失败: {cancelError}
        </div>
      )}

      {/* 已取消提示 */}
      {task.status === "cancelled" && (
        <div className="mt-2 p-3 bg-amber-50 border border-amber-200 rounded-lg text-sm text-amber-700">
          此任务已被手动终止。
        </div>
      )}

      {/* 错误 */}
      {task.status === "failed" && task.error && (
        <div className="mt-2 p-2 bg-red-50 border border-red-100 rounded text-xs text-red-700 font-mono whitespace-pre-wrap max-h-24 overflow-y-auto">
          {task.error.slice(-500)}
        </div>
      )}

      {/* 结果 */}
      {task.status === "completed" && task.output && (
        <ResultPanel taskId={task.task_id} output={task.output} />
      )}
    </div>
  );
}

function StopIcon() {
  return (
    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
      <rect x="4" y="4" width="16" height="16" rx="2" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}
