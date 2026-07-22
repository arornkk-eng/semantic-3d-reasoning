import { useTask } from "../hooks/useTask";
import StatusBadge from "./StatusBadge";
import ResultPanel from "./ResultPanel";

interface Props {
  taskId: string;
}

/** 单个任务卡片：自动轮询状态，展示进度和结果。 */
export default function TaskCard({ taskId }: Props) {
  const { task, error } = useTask(taskId);

  if (error && !task) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-sm text-red-700">
          加载任务 {taskId.slice(0, 8)}… 失败: {error}
        </p>
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

  return (
    <div className="p-4 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow transition-shadow">
      {/* 头部 */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <code className="text-sm font-mono text-gray-500">{shortId}…</code>
          <StatusBadge status={task.status} />
        </div>
        <span className="text-xs text-gray-400">{timeStr}</span>
      </div>

      {/* 详情 */}
      <div className="text-xs text-gray-500 mb-1">
        {task.input.file_count} 张图片
      </div>

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
