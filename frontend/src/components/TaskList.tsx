import TaskCard from "./TaskCard";

interface Props {
  taskIds: string[];
  onDelete: (taskId: string) => void;
}

/** 任务历史列表。 */
export default function TaskList({ taskIds, onDelete }: Props) {
  if (taskIds.length === 0) {
    return (
      <div className="text-center py-16 text-gray-400">
        <div className="text-4xl mb-3">📭</div>
        <p>暂无任务</p>
        <p className="text-sm mt-1">上传图片开始你的第一次 3D 重建</p>
      </div>
    );
  }

  // 最新的排在最前面
  const reversed = [...taskIds].reverse();

  return (
    <div className="space-y-3">
      {reversed.map((id) => (
        <TaskCard key={id} taskId={id} onDelete={onDelete} />
      ))}
    </div>
  );
}
