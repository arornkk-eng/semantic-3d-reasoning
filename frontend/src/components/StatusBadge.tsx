import type { TaskStatus } from "../types";

const STATUS_STYLE: Record<TaskStatus, string> = {
  waiting: "bg-gray-100 text-gray-600 border-gray-300",
  running: "bg-blue-50 text-blue-700 border-blue-300",
  completed: "bg-green-50 text-green-700 border-green-300",
  failed: "bg-red-50 text-red-700 border-red-300",
};

const STATUS_LABEL: Record<TaskStatus, string> = {
  waiting: "排队中",
  running: "重建中",
  completed: "已完成",
  failed: "失败",
};

interface Props {
  status: TaskStatus;
}

export default function StatusBadge({ status }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium border ${STATUS_STYLE[status]}`}
    >
      {status === "running" && <Spinner />}
      {STATUS_LABEL[status]}
    </span>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin h-3 w-3 text-blue-600"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle
        className="opacity-25"
        cx="12" cy="12" r="10"
        stroke="currentColor" strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  );
}
