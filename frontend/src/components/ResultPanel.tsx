import type { TaskOutput } from "../types";
import { getDownloadUrl } from "../api/client";

interface Props {
  taskId: string;
  output: TaskOutput;
}

/** 展示重建结果：统计 + PLY 下载 + SuperSplat 3D 预览。 */
export default function ResultPanel({ taskId, output }: Props) {
  const plyUrl = getDownloadUrl(taskId, "scene.ply");
  const fullPlyUrl = window.location.origin + plyUrl;
  // URL 参数优化画质: hpr=高精度渲染, aa=抗锯齿
  const viewerUrl = `/splat-viewer/index.html?content=${encodeURIComponent(fullPlyUrl)}&hpr=1&aa`;

  return (
    <div className="mt-3 p-4 bg-green-50 border border-green-200 rounded-lg">
      <h4 className="text-sm font-semibold text-green-800 mb-2">重建完成</h4>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <StatCard label="高斯球数" value={output.num_gaussians.toLocaleString()} />
        <StatCard label="PLY 大小" value={formatBytes(output.ply_size)} />
        <StatCard label="文件名" value={output.ply} />
      </div>

      <div className="flex flex-wrap gap-2">
        <a
          href={plyUrl}
          download
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 transition-colors"
        >
          <DownloadIcon />
          下载 scene.ply
        </a>

        <a
          href={viewerUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-purple-600 text-white font-medium rounded-lg hover:bg-purple-700 transition-colors"
        >
          🔍 SuperSplat 3D 预览
        </a>
      </div>

      <p className="mt-2 text-xs text-gray-500">
        一键直达 SuperSplat 查看器，PLY 自动加载
      </p>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded p-2 text-center border border-green-100">
      <div className="text-xs text-gray-500 mb-0.5">{label}</div>
      <div className="text-sm font-semibold text-gray-800 truncate" title={value}>
        {value}
      </div>
    </div>
  );
}

function DownloadIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
      />
    </svg>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
