import { useState, useCallback, useEffect } from "react";
import { useHealth } from "./hooks/useHealth";
import HealthBadge from "./components/HealthBadge";
import PlyViewer from "./components/PlyViewer";
import UploadZone from "./components/UploadZone";
import TaskList from "./components/TaskList";
import { listTasks } from "./api/client";
import type { TaskHistoryEntry } from "./types";

const HISTORY_KEY = "zipsplat_task_history";

type Tab = "upload" | "tasks" | "ply-viewer" | "about";

function loadHistory(): TaskHistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: TaskHistoryEntry[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
}

function addToHistory(taskId: string): TaskHistoryEntry[] {
  const existing = loadHistory();
  const filtered = existing.filter((e) => e.task_id !== taskId);
  const entry: TaskHistoryEntry = {
    task_id: taskId,
    created_at: new Date().toISOString(),
  };
  const updated = [...filtered, entry];
  saveHistory(updated);
  return updated;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>("upload");
  const [taskIds, setTaskIds] = useState<string[]>(() =>
    loadHistory().map((e) => e.task_id)
  );
  const { health, error: healthError } = useHealth();

  const handleTaskCreated = useCallback((taskId: string) => {
    const updated = addToHistory(taskId);
    setTaskIds(updated.map((e) => e.task_id));
    setActiveTab("tasks");
  }, []);

  const handleTaskDeleted = useCallback((taskId: string) => {
    const existing = loadHistory();
    const filtered = existing.filter((e) => e.task_id !== taskId);
    saveHistory(filtered);
    setTaskIds(filtered.map((e) => e.task_id));
  }, []);

  // 同步跨标签页的 localStorage 变更
  useEffect(() => {
    const onStorage = () => {
      setTaskIds(loadHistory().map((e) => e.task_id));
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // 启动时从服务端拉取任务列表（合并跨设备的任务）
  useEffect(() => {
    listTasks()
      .then((data) => {
        const localIds = new Set(loadHistory().map((e) => e.task_id));
        let changed = false;
        for (const id of data.task_ids) {
          if (!localIds.has(id)) {
            addToHistory(id);
            changed = true;
          }
        }
        if (changed) {
          setTaskIds(loadHistory().map((e) => e.task_id));
        }
      })
      .catch(() => {}); // 静默失败
  }, []);

  const tabs: { key: Tab; label: string }[] = [
    { key: "upload", label: "📤 上传" },
    { key: "tasks", label: `📋 任务 (${taskIds.length})` },
    { key: "ply-viewer", label: "📂 PLY 查看" },
    { key: "about", label: "ℹ️ 关于" },
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* 顶部 GPU 状态栏 */}
      <HealthBadge health={health} error={healthError} />

      {/* 标签页导航 */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-3xl mx-auto px-4">
          <div className="flex items-center gap-6">
            <h1 className="text-lg font-bold text-gray-800 mr-4">
              ZipSplat Demo
            </h1>
            <nav className="flex gap-1">
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`px-4 py-3 text-sm font-medium transition-colors border-b-2 ${
                    activeTab === tab.key
                      ? "tab-active"
                      : "text-gray-500 border-transparent hover:text-gray-700 hover:border-gray-300"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* 主内容 */}
      <main className="flex-1 max-w-3xl mx-auto px-4 py-8 w-full">
        {activeTab === "upload" && (
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              上传图片开始 3D 重建
            </h2>
            <UploadZone onTaskCreated={handleTaskCreated} />
            <div className="mt-6 p-4 bg-blue-50 border border-blue-100 rounded-lg text-sm text-blue-800">
              <strong>💡 拍摄建议：</strong>
              拍摄整个空间的全景照片 10~50 张或录制视频，覆盖所有视角。
              保持相邻照片之间有足够的重叠（约 60%~80%），
              光线均匀、避免过曝或欠曝。完整环境将被保留。
            </div>
          </section>
        )}

        {activeTab === "tasks" && (
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              重建任务
            </h2>
            <TaskList taskIds={taskIds} onDelete={handleTaskDeleted} />
          </section>
        )}

        {activeTab === "ply-viewer" && (
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              直接查看 PLY 文件
            </h2>
            <PlyViewer />
          </section>
        )}

        {activeTab === "about" && (
          <section>
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              关于 ZipSplat Demo
            </h2>
            <div className="space-y-4">
              <div className="p-4 bg-white border border-gray-200 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">技术原理</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  ZipSplat 是一种前馈式 3D 高斯泼溅（Feed-Forward 3D Gaussian
                  Splatting）方法，使用 DA3-Giant 主干网络（基于 ViT-G），仅需单次前向传播即可从
                  多视角图片中预测 3D 高斯点云，无需逐场景优化。
                </p>
              </div>

              <div className="p-4 bg-white border border-gray-200 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">硬件要求</h3>
                <ul className="text-gray-600 text-sm space-y-1 list-disc list-inside">
                  <li>NVIDIA GPU ≥ 6 GB VRAM（RTX 系列推荐）</li>
                  <li>CUDA 12.x + PyTorch 2.x</li>
                  <li>系统内存 ≥ 16 GB</li>
                </ul>
              </div>

              <div className="p-4 bg-white border border-gray-200 rounded-lg">
                <h3 className="font-semibold text-gray-800 mb-2">版本信息</h3>
                <ul className="text-gray-600 text-sm space-y-1">
                  <li>后端: FastAPI + ZipSplat (DA3-Giant)</li>
                  <li>前端: React + TypeScript + Vite</li>
                  <li>推理: 单 GPU 串行，queue.Queue 任务队列</li>
                  <li>输出: PLY 格式高斯点云</li>
                </ul>
              </div>
            </div>
          </section>
        )}
      </main>

      {/* 页脚 */}
      <footer className="text-center py-4 text-xs text-gray-400 border-t border-gray-200">
        ZipSplat-Demo · 本地 AI 3D 重建 ·{" "}
        {health?.gpu_name ? `运行在 ${health.gpu_name}` : "加载中…"}
      </footer>

    </div>
  );
}
