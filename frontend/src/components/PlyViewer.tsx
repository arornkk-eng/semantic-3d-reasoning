import { useRef, useState, useCallback } from "react";
import { uploadPlyForViewing } from "../api/client";

export default function PlyViewer() {
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{ filename: string; size: number; url: string } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".ply")) {
      setError("仅支持 .ply 格式的 3D 高斯点云文件");
      return;
    }
    if (file.size > 300 * 1024 * 1024) {
      setError(`文件过大（${(file.size / 1024 / 1024).toFixed(1)} MB），最大 300 MB`);
      return;
    }
    setError(null);
    setUploading(true);
    try {
      const data = await uploadPlyForViewing(file);
      setResult(data);
      // 在新标签页打开查看器，传 ply_id 用于 3D 聚类
      const backendHost = window.location.hostname === "localhost"
        ? "http://localhost:8000"
        : window.location.origin;
      const contentUrl = backendHost + data.url;
      const viewerUrl = `/splat-viewer/index.html?content=${encodeURIComponent(contentUrl)}&hpr=1&aa&ply_id=${encodeURIComponent(data.ply_id)}`;
      window.open(viewerUrl, "_blank");
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
      e.target.value = "";
    }
  };

  return (
    <div>
      {/* 拖拽区域 */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragOver
            ? "border-purple-400 bg-purple-50"
            : "border-gray-300 hover:border-purple-400 hover:bg-gray-50"
        } ${uploading ? "pointer-events-none opacity-60" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".ply"
          onChange={handleSelect}
          className="hidden"
        />
        <div className="text-4xl mb-3">📂</div>
        <p className="text-gray-700 font-medium">
          {uploading ? "上传中…" : "拖拽 .ply 文件到这里，或点击选择"}
        </p>
        <p className="text-gray-400 text-sm mt-1">
          支持 3D 高斯点云 PLY 格式 · 最大 300 MB
        </p>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {/* 成功提示 */}
      {result && (
        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg text-sm">
          <p className="text-green-800 font-medium">已加载: {result.filename}</p>
          <p className="text-green-600 mt-1">
            {(result.size / 1024 / 1024).toFixed(1)} MB · 查看器已在新标签页打开
          </p>
        </div>
      )}

      {/* 使用说明 */}
      <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-600">
        <strong>📌 使用说明：</strong>
        直接将任意 3DGS 工具（PostShot / Nerfstudio / Luma AI / ZipSplat）导出的
        .ply 文件拖入上方区域，即可在 SuperSplat 查看器中预览 3D 模型。
      </div>
    </div>
  );
}
