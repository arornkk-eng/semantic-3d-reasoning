import { useCallback, useRef, useState } from "react";
import { uploadImages, uploadVideos } from "../api/client";
import FilePreview from "./FilePreview";
import type { UploadResponse, VideoUploadResponse } from "../types";

const IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv"];
const MAX_IMG_SIZE = 100 * 1024 * 1024; // 100 MB
const MAX_VIDEO_SIZE = 500 * 1024 * 1024; // 500 MB per video
const MAX_FILES = 50;
const MAX_VIDEOS = 10;
const MAX_FRAMES_DEFAULT = 25;  // RTX 4050 6.4GB 安全上限
const DEFAULT_MODE = "scene";    // 固定场景模式

type InputMode = "image" | "video";

/** 移动端快捷入口类型 */
type QuickAction = "camera-photo" | "camera-video" | "library-photo" | "library-video";

interface Props {
  onTaskCreated: (taskId: string) => void;
}

export default function UploadZone({ onTaskCreated }: Props) {
  const [inputMode, setInputMode] = useState<InputMode>("image");
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const quickRef = useRef<HTMLInputElement>(null);
  const [quickAccept, setQuickAccept] = useState("");
  const [quickCapture, setQuickCapture] = useState<boolean | undefined>(undefined);
  const [quickMultiple, setQuickMultiple] = useState(false);

  const isImage = inputMode === "image";
  const allowedExts = isImage ? IMG_EXTS : VIDEO_EXTS;
  const maxCount = isImage ? MAX_FILES : MAX_VIDEOS;
  const maxSize = isImage ? MAX_IMG_SIZE : MAX_VIDEO_SIZE;
  const label = isImage ? "图片" : "视频";

  const validate = useCallback((newFiles: File[]): string | null => {
    const combined = [...files, ...newFiles];
    if (combined.length > maxCount) {
      return `最多上传 ${maxCount} 个${label}，当前已有 ${files.length} 个，试图添加 ${newFiles.length} 个`;
    }
    for (const f of newFiles) {
      const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
      if (!allowedExts.includes(ext)) {
        return `不支持的文件格式: ${f.name}（支持: ${allowedExts.join(", ")}）`;
      }
      if (f.size > maxSize) {
        return `${f.name} 大小 ${(f.size / 1024 / 1024).toFixed(1)} MB 超出限制（${maxSize / 1024 / 1024} MB）`;
      }
    }
    const total = combined.reduce((s, f) => s + f.size, 0);
    if (isImage && total > MAX_IMG_SIZE) {
      return `总大小 ${(total / 1024 / 1024).toFixed(1)} MB 超出限制（100 MB）`;
    }
    return null;
  }, [files, isImage, maxCount, maxSize, label, allowedExts]);

  const addFiles = useCallback(
    (newFiles: File[]) => {
      const err = validate(newFiles);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      setFiles((prev) => [...prev, ...newFiles]);
    },
    [validate]
  );

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
    setError(null);
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const dropped = Array.from(e.dataTransfer.files);
      addFiles(dropped);
    },
    [addFiles]
  );

  const handleSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addFiles(Array.from(e.target.files));
      e.target.value = ""; // 允许重选同一文件
    }
  };

  const switchMode = (m: InputMode) => {
    if (m !== inputMode) {
      setFiles([]);  // 切换时清空已选文件
      setError(null);
      setInputMode(m);
    }
  };

  const handleQuickAction = (action: QuickAction) => {
    // 切换模式并清理已选文件
    if (action === "camera-photo" || action === "library-photo") {
      switchMode("image");
      setQuickAccept("image/*");
      setQuickCapture(action === "camera-photo");
      setQuickMultiple(action === "library-photo" || action === "camera-photo");
    } else {
      switchMode("video");
      setQuickAccept("video/*");
      setQuickCapture(action === "camera-video");
      setQuickMultiple(action === "library-video");
    }
    // 延迟触发确保 state 已更新
    setTimeout(() => quickRef.current?.click(), 0);
  };

  const handleQuickChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      addFiles(Array.from(e.target.files));
      e.target.value = "";
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError(`请先选择${label}`);
      return;
    }
    setUploading(true);
    setError(null);
    try {
      if (isImage) {
        const result: UploadResponse = await uploadImages(files, DEFAULT_MODE);
        onTaskCreated(result.task_id);
      } else {
        const result: VideoUploadResponse = await uploadVideos(files, DEFAULT_MODE, MAX_FRAMES_DEFAULT);
        onTaskCreated(result.task_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      {/* 文件类型切换 */}
      <div className="flex gap-2 mb-4">
        <button
          type="button"
          onClick={() => switchMode("image")}
          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
            isImage
              ? "bg-blue-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          🖼️ 图片上传
        </button>
        <button
          type="button"
          onClick={() => switchMode("video")}
          className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
            !isImage
              ? "bg-blue-600 text-white"
              : "bg-gray-100 text-gray-600 hover:bg-gray-200"
          }`}
        >
          🎬 视频上传（智能抽帧）
        </button>
      </div>

      {/* 拖拽区域 */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${
          dragOver
            ? "border-blue-400 bg-blue-50"
            : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={allowedExts.join(",")}
          onChange={handleSelect}
          className="hidden"
        />
        {/* 移动端快捷入口：拍照用 capture="environment"，相册不设 capture */}
        <input
          ref={quickRef}
          type="file"
          accept={quickAccept}
          {...(quickCapture ? { capture: "environment" } : {})}
          multiple={quickMultiple}
          onChange={handleQuickChange}
          className="hidden"
        />
        <div className="text-5xl mb-3">{isImage ? "📁" : "🎬"}</div>
        <p className="text-gray-700 font-medium text-base">
          {isImage
            ? "拖拽图片到这里，或点击选择"
            : "拖拽视频到这里，或点击选择"}
        </p>
        <p className="text-gray-400 text-sm mt-1">
          {isImage
            ? "支持 JPG / PNG / BMP / WebP · 最多 50 张 · 总大小 ≤ 100 MB"
            : "支持 MP4 / AVI / MOV / MKV · 最多 10 段 · 单段 ≤ 500 MB"}
        </p>
      </div>

      {/* 移动端快捷入口：拍照 / 录像 / 相册 */}
      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <button
          type="button"
          onClick={() => handleQuickAction("camera-photo")}
          className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl
                     bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200
                     text-blue-700 font-medium text-sm
                     hover:from-blue-100 hover:to-blue-200 active:scale-95 transition-all"
        >
          <span className="text-xl">📷</span>
          <span>拍照</span>
        </button>
        <button
          type="button"
          onClick={() => handleQuickAction("camera-video")}
          className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl
                     bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200
                     text-purple-700 font-medium text-sm
                     hover:from-purple-100 hover:to-purple-200 active:scale-95 transition-all"
        >
          <span className="text-xl">🎥</span>
          <span>录像</span>
        </button>
        <button
          type="button"
          onClick={() => handleQuickAction("library-photo")}
          className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl
                     bg-gradient-to-br from-green-50 to-green-100 border border-green-200
                     text-green-700 font-medium text-sm
                     hover:from-green-100 hover:to-green-200 active:scale-95 transition-all"
        >
          <span className="text-xl">🖼️</span>
          <span>相册</span>
        </button>
        <button
          type="button"
          onClick={() => handleQuickAction("library-video")}
          className="flex items-center justify-center gap-2 px-3 py-3 rounded-xl
                     bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200
                     text-orange-700 font-medium text-sm
                     hover:from-orange-100 hover:to-orange-200 active:scale-95 transition-all"
        >
          <span className="text-xl">📂</span>
          <span>视频库</span>
        </button>
      </div>

      {/* 预览 */}
      <FilePreview files={files} onRemove={removeFile} />

      {/* 错误提示 */}
      {error && (
        <div className="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {error}
        </div>
      )}

      {/* 上传按钮 */}
      {files.length > 0 && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="mt-4 w-full py-3 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {uploading ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
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
              上传并重建中…
            </span>
          ) : (
            isImage
              ? `开始重建（${files.length} 张图片）`
              : `上传并智能抽帧（${files.length} 段视频）`
          )}
        </button>
      )}
    </div>
  );
}
