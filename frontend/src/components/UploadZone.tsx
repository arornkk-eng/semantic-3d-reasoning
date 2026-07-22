import { useCallback, useRef, useState } from "react";
import { uploadImages } from "../api/client";
import FilePreview from "./FilePreview";
import type { UploadResponse } from "../types";

const ALLOWED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"];
const MAX_SIZE = 50 * 1024 * 1024; // 50 MB
const MAX_FILES = 50;

interface Props {
  onTaskCreated: (taskId: string) => void;
}

export default function UploadZone({ onTaskCreated }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback((newFiles: File[]): string | null => {
    const combined = [...files, ...newFiles];
    if (combined.length > MAX_FILES) {
      return `最多上传 ${MAX_FILES} 张图片，当前已有 ${files.length} 张，试图添加 ${newFiles.length} 张`;
    }
    for (const f of newFiles) {
      const ext = f.name.toLowerCase().slice(f.name.lastIndexOf("."));
      if (!ALLOWED_EXTS.includes(ext)) {
        return `不支持的文件格式: ${f.name}（支持: ${ALLOWED_EXTS.join(", ")}）`;
      }
    }
    const total = combined.reduce((s, f) => s + f.size, 0);
    if (total > MAX_SIZE) {
      return `总大小 ${(total / 1024 / 1024).toFixed(1)} MB 超出限制（50 MB）`;
    }
    return null;
  }, [files]);

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

  const handleUpload = async () => {
    if (files.length === 0) {
      setError("请先选择图片");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const result: UploadResponse = await uploadImages(files);
      onTaskCreated(result.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败");
    } finally {
      setUploading(false);
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
          accept={ALLOWED_EXTS.join(",")}
          onChange={handleSelect}
          className="hidden"
        />
        <div className="text-5xl mb-3">📁</div>
        <p className="text-gray-700 font-medium text-base">
          拖拽图片到这里，或点击选择
        </p>
        <p className="text-gray-400 text-sm mt-1">
          支持 JPG / PNG / BMP / WebP · 最多 50 张 · 总大小 ≤ 50 MB
        </p>
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
            `开始重建（${files.length} 张图片）`
          )}
        </button>
      )}
    </div>
  );
}
