import { useMemo } from "react";

interface Props {
  files: File[];
  onRemove: (index: number) => void;
}

/**
 * 已选图片缩略图列表，使用 URL.createObjectURL 生成预览。
 */
export default function FilePreview({ files, onRemove }: Props) {
  const previews = useMemo(
    () => files.map((f) => URL.createObjectURL(f)),
    [files]
  );

  if (files.length === 0) return null;

  const totalSize = files.reduce((s, f) => s + f.size, 0);

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-600">
          已选择 <strong>{files.length}</strong> 张图片
          {" · "}
          {formatBytes(totalSize)}
        </span>
        <button
          onClick={() => files.forEach((_, i) => onRemove(i))}
          className="text-xs text-red-500 hover:text-red-700"
        >
          清空全部
        </button>
      </div>
      <div className="flex flex-wrap gap-2 max-h-40 overflow-y-auto">
        {files.map((f, i) => (
          <div key={i} className="relative group">
            <img
              src={previews[i]}
              alt={f.name}
              className="h-20 w-20 object-cover rounded border border-gray-200"
            />
            <button
              onClick={() => onRemove(i)}
              className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-red-500 text-white rounded-full text-xs flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
              title="移除"
            >
              ×
            </button>
            <span
              className="block text-[10px] text-gray-500 truncate w-20 text-center"
              title={f.name}
            >
              {f.name}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
