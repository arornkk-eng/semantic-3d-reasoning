import type {
  HealthResponse,
  UploadResponse,
  VideoUploadResponse,
  TaskMeta,
  ResultResponse,
  ReconstructMode,
} from "../types";

const BASE = "/api";

async function request<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const res = await fetch(`${BASE}${url}`, options);
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
    throw new Error(detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function formRequest<T>(url: string, form: FormData): Promise<T> {
  return request<T>(url, { method: "POST", body: form });
}

/** 健康检查 + GPU 信息 */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/** 获取服务端所有任务 ID */
export function listTasks(): Promise<{ task_ids: string[] }> {
  return request<{ task_ids: string[] }>("/tasks");
}

/** 上传图片，返回任务 ID */
export function uploadImages(files: File[], mode: ReconstructMode = "object"): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("mode", mode);

  return formRequest<UploadResponse>("/upload", form);
}

/** 查询任务状态 */
export function getTask(taskId: string): Promise<TaskMeta> {
  return request<TaskMeta>(`/task/${taskId}`);
}

/** 查询任务输出文件列表 */
export function getResult(taskId: string): Promise<ResultResponse> {
  return request<ResultResponse>(`/result/${taskId}`);
}

/** 生成输出文件下载 URL */
export function getDownloadUrl(taskId: string, filename: string): string {
  return `${BASE}/result/${taskId}/${encodeURIComponent(filename)}`;
}

/** 上传视频，返回任务 ID */
export function uploadVideos(
  videos: File[],
  mode: ReconstructMode = "scene",
  maxFrames: number = 25,  // RTX 4050 6.4GB 安全上限
  sampleInterval: number = 1.0,
): Promise<VideoUploadResponse> {
  const form = new FormData();
  videos.forEach((f) => form.append("videos", f));
  form.append("mode", mode);
  form.append("max_frames", String(maxFrames));
  form.append("sample_interval", String(sampleInterval));

  return formRequest<VideoUploadResponse>("/upload-video", form);
}

/** 取消任务（等待中移出队列，运行中终止进程） */
export function cancelTask(taskId: string): Promise<{ task_id: string; cancelled: boolean }> {
  return request<{ task_id: string; cancelled: boolean }>(`/task/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
  });
}

/** 删除任务及其所有文件 */
export function deleteTask(taskId: string): Promise<void> {
  return request<void>(`/task/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  });
}

/** 上传 PLY 文件用于临时查看 */
export function uploadPlyForViewing(
  file: File
): Promise<{ ply_id: string; filename: string; size: number; url: string }> {
  const form = new FormData();
  form.append("file", file);
  return formRequest("/view-ply", form);
}
