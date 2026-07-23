import type {
  HealthResponse,
  UploadResponse,
  TaskMeta,
  ResultResponse,
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

/** 健康检查 + GPU 信息 */
export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/** 上传图片，返回任务 ID */
export async function uploadImages(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));

  const res = await fetch(`${BASE}/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
    throw new Error(detail || `上传失败 (HTTP ${res.status})`);
  }

  return res.json();
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

/** 取消任务（等待中移出队列，运行中终止进程） */
export async function cancelTask(taskId: string): Promise<{ task_id: string; cancelled: boolean }> {
  const res = await fetch(`${BASE}/task/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
  });
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
    throw new Error(detail || `取消失败 (HTTP ${res.status})`);
  }
  return res.json();
}

/** 删除任务及其所有文件 */
export async function deleteTask(taskId: string): Promise<void> {
  const res = await fetch(`${BASE}/task/${encodeURIComponent(taskId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const detail = await res.json().then((d) => d.detail).catch(() => res.statusText);
    throw new Error(detail || `删除失败 (HTTP ${res.status})`);
  }
}

