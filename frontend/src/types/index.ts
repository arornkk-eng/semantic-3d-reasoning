// ---- API 响应类型 ----

export type TaskStatus = "waiting" | "running" | "completed" | "failed";

export interface HealthResponse {
  status: string;
  gpu_name?: string;
  gpu_memory_total?: string;
  cuda_version?: string;
  queue_size: number;
}

export interface UploadResponse {
  task_id: string;
  status: "waiting";
  file_count: number;
  queue_position: number;
}

export interface TaskInput {
  file_count: number;
  filenames: string[];
}

export interface TaskOutput {
  ply: string;
  ply_size: number;
  num_gaussians: number;
}

export interface TaskMeta {
  task_id: string;
  status: TaskStatus;
  created_at: string;
  updated_at: string;
  input: TaskInput;
  output: TaskOutput | null;
  error: string | null;
}

export interface OutputFile {
  name: string;
  size: number;
  url: string;
}

export interface ResultResponse {
  task_id: string;
  status: "completed";
  files: OutputFile[];
}

// ---- 前端内部类型 ----

export interface TaskHistoryEntry {
  task_id: string;
  created_at: string;
}
