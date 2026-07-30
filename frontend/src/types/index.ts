// ---- API 响应类型 ----

export type TaskStatus = "waiting" | "running" | "completed" | "failed" | "cancelled";
export type ReconstructMode = "object" | "scene";

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

export interface VideoUploadResponse {
  task_id: string;
  status: "waiting";
  video_count: number;
  queue_position: number;
}

export interface TaskInput {
  file_count?: number;
  video_count?: number;
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
  type?: "image" | "video";
  mode: ReconstructMode;
  created_at: string;
  updated_at: string;
  started_at?: string;
  estimated_seconds?: number;
  stage?: string;
  progress_pct?: number;
  input: TaskInput;
  output: TaskOutput | null;
  error: string | null;
  video_config?: { max_frames: number; sample_interval: number };
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

// ---- 物体识别类型 ----

export interface RecognizeRequest {
  objects: string[];
  box_threshold?: number;
  use_clip?: boolean;   // CLIP 语义验证（默认 true）
  use_sam?: boolean;    // SAM 精确分割（默认 true）
}

export interface ObjectLabel {
  count: number;
  score: number;
}

export interface RecognizeResponse {
  task_id: string;
  status: string;
  labels: Record<string, ObjectLabel>;
}

export interface ObjectLabelDetail {
  count: number;
  score: number;
  center_3d?: number[];
  bbox_3d?: { min: number[]; max: number[] };
}

export interface LabelsResponse {
  task_id: string;
  objects: Record<string, ObjectLabelDetail>;
}

// ---- 3D 自监督发现类型 ----

export interface DiscoverRequest {
  n_clusters?: number;
  n_samples?: number;
  pos_weight?: number;
  col_weight?: number;
  min_cluster_size?: number;
}

export interface ClusterInfo {
  cluster_id: number;
  count: number;
  ratio: number;
  center_3d: number[];
  bbox_3d: { min: number[]; max: number[] };
  dominant_color_rgb: number[];
}

export interface DiscoverResponse {
  task_id: string;
  status: string;
  n_clusters_found: number;
  clusters: ClusterInfo[];
}

export interface ClustersResponse {
  task_id: string;
  method?: string;
  n_clusters_found: number;
  total_vertices: number;
  clusters: ClusterInfo[];
}

