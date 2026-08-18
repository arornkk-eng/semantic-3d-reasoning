import type { Splat } from '../splat';

export type PromptPoint = { x: number; y: number; label: 0 | 1 };
export type SemanticViewMask = { view_index: number; mask_url: string };
export type SemanticInstance = {
    instance_id: string;
    category: string;
    category_zh: string;
    instance_index: number;
    score: number;
    mask_url: string;
    selected: boolean;
    depth_coverage?: number;
    view_support?: number;
    view_count?: number;
    view_masks?: SemanticViewMask[];
};

export type CapturedView = {
    image: Blob;
    depth: Blob;
    depthValues: Float32Array;
    depthCoverage: Float32Array;
    metadata: {
        task_id: string | null;
        source_ply: string;
        viewport_width: number;
        viewport_height: number;
        capture_width: number;
        capture_height: number;
        near: number;
        far: number;
        projection: 'orthographic' | 'perspective';
        fov: number;
        camera_position: number[];
        camera_rotation: number[];
        view_matrix: number[];
        projection_matrix: number[];
    };
};

export type ProjectedSplat = {
    splat: Splat;
    indices: Uint32Array;
    expansionCandidates: Uint32Array;
    expansionApplied: boolean;
};

export type SavedLayer = {
    layer_id: string;
    name: string;
    category?: string;
    instance_index?: number;
    observation_count?: number;
    gaussian_indices: Array<{ source_index: number; count: number; vertex_count: number }>;
};
