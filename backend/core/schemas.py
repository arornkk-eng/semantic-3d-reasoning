"""API 响应契约（Pydantic models）。

集中定义响应模型，让 OpenAPI 文档可用、前端对接有类型、返回字段受控。
这是「后端契约化」的第一步：后续端点逐步迁移到 response_model，
替换原先直接返回裸 dict 的写法，避免字段漂移与前后端对接踩坑。
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

MAX_GAUSSIAN_INDICES_PER_SET = 1_000_000
MAX_GAUSSIAN_INDICES_TOTAL = 2_000_000
MAX_GAUSSIAN_INDEX_SETS = 200
UINT32_MAX = 2**32 - 1

StrictSourceIndex = Annotated[StrictInt, Field(ge=0, le=65_535)]
StrictVertexCount = Annotated[StrictInt, Field(gt=0, le=UINT32_MAX + 1)]
StrictGaussianIndex = Annotated[StrictInt, Field(ge=0, le=UINT32_MAX)]


class UploadResponse(BaseModel):
    task_id: str
    status: str
    file_count: int
    queue_position: int


class VideoUploadResponse(BaseModel):
    task_id: str
    status: str
    video_count: int
    queue_position: int


class PlyUploadResponse(BaseModel):
    ply_id: str
    filename: str
    size: int
    url: str


class HealthResponse(BaseModel):
    status: str
    queue_size: int
    gpu_name: str | None = None
    gpu_memory_total: str | None = None
    cuda_version: str | None = None


class SegmentationPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    label: int = Field(ge=0, le=1)


class SegmentationPredictRequest(BaseModel):
    points: list[SegmentationPoint] = Field(min_length=1, max_length=64)


class SegmentationSessionResponse(BaseModel):
    session_id: str
    width: int
    height: int
    expires_in: int


class SegmentationPredictResponse(BaseModel):
    session_id: str
    score: float
    bbox: list[int]
    mask_url: str


class GaussianIndexFileResponse(BaseModel):
    source_index: int
    encoding: str
    count: int
    vertex_count: int
    url: str


class SegmentationLayerResponse(BaseModel):
    layer_id: str
    task_id: str
    name: str
    mask_url: str
    created_at: str
    gaussian_indices: list[GaussianIndexFileResponse] = Field(default_factory=list)
    source_ply_sha256: str | None = None
    source_ply_sha256_status: str | None = None
    category: str | None = None
    category_zh: str | None = None
    instance_index: int | None = None
    observation_count: int = 1


class SemanticViewMaskResponse(BaseModel):
    view_index: int = Field(ge=0)
    mask_url: str


class SemanticInstanceResponse(BaseModel):
    instance_id: str
    category: str
    category_zh: str
    instance_index: int
    score: float
    bbox: list[int]
    mask_url: str
    depth_coverage: float | None = None
    view_support: int = 1
    view_count: int = 1
    view_masks: list[SemanticViewMaskResponse] = Field(default_factory=list)


class SemanticPredictResponse(BaseModel):
    result_id: str
    instances: list[SemanticInstanceResponse]


class RealtimeDetectionBox(BaseModel):
    category: str
    score: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_bbox(self):
        x1, y1, x2, y2 = self.bbox
        if any(value < 0 or value > 1 for value in self.bbox):
            raise ValueError("实时检测 bbox 必须是 0 至 1 的归一化坐标")
        if x2 < x1 or y2 < y1:
            raise ValueError("实时检测 bbox 坐标顺序无效")
        return self


class RealtimeDetectionResponse(BaseModel):
    frame_id: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    inference_ms: float = Field(ge=0)
    detections: list[RealtimeDetectionBox]


class GeometricRefineMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=64)
    instance_id: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    source_index: StrictSourceIndex
    source_vertex_count: StrictVertexCount
    scene_radius: float = Field(gt=0)
    seed_indices: list[StrictGaussianIndex] = Field(
        min_length=1, max_length=MAX_GAUSSIAN_INDICES_PER_SET
    )
    candidate_indices: list[StrictGaussianIndex] = Field(
        default_factory=list, max_length=MAX_GAUSSIAN_INDICES_PER_SET
    )

    @model_validator(mode="after")
    def validate_indices(self):
        for values in (self.seed_indices, self.candidate_indices):
            if len(values) != len(set(values)):
                raise ValueError("Gaussian 索引不能重复")
            if any(index >= self.source_vertex_count for index in values):
                raise ValueError("Gaussian 索引超出 source_vertex_count 范围")
        return self


class GeometricRefineResponse(BaseModel):
    instance_id: str
    source_index: int
    source_vertex_count: int
    indices: list[int]
    seed_count: int
    added_count: int
    engine: str
    duration_ms: float


class SemanticGaussianIndexSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1, max_length=64)
    source_index: StrictSourceIndex
    source_vertex_count: StrictVertexCount
    indices: list[StrictGaussianIndex] = Field(
        default_factory=list,
        max_length=MAX_GAUSSIAN_INDICES_PER_SET,
    )

    @model_validator(mode="after")
    def validate_indices(self):
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("indices 不能重复")
        if any(index >= self.source_vertex_count for index in self.indices):
            raise ValueError("indices 超出 source_vertex_count 范围")
        return self


def _validate_gaussian_index_set_collection(
    index_sets: list[SemanticGaussianIndexSet],
) -> None:
    keys: set[tuple[str, int]] = set()
    total = 0
    for index_set in index_sets:
        key = (index_set.instance_id, index_set.source_index)
        if key in keys:
            raise ValueError("同一实例的 source_index 不能重复")
        keys.add(key)
        total += len(index_set.indices)
    if total > MAX_GAUSSIAN_INDICES_TOTAL:
        raise ValueError("Gaussian indices 总数超出限制")


class LayerMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(min_length=1, max_length=64)
    instance_id: str = Field(min_length=1, max_length=64)
    gaussian_index_sets: list[SemanticGaussianIndexSet] = Field(
        min_length=1,
        max_length=MAX_GAUSSIAN_INDEX_SETS,
    )

    @model_validator(mode="after")
    def validate_index_sets(self):
        if any(item.instance_id != self.instance_id for item in self.gaussian_index_sets):
            raise ValueError("Gaussian index set 与增量观测实例不匹配")
        _validate_gaussian_index_set_collection(self.gaussian_index_sets)
        return self


class LayerMergeResponse(BaseModel):
    layer: SegmentationLayerResponse
    added_count: int
    total_count: int
    observation_count: int


class LayerCreateRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=80)
    gaussian_index_sets: list[SemanticGaussianIndexSet] = Field(
        default_factory=list,
        max_length=MAX_GAUSSIAN_INDEX_SETS,
    )

    @model_validator(mode="after")
    def validate_index_sets(self):
        if any(index_set.instance_id != self.session_id for index_set in self.gaussian_index_sets):
            raise ValueError("Gaussian index set 不属于当前分割会话")
        _validate_gaussian_index_set_collection(self.gaussian_index_sets)
        return self


class SemanticConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_ids: list[str] = Field(min_length=1, max_length=50)
    gaussian_index_sets: list[SemanticGaussianIndexSet] = Field(
        default_factory=list,
        max_length=MAX_GAUSSIAN_INDEX_SETS,
    )

    @model_validator(mode="after")
    def validate_index_sets(self):
        selected = set(self.instance_ids)
        if len(selected) != len(self.instance_ids):
            raise ValueError("instance_ids 不能重复")
        for index_set in self.gaussian_index_sets:
            if index_set.instance_id not in selected:
                raise ValueError("Gaussian index set 不属于已选实例")
        _validate_gaussian_index_set_collection(self.gaussian_index_sets)
        return self


class LayerRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)


class SceneSnapshotCamera(BaseModel):
    model_config = ConfigDict(extra="forbid")
    view_matrix: list[float] = Field(min_length=16, max_length=16)
    projection_matrix: list[float] = Field(min_length=16, max_length=16)
    position: list[float] = Field(min_length=3, max_length=3)
    rotation: list[float] = Field(min_length=4, max_length=4)
    focal_point: list[float] = Field(min_length=3, max_length=3)
    azim: float
    elevation: float
    distance: float = Field(gt=0)
    fov: float = Field(gt=0, lt=180)


class SceneSnapshotObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer_id: str = Field(min_length=1, max_length=32)
    name: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=64)
    center_camera: list[float] = Field(min_length=3, max_length=3)
    bounds_min_camera: list[float] = Field(min_length=3, max_length=3)
    bounds_max_camera: list[float] = Field(min_length=3, max_length=3)


class SceneSnapshotRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(min_length=1, max_length=32)
    predicate: Literal[
        "left_of",
        "right_of",
        "front_of",
        "behind",
        "left_front_of",
        "right_front_of",
        "left_behind",
        "right_behind",
        "near",
        "overlap",
    ]
    object: str = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0, le=1)


class SceneSnapshotCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera: SceneSnapshotCamera
    objects: list[SceneSnapshotObject] = Field(min_length=1, max_length=10)
    relations: list[SceneSnapshotRelation] = Field(default_factory=list, max_length=300)
    functions: dict[str, list[str]] = Field(default_factory=dict)
    description: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_references(self):
        layer_ids = [item.layer_id for item in self.objects]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("快照对象图层不能重复")
        known = set(layer_ids)
        if any(item.subject not in known or item.object not in known for item in self.relations):
            raise ValueError("空间关系引用了快照之外的图层")
        return self


class SceneSnapshotRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=80)


class SceneSnapshotResponse(SceneSnapshotCreateRequest):
    snapshot_id: str
    task_id: str
    name: str
    sequence: int
    created_at: str
