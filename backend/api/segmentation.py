"""Interactive segmentation, semantic layer, and scene snapshot endpoints."""

import io
import json

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.core.config import SEGMENTATION_SESSION_TTL_SECONDS
from backend.core.schemas import (
    GeometricRefineMetadata,
    GeometricRefineResponse,
    LayerCreateRequest,
    LayerMergeRequest,
    LayerMergeResponse,
    LayerRenameRequest,
    SceneSnapshotCreateRequest,
    SceneSnapshotRenameRequest,
    SceneSnapshotResponse,
    SegmentationLayerResponse,
    SegmentationPredictRequest,
    SegmentationPredictResponse,
    SegmentationSessionResponse,
    SemanticConfirmRequest,
    SemanticGaussianIndexSet,
    SemanticInstanceResponse,
    SemanticPredictResponse,
    SemanticViewMaskResponse,
)
from backend.segmentation.geometric_refinement import refine_gaussian_selection
from backend.segmentation.mesh_generation import LayerMeshError, generate_layer_mesh
from backend.segmentation.semantic_service import semantic_service
from backend.segmentation.service import (
    SegmentationBusyError,
    SegmentationUnavailableError,
    segmentation_service,
)
from backend.storage.file_manager import get_task_meta
from backend.storage.layer_store import (
    create_layer,
    delete_layer,
    get_gaussian_indices_path,
    get_layer_metadata,
    get_mask_path,
    hash_task_scene_ply,
    list_layers,
    merge_layer_observation,
    rename_layer,
)
from backend.storage.scene_snapshot_store import (
    create_snapshot,
    delete_snapshot,
    delete_snapshots_using_layer,
    list_snapshots,
    rename_snapshot,
    snapshots_using_layer,
)

router = APIRouter()


@router.post("/semantic/refine3d", response_model=GeometricRefineResponse)
async def refine_semantic_3d(metadata: str = Form(...), geometry: UploadFile = File(...)):
    try:
        data = GeometricRefineMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="精细补全 metadata 无效") from exc
    if geometry.content_type not in {"application/octet-stream", None}:
        raise HTTPException(status_code=415, detail="geometry 必须是 float32 二进制")
    try:
        _validate_source_vertex_counts(
            data.task_id,
            [
                SemanticGaussianIndexSet(
                    instance_id=data.instance_id,
                    source_index=data.source_index,
                    source_vertex_count=data.source_vertex_count,
                    indices=[],
                )
            ],
        )
        payload = await geometry.read()
        expected = data.source_vertex_count * 8 * 4
        if len(payload) != expected:
            raise ValueError(f"geometry 字节数应为 {expected}")
        values = np.frombuffer(payload, dtype="<f4").reshape((-1, 8)).copy()
        result = await run_in_threadpool(
            refine_gaussian_selection,
            values,
            data.seed_indices,
            data.candidate_indices,
            data.scene_radius,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GeometricRefineResponse(
        instance_id=data.instance_id,
        source_index=data.source_index,
        source_vertex_count=data.source_vertex_count,
        **result,
    )


def _validate_semantic_upload_counts(image_count: int, depth_count: int) -> None:
    if not 1 <= image_count <= 3:
        raise HTTPException(status_code=400, detail="需要 1 至 3 个视角")
    if depth_count not in {0, image_count}:
        raise HTTPException(status_code=400, detail="depth 数量必须为 0 或与 image 数量一致")


def _validate_source_vertex_counts(
    task_id: str,
    index_sets: list[SemanticGaussianIndexSet],
) -> None:
    if not index_sets:
        return
    task_meta = get_task_meta(task_id) or {}
    output = task_meta.get("output") or {}
    known_vertex_count = output.get("num_gaussians")
    if isinstance(known_vertex_count, bool) or not isinstance(known_vertex_count, int):
        return
    if any(
        index_set.source_index == 0 and index_set.source_vertex_count != known_vertex_count
        for index_set in index_sets
    ):
        raise ValueError("source_vertex_count 与任务 scene.ply 不匹配")


@router.post("/semantic/predict", response_model=SemanticPredictResponse)
async def predict_semantic_view(
    image: list[UploadFile] = File(...),
    metadata: str = Form(...),
    depth: list[UploadFile] | None = File(None),
):
    _validate_semantic_upload_counts(len(image), len(depth or []))
    if any(item.content_type not in {"image/png", "image/jpeg", "image/webp"} for item in image):
        raise HTTPException(status_code=415, detail="截图必须是 PNG、JPEG 或 WebP")
    data = _parse_metadata(metadata)
    try:
        image_bytes = [await item.read() for item in image]
        depth_bytes = [await item.read() for item in depth] if depth else []
        result = semantic_service.predict(
            image_bytes[0],
            data,
            depth_bytes[0] if depth_bytes else None,
            support_image_bytes=image_bytes[1:],
            support_depth_bytes=depth_bytes[1:] if depth_bytes else None,
        )
    except SegmentationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    offsets: dict[str, int] = {}
    for layer in list_layers(result.task_id):
        category = layer.get("category")
        index = layer.get("instance_index")
        if isinstance(category, str) and isinstance(index, int) and not isinstance(index, bool):
            offsets[category] = max(offsets.get(category, 0), index)
    requested_offsets = data.get("instance_index_offsets", {})
    if isinstance(requested_offsets, dict):
        for category, index in requested_offsets.items():
            if (
                isinstance(category, str)
                and isinstance(index, int)
                and not isinstance(index, bool)
                and 0 <= index <= 100000
            ):
                offsets[category] = max(offsets.get(category, 0), index)
    for item in result.instances.values():
        category = item.category or ""
        item.instance_index = (item.instance_index or 1) + offsets.get(category, 0)
    instances = [
        SemanticInstanceResponse(
            instance_id=item.session_id,
            category=item.category or "",
            category_zh=item.category_zh or "",
            instance_index=item.instance_index or 1,
            score=item.score or 0.0,
            bbox=item.bbox or [0, 0, 0, 0],
            mask_url=f"/api/semantic/results/{result.result_id}/instances/{item.session_id}/mask",
            depth_coverage=item.depth_coverage,
            view_support=item.view_support or 1,
            view_count=item.view_count or 1,
            view_masks=[
                SemanticViewMaskResponse(
                    view_index=view_mask.view_index,
                    mask_url=(
                        f"/api/semantic/results/{result.result_id}/instances/"
                        f"{item.session_id}/views/{view_mask.view_index}/mask"
                    ),
                )
                for view_mask in item.view_masks or []
            ],
        )
        for item in result.instances.values()
    ]
    return SemanticPredictResponse(result_id=result.result_id, instances=instances)


@router.get("/semantic/results/{result_id}/instances/{instance_id}/mask")
async def get_semantic_mask(result_id: str, instance_id: str):
    try:
        instance = semantic_service.get(result_id).instances[instance_id]
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="候选 mask 不存在") from exc
    return StreamingResponse(io.BytesIO(instance.mask_png), media_type="image/png")


@router.get("/semantic/results/{result_id}/instances/{instance_id}/views/{view_index}/mask")
async def get_semantic_view_mask(result_id: str, instance_id: str, view_index: int):
    try:
        instance = semantic_service.get(result_id).instances[instance_id]
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=404, detail="候选 mask 不存在") from exc
    view_mask = next(
        (
            candidate
            for candidate in instance.view_masks or []
            if candidate.view_index == view_index
        ),
        None,
    )
    if view_mask is None:
        raise HTTPException(status_code=404, detail="视角 mask 不存在")
    return StreamingResponse(io.BytesIO(view_mask.mask_png), media_type="image/png")


@router.post(
    "/semantic/results/{result_id}/confirm",
    response_model=list[SegmentationLayerResponse],
)
async def confirm_semantic_layers(result_id: str, request: SemanticConfirmRequest):
    try:
        result = semantic_service.get(result_id)
        selected = [result.instances[instance_id] for instance_id in request.instance_ids]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="候选结果不存在") from exc
    if any(item.task_id != result.task_id for item in selected):
        raise HTTPException(status_code=409, detail="候选实例与任务不匹配")

    index_sets_by_instance = {instance_id: [] for instance_id in request.instance_ids}
    for index_set in request.gaussian_index_sets:
        if index_set.instance_id not in result.instances:
            raise HTTPException(status_code=404, detail="Gaussian index set 实例不存在")
        index_sets_by_instance[index_set.instance_id].append(index_set)

    source_ply_fingerprint = hash_task_scene_ply(result.task_id)
    try:
        _validate_source_vertex_counts(result.task_id, request.gaussian_index_sets)
        layers = [
            create_layer(
                item,
                f"{item.category_zh}{item.instance_index}",
                gaussian_index_sets=index_sets_by_instance[item.session_id],
                source_ply_fingerprint=source_ply_fingerprint,
            )
            for item in selected
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    semantic_service.close(result_id)
    return layers


def _parse_metadata(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata 不是有效 JSON") from exc
    required = {
        "task_id",
        "source_ply",
        "viewport_width",
        "viewport_height",
        "capture_width",
        "capture_height",
        "view_matrix",
        "projection_matrix",
    }
    if not required.issubset(data):
        raise HTTPException(status_code=400, detail="metadata 缺少必要字段")
    if get_task_meta(str(data["task_id"])) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    for key in ("view_matrix", "projection_matrix"):
        values = data[key]
        if not isinstance(values, list) or len(values) != 16:
            raise HTTPException(status_code=400, detail=f"{key} 必须包含 16 个数值")
        try:
            data[key] = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"{key} 含非数值") from exc
    for key in ("viewport_width", "viewport_height", "capture_width", "capture_height"):
        value = data[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(status_code=400, detail=f"{key} 必须是整数")
        if value < 1 or value > 8192:
            raise HTTPException(status_code=400, detail=f"{key} 超出范围")
        data[key] = value
    return data


@router.post("/segmentation/sessions", response_model=SegmentationSessionResponse)
async def create_segmentation_session(image: UploadFile = File(...), metadata: str = Form(...)):
    if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="截图必须是 PNG、JPEG 或 WebP")
    data = _parse_metadata(metadata)
    try:
        session = segmentation_service.create(await image.read(), data)
    except SegmentationBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SegmentationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SegmentationSessionResponse(
        session_id=session.session_id,
        width=session.width,
        height=session.height,
        expires_in=SEGMENTATION_SESSION_TTL_SECONDS,
    )


@router.post(
    "/segmentation/sessions/{session_id}/predict",
    response_model=SegmentationPredictResponse,
)
async def predict_segmentation(session_id: str, request: SegmentationPredictRequest):
    try:
        session = segmentation_service.predict(
            session_id, [point.model_dump() for point in request.points]
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="分割会话不存在或已过期") from exc
    return SegmentationPredictResponse(
        session_id=session_id,
        score=session.score or 0.0,
        bbox=session.bbox or [0, 0, 0, 0],
        mask_url=f"/api/segmentation/sessions/{session_id}/mask",
    )


@router.get("/segmentation/sessions/{session_id}/mask")
async def get_session_mask(session_id: str):
    try:
        session = segmentation_service.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="分割会话不存在或已过期") from exc
    if session.mask_png is None:
        raise HTTPException(status_code=404, detail="会话尚无 mask")
    return StreamingResponse(io.BytesIO(session.mask_png), media_type="image/png")


@router.delete("/segmentation/sessions/{session_id}")
async def close_segmentation_session(session_id: str):
    segmentation_service.close(session_id)
    return {"closed": True}


@router.post("/tasks/{task_id}/layers", response_model=SegmentationLayerResponse)
async def confirm_segmentation_layer(task_id: str, request: LayerCreateRequest):
    try:
        session = segmentation_service.get(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="分割会话不存在或已过期") from exc
    if session.task_id != task_id:
        raise HTTPException(status_code=409, detail="会话与任务不匹配")
    try:
        _validate_source_vertex_counts(task_id, request.gaussian_index_sets)
        layer = create_layer(
            session,
            request.name.strip(),
            gaussian_index_sets=request.gaussian_index_sets,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    segmentation_service.close(request.session_id)
    return layer


@router.get("/tasks/{task_id}/layers", response_model=list[SegmentationLayerResponse])
async def get_segmentation_layers(task_id: str):
    return list_layers(task_id)


@router.patch("/tasks/{task_id}/layers/{layer_id}", response_model=SegmentationLayerResponse)
async def update_segmentation_layer(task_id: str, layer_id: str, request: LayerRenameRequest):
    try:
        return rename_layer(task_id, layer_id, request.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="语义图层不存在") from exc


@router.post(
    "/tasks/{task_id}/layers/{layer_id}/observations",
    response_model=LayerMergeResponse,
)
async def add_layer_observation(task_id: str, layer_id: str, request: LayerMergeRequest):
    try:
        result = semantic_service.get(request.result_id)
        instance = result.instances[request.instance_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="识别结果不存在或已过期") from exc
    if result.task_id != task_id or instance.task_id != task_id:
        raise HTTPException(status_code=409, detail="识别结果与任务不匹配")
    try:
        _validate_source_vertex_counts(task_id, request.gaussian_index_sets)
        layer, added_count, total_count = merge_layer_observation(
            task_id, layer_id, instance, request.gaussian_index_sets
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="目标语义图层不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    semantic_service.close(request.result_id)
    return LayerMergeResponse(
        layer=layer,
        added_count=added_count,
        total_count=total_count,
        observation_count=layer["observation_count"],
    )


@router.get("/tasks/{task_id}/layers/{layer_id}/delete-impact")
async def get_layer_delete_impact(task_id: str, layer_id: str):
    if not any(item["layer_id"] == layer_id for item in list_layers(task_id)):
        raise HTTPException(status_code=404, detail="语义图层不存在")
    return {"layer_count": 1, "snapshot_count": len(snapshots_using_layer(task_id, layer_id))}


@router.delete("/tasks/{task_id}/layers/{layer_id}")
async def remove_segmentation_layer(task_id: str, layer_id: str):
    try:
        snapshot_count = delete_snapshots_using_layer(task_id, layer_id)
        delete_layer(task_id, layer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="语义图层不存在") from exc
    return {"deleted": True, "layer_count": 1, "snapshot_count": snapshot_count}


@router.post("/tasks/{task_id}/layers/cleanup")
async def cleanup_segmentation_layers(task_id: str):
    layers = list_layers(task_id)
    snapshot_count = 0
    for layer in layers:
        layer_id = layer["layer_id"]
        snapshot_count += delete_snapshots_using_layer(task_id, layer_id)
        delete_layer(task_id, layer_id)
    return {
        "deleted": True,
        "layer_count": len(layers),
        "snapshot_count": snapshot_count,
    }


@router.get(
    "/tasks/{task_id}/scene-snapshots",
    response_model=list[SceneSnapshotResponse],
)
async def get_scene_snapshots(task_id: str):
    return list_snapshots(task_id)


@router.post(
    "/tasks/{task_id}/scene-snapshots",
    response_model=SceneSnapshotResponse,
)
async def add_scene_snapshot(task_id: str, request: SceneSnapshotCreateRequest):
    known_layers = {item["layer_id"] for item in list_layers(task_id)}
    if any(item.layer_id not in known_layers for item in request.objects):
        raise HTTPException(status_code=409, detail="快照引用了不存在的语义图层")
    return create_snapshot(task_id, request)


@router.patch(
    "/tasks/{task_id}/scene-snapshots/{snapshot_id}",
    response_model=SceneSnapshotResponse,
)
async def update_scene_snapshot(
    task_id: str, snapshot_id: str, request: SceneSnapshotRenameRequest
):
    try:
        return rename_snapshot(task_id, snapshot_id, request.name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="视角快照不存在") from exc


@router.delete("/tasks/{task_id}/scene-snapshots/{snapshot_id}")
async def remove_scene_snapshot(task_id: str, snapshot_id: str):
    try:
        delete_snapshot(task_id, snapshot_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="视角快照不存在") from exc
    return {"deleted": True}


@router.get("/tasks/{task_id}/layers/{layer_id}/mask")
async def get_layer_mask(task_id: str, layer_id: str):
    path = get_mask_path(task_id, layer_id)
    if path is None:
        raise HTTPException(status_code=404, detail="图层 mask 不存在")
    return FileResponse(path, media_type="image/png")


@router.get("/tasks/{task_id}/layers/{layer_id}/gaussian-indices/{source_index}")
async def get_layer_gaussian_indices(task_id: str, layer_id: str, source_index: int):
    path = get_gaussian_indices_path(task_id, layer_id, source_index)
    if path is None:
        raise HTTPException(status_code=404, detail="Gaussian indices 不存在")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.post("/tasks/{task_id}/layers/{layer_id}/mesh")
async def create_layer_mesh(task_id: str, layer_id: str):
    if get_layer_metadata(task_id, layer_id) is None:
        raise HTTPException(status_code=404, detail="语义图层不存在")
    try:
        path, report = await run_in_threadpool(generate_layer_mesh, task_id, layer_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="语义图层不存在") from exc
    except LayerMeshError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    headers = {
        "X-Mesh-Vertices": str(report.get("vertices", 0)),
        "X-Mesh-Triangles": str(report.get("triangles", 0)),
        "X-Mesh-Collision-Triangles": str(report.get("collision_triangles", 0)),
        "X-Mesh-Safe-For-Collision": str(bool(report.get("safe_for_collision", False))).lower(),
        "Access-Control-Expose-Headers": (
            "X-Mesh-Vertices, X-Mesh-Triangles, X-Mesh-Collision-Triangles, "
            "X-Mesh-Safe-For-Collision"
        ),
    }
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{layer_id}-visual-mesh.ply",
        headers=headers,
    )
