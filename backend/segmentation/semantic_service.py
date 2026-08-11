"""Automatic fixed-vocabulary instance segmentation for the current camera view."""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.core.config import LAYER_DIR, MAX_SEGMENTATION_IMAGE_BYTES, MAX_SEGMENTATION_IMAGE_SIDE
from backend.core.gpu_coordinator import finish_segmentation, try_begin_segmentation
from backend.segmentation.service import (
    SegmentationBusyError,
    SegmentationSession,
    SegmentationViewMask,
    _encode_mask,
)

TARGETS = {
    "cup": ("cup", "杯子"),
    "chair": ("chair", "椅子"),
    "bottle": ("bottle", "瓶子"),
}
SCORE_THRESHOLD = 0.40
MULTIVIEW_OVERLAP_THRESHOLD = 0.15


@dataclass(frozen=True)
class _TargetInstance:
    category: str
    category_zh: str
    score: float
    mask: np.ndarray
    bbox: list[int]


def _target_instances(prediction: dict, categories: list[str]) -> list[_TargetInstance]:
    result: list[_TargetInstance] = []
    for label, score, box, mask in zip(
        prediction["labels"],
        prediction["scores"],
        prediction["boxes"],
        prediction["masks"],
        strict=True,
    ):
        source = categories[int(label)]
        if float(score) < SCORE_THRESHOLD or source not in TARGETS:
            continue
        category, category_zh = TARGETS[source]
        result.append(
            _TargetInstance(
                category=category,
                category_zh=category_zh,
                score=float(score),
                mask=mask[0].detach().cpu().numpy() >= 0.5,
                bbox=[int(value) for value in box.detach().cpu().tolist()],
            )
        )
    return result


def _mask_overlap(center_mask: np.ndarray, target_mask: np.ndarray) -> float:
    if center_mask.shape != target_mask.shape:
        raise ValueError("semantic masks have inconsistent dimensions")
    center_count = int(np.count_nonzero(center_mask))
    if center_count == 0:
        return 0.0
    return float(np.count_nonzero(center_mask & target_mask) / center_count)


def _multiview_overlap(
    center_mask: np.ndarray,
    center_depth: np.ndarray,
    center_view: dict,
    target_mask: np.ndarray,
    target_view: dict,
) -> float:
    valid = center_mask & np.isfinite(center_depth)
    ys, xs = np.nonzero(valid)
    if not len(xs):
        return 0.0
    if len(xs) > 4096:
        indices = np.linspace(0, len(xs) - 1, 4096, dtype=np.int64)
        xs, ys = xs[indices], ys[indices]
    height, width = center_mask.shape
    near = float(center_view["near"])
    far = float(center_view["far"])
    linear_depth = near + center_depth[ys, xs] * (far - near)
    projection = np.asarray(center_view["projection_matrix"], dtype=np.float64).reshape(
        4, 4, order="F"
    )
    nx = 2 * (xs + 0.5) / width - 1
    ny = 1 - 2 * (ys + 0.5) / height
    if center_view.get("projection") == "orthographic":
        camera_x = (nx - projection[0, 3]) / projection[0, 0]
        camera_y = (ny - projection[1, 3]) / projection[1, 1]
    else:
        camera_x = nx * linear_depth / projection[0, 0]
        camera_y = ny * linear_depth / projection[1, 1]
    camera_points = np.stack([camera_x, camera_y, -linear_depth, np.ones_like(linear_depth)])
    view = np.asarray(center_view["view_matrix"], dtype=np.float64).reshape(4, 4, order="F")
    world_points = np.linalg.inv(view) @ camera_points
    target_projection = np.asarray(target_view["projection_matrix"], dtype=np.float64).reshape(
        4, 4, order="F"
    )
    target_view_matrix = np.asarray(target_view["view_matrix"], dtype=np.float64).reshape(
        4, 4, order="F"
    )
    clip = target_projection @ target_view_matrix @ world_points
    in_front = clip[3] > 1e-8
    ndc = clip[:3] / np.where(in_front, clip[3], 1)
    target_height, target_width = target_mask.shape
    tx = np.floor((ndc[0] + 1) * 0.5 * target_width).astype(np.int64)
    ty = np.floor((1 - ndc[1]) * 0.5 * target_height).astype(np.int64)
    inside = in_front & (tx >= 0) & (tx < target_width) & (ty >= 0) & (ty < target_height)
    hits = np.zeros(len(tx), dtype=bool)
    hits[inside] = target_mask[ty[inside], tx[inside]]
    return float(hits.mean())


def _match_instances(
    center_instances: list[_TargetInstance],
    target_instances: list[_TargetInstance],
    center_depth: np.ndarray | None = None,
    center_view: dict | None = None,
    target_view: dict | None = None,
) -> dict[int, tuple[int, float]]:
    """Greedily match highest-overlap pairs while using each instance at most once."""
    candidates: list[tuple[float, int, int]] = []
    for center_index, center in enumerate(center_instances):
        for target_index, target in enumerate(target_instances):
            if center.category != target.category:
                continue
            if center_depth is None:
                overlap = _mask_overlap(center.mask, target.mask)
            else:
                if center_view is None or target_view is None:
                    raise ValueError("camera metadata is required for depth-based matching")
                overlap = _multiview_overlap(
                    center.mask,
                    center_depth,
                    center_view,
                    target.mask,
                    target_view,
                )
            if overlap >= MULTIVIEW_OVERLAP_THRESHOLD:
                candidates.append((overlap, center_index, target_index))

    matches: dict[int, tuple[int, float]] = {}
    matched_targets: set[int] = set()
    for overlap, center_index, target_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if center_index in matches or target_index in matched_targets:
            continue
        matches[center_index] = (target_index, overlap)
        matched_targets.add(target_index)
    return matches


def _decode_semantic_image(image_bytes: bytes, view_index: int) -> Image.Image:
    if len(image_bytes) > MAX_SEGMENTATION_IMAGE_BYTES:
        raise ValueError(f"view {view_index} exceeds the image size limit")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"view {view_index} is not a valid image") from exc
    image = image.convert("RGB")
    if max(image.size) > MAX_SEGMENTATION_IMAGE_SIDE:
        raise ValueError(f"view {view_index} exceeds the image dimension limit")
    return image


def _decode_depth_map(
    depth_bytes: bytes,
    width: int,
    height: int,
    view_index: int,
) -> np.ndarray:
    expected_size = width * height * np.dtype("<f4").itemsize
    if len(depth_bytes) != expected_size:
        raise ValueError(
            f"depth {view_index} has {len(depth_bytes)} bytes; expected {expected_size}"
        )
    values = np.frombuffer(depth_bytes, dtype="<f4")
    if np.isinf(values).any():
        raise ValueError(f"depth {view_index} contains infinity")
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError(f"depth {view_index} contains values outside [0, 1]")
    return values.reshape(height, width)


def _validate_capture_dimensions(image: Image.Image, metadata: dict) -> None:
    capture_width = metadata.get("capture_width")
    capture_height = metadata.get("capture_height")
    if capture_width != image.width or capture_height != image.height:
        raise ValueError(
            "capture dimensions do not match the decoded image: "
            f"metadata={capture_width}x{capture_height}, image={image.width}x{image.height}"
        )


def _view_metadata(metadata: dict, view_count: int) -> list[dict]:
    views = metadata.get("views")
    if views is None:
        if view_count != 1:
            raise ValueError("metadata.views must contain one entry per image")
        return [metadata]
    if not isinstance(views, list) or len(views) != view_count:
        raise ValueError("metadata.views must contain one entry per image")
    return views


@dataclass
class SemanticResult:
    result_id: str
    task_id: str
    created_at: float
    instances: dict[str, SegmentationSession]


class SemanticSegmentationService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._results: dict[str, SemanticResult] = {}

    def predict(
        self,
        image_bytes: bytes,
        metadata: dict,
        depth_bytes: bytes | None = None,
        support_image_bytes: list[bytes] | None = None,
        support_depth_bytes: list[bytes] | None = None,
    ) -> SemanticResult:
        with self._lock:
            result_id = uuid.uuid4().hex
            if not try_begin_segmentation(result_id):
                raise SegmentationBusyError("GPU 正在执行其他任务")
            try:
                support_payloads = support_image_bytes or []
                images = [
                    _decode_semantic_image(payload, view_index)
                    for view_index, payload in enumerate([image_bytes, *support_payloads])
                ]
                image = images[0]
                _validate_capture_dimensions(image, metadata)
                if any(support_image.size != image.size for support_image in images[1:]):
                    raise ValueError("multi-view images have inconsistent dimensions")
                views = _view_metadata(metadata, len(images))

                support_depth_payloads = support_depth_bytes or []
                if depth_bytes is None:
                    if support_depth_payloads:
                        raise ValueError("support depth maps require a center depth map")
                    depth_maps: list[np.ndarray | None] = [None] * len(images)
                else:
                    if len(support_depth_payloads) != len(support_payloads):
                        raise ValueError("depth count must match image count")
                    depth_maps = [
                        _decode_depth_map(payload, image.width, image.height, view_index)
                        for view_index, payload in enumerate([depth_bytes, *support_depth_payloads])
                    ]
                center_depth = depth_maps[0]

                task_id = str(metadata["task_id"])
                if task_id.isalnum() and len(task_id) <= 64:
                    preview_dir = LAYER_DIR / task_id
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    image.save(preview_dir / "latest_semantic_view.png", format="PNG")

                from torchvision.models.detection import (
                    MaskRCNN_ResNet50_FPN_V2_Weights,
                    maskrcnn_resnet50_fpn_v2,
                )

                weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                categories = weights.meta["categories"]
                model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().cuda()
                tensor = weights.transforms()(image).cuda()
                with torch.inference_mode():
                    output = model([tensor])[0]
                center_instances = _target_instances(output, categories)
                del tensor, output

                support_instances_by_view: list[list[_TargetInstance]] = []
                for support_image in images[1:]:
                    support_tensor = weights.transforms()(support_image).cuda()
                    with torch.inference_mode():
                        support_output = model([support_tensor])[0]
                    support_instances_by_view.append(_target_instances(support_output, categories))
                    del support_tensor, support_output

                matched_by_center: dict[int, list[tuple[int, _TargetInstance]]] = {
                    index: [] for index in range(len(center_instances))
                }
                for view_index, target_instances in enumerate(support_instances_by_view, start=1):
                    matches = _match_instances(
                        center_instances,
                        target_instances,
                        center_depth=center_depth,
                        center_view=views[0],
                        target_view=views[view_index],
                    )
                    for center_index, (target_index, _) in matches.items():
                        matched_by_center[center_index].append(
                            (view_index, target_instances[target_index])
                        )

                instances: dict[str, SegmentationSession] = {}
                counters: dict[str, int] = {}
                for center_index, detected in enumerate(center_instances):
                    matched_views = matched_by_center[center_index]
                    view_support = 1 + len(matched_views)
                    if len(images) > 1 and view_support < 2:
                        continue

                    depth_coverage = None
                    if center_depth is not None and detected.mask.any():
                        depth_coverage = float(np.isfinite(center_depth)[detected.mask].mean())
                        if depth_coverage < 0.15:
                            continue

                    counters[detected.category] = counters.get(detected.category, 0) + 1
                    instance_id = uuid.uuid4().hex[:12]
                    png, rle, _ = _encode_mask(detected.mask)
                    view_masks = [SegmentationViewMask(view_index=0, mask_png=png)]
                    for view_index, matched in matched_views:
                        matched_png, _, _ = _encode_mask(matched.mask)
                        view_masks.append(
                            SegmentationViewMask(
                                view_index=view_index,
                                mask_png=matched_png,
                            )
                        )
                    now = time.monotonic()
                    session = SegmentationSession(
                        session_id=instance_id,
                        task_id=task_id,
                        source_ply=str(metadata["source_ply"]),
                        width=image.width,
                        height=image.height,
                        viewport_width=metadata["viewport_width"],
                        viewport_height=metadata["viewport_height"],
                        view_matrix=metadata["view_matrix"],
                        projection_matrix=metadata["projection_matrix"],
                        created_at=now,
                        touched_at=now,
                        points=[],
                        mask_png=png,
                        mask_rle=rle,
                        score=detected.score,
                        bbox=detected.bbox,
                        category=detected.category,
                        category_zh=detected.category_zh,
                        instance_index=counters[detected.category],
                        depth_coverage=depth_coverage,
                        view_support=view_support,
                        view_count=len(images),
                        near=metadata.get("near"),
                        far=metadata.get("far"),
                        projection=metadata.get("projection"),
                        auxiliary_views=views,
                        depth_f32=depth_bytes,
                        view_masks=view_masks,
                    )
                    instances[instance_id] = session
                result = SemanticResult(result_id, task_id, time.monotonic(), instances)
                self._results[result_id] = result
                return result
            finally:
                if "model" in locals():
                    del model
                torch.cuda.empty_cache()
                finish_segmentation(result_id)

    def get(self, result_id: str) -> SemanticResult:
        result = self._results.get(result_id)
        if result is None:
            raise KeyError(result_id)
        return result

    def close(self, result_id: str) -> None:
        self._results.pop(result_id, None)


semantic_service = SemanticSegmentationService()
