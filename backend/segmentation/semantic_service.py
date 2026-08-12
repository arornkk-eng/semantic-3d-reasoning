"""Automatic fixed-vocabulary instance segmentation for the current camera view."""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from backend.core.config import (
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_MODEL,
    GROUNDING_DINO_TEXT_THRESHOLD,
    LAYER_DIR,
    MAX_SEGMENTATION_IMAGE_BYTES,
    MAX_SEGMENTATION_IMAGE_SIDE,
    SEMANTIC_DETECTOR,
)
from backend.core.gpu_coordinator import finish_segmentation, try_begin_segmentation
from backend.segmentation.service import (
    SegmentationBusyError,
    SegmentationSession,
    SegmentationUnavailableError,
    SegmentationViewMask,
    _encode_mask,
    segmentation_service,
)

logger = logging.getLogger(__name__)

TARGETS = {
    "cup": ("cup", "杯子"),
    "chair": ("chair", "椅子"),
    "bottle": ("bottle", "瓶子"),
    "bed": ("bed", "床"),
    "couch": ("couch", "沙发"),
    "tv": ("tv", "电视"),
    "laptop": ("laptop", "笔记本电脑"),
    "keyboard": ("keyboard", "键盘"),
    "mouse": ("mouse", "鼠标"),
    "cell phone": ("cell_phone", "手机"),
    "book": ("book", "书"),
    "potted plant": ("potted_plant", "盆栽"),
    "vase": ("vase", "花瓶"),
    "clock": ("clock", "时钟"),
    "refrigerator": ("refrigerator", "冰箱"),
    "microwave": ("microwave", "微波炉"),
    "oven": ("oven", "烤箱"),
    "sink": ("sink", "水槽"),
    "toilet": ("toilet", "马桶"),
    "dining table": ("dining_table", "餐桌"),
    "table": ("table", "桌子"),
    "desk": ("desk", "书桌"),
    "coffee table": ("coffee_table", "茶几"),
    "cabinet": ("cabinet", "柜子"),
    "wardrobe": ("wardrobe", "衣柜"),
    "nightstand": ("nightstand", "床头柜"),
    "table lamp": ("table_lamp", "台灯"),
    "computer monitor": ("computer_monitor", "显示器"),
    "trash can": ("trash_can", "垃圾桶"),
    "door": ("door", "门"),
    "window": ("window", "窗"),
    "bookshelf": ("bookshelf", "书架"),
}
OPEN_VOCAB_PROMPTS = {value[0].replace("_", " "): value for value in TARGETS.values()}
SCORE_THRESHOLD = 0.40
MULTIVIEW_OVERLAP_THRESHOLD = 0.15
SAM_REFINEMENT_MIN_IOU = 0.25
SAM_REFINEMENT_MIN_AREA_RATIO = 0.30
SAM_REFINEMENT_MAX_AREA_RATIO = 3.0


@dataclass(frozen=True)
class _TargetInstance:
    category: str
    category_zh: str
    score: float
    mask: np.ndarray
    bbox: list[int]


@dataclass(frozen=True)
class _GroundedDetection:
    category: str
    category_zh: str
    score: float
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


def _normalize_grounded_label(label: str) -> tuple[str, str] | None:
    normalized = label.strip().lower().replace("_", " ").rstrip(".")
    if normalized in OPEN_VOCAB_PROMPTS:
        return OPEN_VOCAB_PROMPTS[normalized]
    # Grounding DINO may return a short phrase containing the requested noun.
    matches = [
        (prompt, value) for prompt, value in OPEN_VOCAB_PROMPTS.items() if prompt in normalized
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


def _grounded_detections(result: dict, width: int, height: int) -> list[_GroundedDetection]:
    detections: list[_GroundedDetection] = []
    for score, box, label in zip(
        result["scores"], result["boxes"], result["text_labels"], strict=True
    ):
        target = _normalize_grounded_label(str(label))
        if target is None:
            continue
        x1, y1, x2, y2 = [round(float(value)) for value in box]
        x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
        y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(_GroundedDetection(*target, float(score), [x1, y1, x2, y2]))
    kept: list[_GroundedDetection] = []
    for detection in sorted(detections, key=lambda item: -item.score):
        if any(
            detection.category == existing.category
            and _box_iou(detection.bbox, existing.bbox) >= 0.70
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept


def _box_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_a = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1)
    area_b = max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _detect_with_grounding_dino(images: list[Image.Image]) -> list[list[_GroundedDetection]]:
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    local_only = SEMANTIC_DETECTOR != "grounding_dino"
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL, local_files_only=local_only)
    model = GroundingDinoForObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL, local_files_only=local_only
    ).eval().cuda()
    labels = list(OPEN_VOCAB_PROMPTS)
    prompts = [labels for _ in images]
    inputs = processor(images=images, text=prompts, return_tensors="pt").to("cuda")
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=GROUNDING_DINO_BOX_THRESHOLD,
            text_threshold=GROUNDING_DINO_TEXT_THRESHOLD,
            target_sizes=[image.size[::-1] for image in images],
            text_labels=prompts,
        )
        return [
            _grounded_detections(result, image.width, image.height)
            for result, image in zip(results, images, strict=True)
        ]
    finally:
        del inputs, model
        torch.cuda.empty_cache()


def _accept_box_mask(mask: np.ndarray, bbox: list[int]) -> bool:
    if not mask.any():
        return False
    x1, y1, x2, y2 = bbox
    box_area = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
    mask_area = int(np.count_nonzero(mask))
    inside = int(np.count_nonzero(mask[y1 : y2 + 1, x1 : x2 + 1]))
    return inside / mask_area >= 0.80 and 0.02 <= mask_area / box_area <= 1.5


def _segment_grounded_detections(
    image: Image.Image,
    detections: list[_GroundedDetection],
    predictor,
) -> list[_TargetInstance]:
    if not detections:
        return []
    predictor.set_image(np.array(image, copy=True))
    instances: list[_TargetInstance] = []
    for detection in detections:
        box = np.asarray(detection.bbox, dtype=np.float32)
        try:
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
            mask = np.asarray(masks[0], dtype=bool)
            if not _accept_box_mask(mask, detection.bbox):
                continue
            ys, xs = np.nonzero(mask)
            instances.append(
                _TargetInstance(
                    detection.category,
                    detection.category_zh,
                    detection.score,
                    mask,
                    [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                )
            )
        except (RuntimeError, ValueError, IndexError) as exc:
            logger.warning("SAM 2 box segmentation failed for %s: %s", detection.category, exc)
    return instances


def _interior_prompt(mask: np.ndarray) -> np.ndarray:
    """Return a stable positive prompt near the center of the coarse mask."""
    from scipy.ndimage import distance_transform_edt

    if not mask.any():
        raise ValueError("cannot create a prompt from an empty mask")
    y, x = np.unravel_index(np.argmax(distance_transform_edt(mask)), mask.shape)
    return np.asarray([[float(x), float(y)]], dtype=np.float32)


def _accept_refined_mask(coarse: np.ndarray, refined: np.ndarray) -> bool:
    if refined.shape != coarse.shape or not refined.any():
        return False
    coarse_area = int(np.count_nonzero(coarse))
    refined_area = int(np.count_nonzero(refined))
    if coarse_area == 0:
        return False
    area_ratio = refined_area / coarse_area
    if not SAM_REFINEMENT_MIN_AREA_RATIO <= area_ratio <= SAM_REFINEMENT_MAX_AREA_RATIO:
        return False
    union = int(np.count_nonzero(coarse | refined))
    intersection = int(np.count_nonzero(coarse & refined))
    return union > 0 and intersection / union >= SAM_REFINEMENT_MIN_IOU


def _refine_instances_with_sam(
    image: Image.Image,
    instances: list[_TargetInstance],
    predictor,
) -> list[_TargetInstance]:
    """Refine Mask R-CNN instances with SAM 2, preserving unsafe coarse masks."""
    if not instances:
        return instances
    predictor.set_image(np.array(image, copy=True))
    refined_instances: list[_TargetInstance] = []
    for instance in instances:
        try:
            points = _interior_prompt(instance.mask)
            labels = np.ones(len(points), dtype=np.int32)
            box = np.asarray(instance.bbox, dtype=np.float32)
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                masks, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=box,
                    multimask_output=False,
                )
            refined = np.asarray(masks[0], dtype=bool)
            if _accept_refined_mask(instance.mask, refined):
                ys, xs = np.nonzero(refined)
                refined_instances.append(
                    _TargetInstance(
                        category=instance.category,
                        category_zh=instance.category_zh,
                        score=instance.score,
                        mask=refined,
                        bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    )
                )
                continue
        except (RuntimeError, ValueError, IndexError) as exc:
            logger.warning("SAM 2 refinement failed for %s: %s", instance.category, exc)
        refined_instances.append(instance)
    return refined_instances


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

                model = None
                sam_predictor = None
                all_instances: list[list[_TargetInstance]] | None = None
                if SEMANTIC_DETECTOR in {"auto", "grounding_dino"}:
                    try:
                        grounded = _detect_with_grounding_dino(images)
                        sam_predictor = segmentation_service._load_predictor()
                        all_instances = [
                            _segment_grounded_detections(view_image, detections, sam_predictor)
                            for view_image, detections in zip(images, grounded, strict=True)
                        ]
                        logger.info("semantic detector: Grounding DINO + SAM 2")
                    except (OSError, ImportError, RuntimeError, ValueError) as exc:
                        logger.warning("Grounding DINO unavailable; using Mask R-CNN: %s", exc)

                if all_instances is None:
                    from torchvision.models.detection import (
                        MaskRCNN_ResNet50_FPN_V2_Weights,
                        maskrcnn_resnet50_fpn_v2,
                    )

                    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                    categories = weights.meta["categories"]
                    model = maskrcnn_resnet50_fpn_v2(weights=weights).eval().cuda()
                    all_instances = []
                    for view_image in images:
                        tensor = weights.transforms()(view_image).cuda()
                        with torch.inference_mode():
                            output = model([tensor])[0]
                        all_instances.append(_target_instances(output, categories))
                        del tensor, output
                    del model
                    model = None
                    torch.cuda.empty_cache()
                    try:
                        sam_predictor = segmentation_service._load_predictor()
                    except (SegmentationUnavailableError, RuntimeError) as exc:
                        logger.warning("SAM 2 automatic refinement unavailable: %s", exc)
                    if sam_predictor is not None:
                        all_instances = [
                            _refine_instances_with_sam(view_image, instances, sam_predictor)
                            for view_image, instances in zip(images, all_instances, strict=True)
                        ]

                center_instances = all_instances[0]
                support_instances_by_view = all_instances[1:]
                if sam_predictor is None and SEMANTIC_DETECTOR == "grounding_dino":
                    raise SegmentationUnavailableError("Grounding DINO requires SAM 2")

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
                if "model" in locals() and model is not None:
                    del model
                segmentation_service.release_predictor()
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
