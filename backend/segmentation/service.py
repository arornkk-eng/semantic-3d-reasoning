"""Lazy SAM 2 image predictor with one GPU-resident interactive session."""

from __future__ import annotations

import io
import threading
import time
import uuid
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from backend.core.config import (
    MAX_SEGMENTATION_IMAGE_BYTES,
    MAX_SEGMENTATION_IMAGE_SIDE,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CONFIG,
    SEGMENTATION_SESSION_TTL_SECONDS,
)
from backend.core.gpu_coordinator import finish_segmentation, try_begin_segmentation


class SegmentationBusyError(RuntimeError):
    pass


class SegmentationUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SegmentationViewMask:
    view_index: int
    mask_png: bytes


@dataclass
class SegmentationSession:
    session_id: str
    task_id: str
    source_ply: str
    width: int
    height: int
    viewport_width: int
    viewport_height: int
    view_matrix: list[float]
    projection_matrix: list[float]
    created_at: float
    touched_at: float
    points: list[dict] | None = None
    mask_png: bytes | None = None
    mask_rle: dict | None = None
    score: float | None = None
    bbox: list[int] | None = None
    category: str | None = None
    category_zh: str | None = None
    instance_index: int | None = None
    depth_coverage: float | None = None
    view_support: int | None = None
    view_count: int | None = None
    near: float | None = None
    far: float | None = None
    projection: str | None = None
    auxiliary_views: list[dict] | None = None
    depth_f32: bytes | None = None
    view_masks: list[SegmentationViewMask] | None = None


def _encode_mask(mask: np.ndarray) -> tuple[bytes, dict, list[int]]:
    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(binary)
    bbox = (
        [0, 0, 0, 0]
        if len(xs) == 0
        else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    )
    image = Image.fromarray(binary.astype(np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    flat = binary.reshape(-1).astype(np.uint8)
    counts: list[int] = []
    value = 0
    run = 0
    for item in flat:
        item_value = int(item)
        if item_value == value:
            run += 1
        else:
            counts.append(run)
            run = 1
            value = item_value
    counts.append(run)
    return (
        buffer.getvalue(),
        {"size": list(binary.shape), "counts": counts, "order": "row-major"},
        bbox,
    )


class SegmentationService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._predictor = None
        self._session: SegmentationSession | None = None

    def _expire(self) -> None:
        if (
            self._session
            and time.monotonic() - self._session.touched_at > SEGMENTATION_SESSION_TTL_SECONDS
        ):
            self.close(self._session.session_id)

    def _load_predictor(self):
        if self._predictor is not None:
            return self._predictor
        if not torch.cuda.is_available():
            raise SegmentationUnavailableError("SAM 2 需要 CUDA GPU")
        if not SAM2_CHECKPOINT.is_file():
            raise SegmentationUnavailableError(f"SAM 2 权重不存在: {SAM2_CHECKPOINT}")
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise SegmentationUnavailableError("SAM 2 尚未安装") from exc
        self._predictor = SAM2ImagePredictor(
            build_sam2(SAM2_MODEL_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
        )
        return self._predictor

    def create(self, image_bytes: bytes, metadata: dict) -> SegmentationSession:
        with self._lock:
            self._expire()
            if len(image_bytes) > MAX_SEGMENTATION_IMAGE_BYTES:
                raise ValueError("截图超过 20 MB")
            session_id = uuid.uuid4().hex
            if not try_begin_segmentation(session_id):
                raise SegmentationBusyError("GPU 正在执行其他任务")
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                if max(image.size) > MAX_SEGMENTATION_IMAGE_SIDE:
                    raise ValueError("截图边长超过 4096")
                predictor = self._load_predictor()
                predictor.set_image(np.array(image, copy=True))
                now = time.monotonic()
                self._session = SegmentationSession(
                    session_id=session_id,
                    task_id=metadata["task_id"],
                    source_ply=metadata["source_ply"],
                    width=image.width,
                    height=image.height,
                    viewport_width=metadata["viewport_width"],
                    viewport_height=metadata["viewport_height"],
                    view_matrix=metadata["view_matrix"],
                    projection_matrix=metadata["projection_matrix"],
                    created_at=now,
                    touched_at=now,
                )
                return self._session
            except Exception:
                finish_segmentation(session_id)
                raise

    def predict(self, session_id: str, points: list[dict]) -> SegmentationSession:
        with self._lock:
            session = self.get(session_id)
            coords = np.asarray(
                [[point["x"] * session.width, point["y"] * session.height] for point in points],
                dtype=np.float32,
            )
            labels = np.asarray([point["label"] for point in points], dtype=np.int32)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks, scores, _ = self._predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=False,
                )
            png, rle, bbox = _encode_mask(masks[0])
            session.points = points
            session.mask_png = png
            session.mask_rle = rle
            session.score = float(scores[0])
            session.bbox = bbox
            session.touched_at = time.monotonic()
            return session

    def get(self, session_id: str) -> SegmentationSession:
        self._expire()
        if self._session is None or self._session.session_id != session_id:
            raise KeyError(session_id)
        self._session.touched_at = time.monotonic()
        return self._session

    def close(self, session_id: str) -> None:
        with self._lock:
            if self._session is None or self._session.session_id != session_id:
                return
            self._session = None
            self._predictor = None
            finish_segmentation(session_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


segmentation_service = SegmentationService()
