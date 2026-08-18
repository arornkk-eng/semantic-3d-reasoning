"""Lazy YOLO detector used only for moving-camera previews."""

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from backend.core.gpu_coordinator import finish_realtime_detection, try_begin_realtime_detection

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "yolo26n-objv1-150.pt"
MODEL_NAME = os.getenv("REALTIME_YOLO_MODEL", str(DEFAULT_MODEL_PATH))
CONFIDENCE = float(os.getenv("REALTIME_YOLO_CONFIDENCE", "0.35"))
IMAGE_SIZE = int(os.getenv("REALTIME_YOLO_IMAGE_SIZE", "640"))
MAX_DETECTIONS = int(os.getenv("REALTIME_YOLO_MAX_DETECTIONS", "15"))


class RealtimeDetectionUnavailableError(RuntimeError):
    pass


class RealtimeDetectionBusyError(RuntimeError):
    pass


class RealtimeDetectionService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RealtimeDetectionUnavailableError(
                    "实时检测未安装 ultralytics，请执行 pip install ultralytics"
                ) from exc
            self._model = YOLO(MODEL_NAME)
        return self._model

    def warmup(self) -> None:
        """Compile kernels before the first moving-camera frame."""
        try:
            self.detect(np.zeros((360, 640, 3), dtype=np.uint8), -1)
        except (RealtimeDetectionBusyError, RealtimeDetectionUnavailableError):
            return

    def detect(self, image_bytes: bytes | np.ndarray, frame_id: int) -> dict:
        token = f"realtime-{frame_id}"
        if not try_begin_realtime_detection(token):
            raise RealtimeDetectionBusyError("GPU 正在执行重建或精细分割")
        started = time.perf_counter()
        try:
            with self._lock:
                image = (
                    Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    if isinstance(image_bytes, bytes)
                    else Image.fromarray(image_bytes).convert("RGB")
                )
                width, height = image.size
                result = self._get_model().predict(
                    image,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    max_det=MAX_DETECTIONS,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                )[0]
                names = result.names
                detections = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = [float(value) for value in box.xyxyn[0].tolist()]
                        category = str(names[int(box.cls[0])])
                        detections.append(
                            {
                                "category": category,
                                "score": float(box.conf[0]),
                                "bbox": [x1, y1, x2, y2],
                            }
                        )
                return {
                    "frame_id": frame_id,
                    "width": width,
                    "height": height,
                    "inference_ms": round((time.perf_counter() - started) * 1000, 1),
                    "detections": detections,
                }
        finally:
            finish_realtime_detection(token)


realtime_detection_service = RealtimeDetectionService()
