"""Realtime detection contract and GPU exclusion tests."""

import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api import realtime_detection
from backend.core import gpu_coordinator
from backend.core.schemas import RealtimeDetectionResponse


def _reset_gpu_coordinator():
    gpu_coordinator.finish_reconstruction()
    gpu_coordinator.finish_segmentation("semantic")
    gpu_coordinator.finish_realtime_detection("realtime")


def test_realtime_detection_response_contract():
    response = RealtimeDetectionResponse(
        frame_id=3,
        width=640,
        height=360,
        inference_ms=12.5,
        detections=[{"category": "bottle", "score": 0.8, "bbox": [0.1, 0.2, 0.3, 0.4]}],
    )
    assert response.detections[0].category == "bottle"


@pytest.mark.parametrize(
    "bbox",
    [([-0.1, 0.2, 0.3, 0.4]), ([0.1, 0.2, 1.1, 0.4]), ([0.8, 0.2, 0.3, 0.4])],
)
def test_realtime_detection_rejects_invalid_normalized_bbox(bbox):
    with pytest.raises(ValidationError):
        RealtimeDetectionResponse(
            frame_id=3,
            width=640,
            height=360,
            inference_ms=12.5,
            detections=[{"category": "bottle", "score": 0.8, "bbox": bbox}],
        )


def test_realtime_detection_excludes_semantic_and_reconstruction():
    _reset_gpu_coordinator()
    try:
        assert gpu_coordinator.try_begin_realtime_detection("realtime")
        assert gpu_coordinator.status() == "realtime_detection"
        assert not gpu_coordinator.try_begin_segmentation("semantic")
        assert not gpu_coordinator.try_begin_reconstruction()
    finally:
        gpu_coordinator.finish_realtime_detection("realtime")
    assert gpu_coordinator.status() == "idle"


def test_semantic_excludes_realtime_detection():
    _reset_gpu_coordinator()
    try:
        assert gpu_coordinator.try_begin_segmentation("semantic")
        assert not gpu_coordinator.try_begin_realtime_detection("realtime")
    finally:
        gpu_coordinator.finish_segmentation("semantic")


def test_semantic_waiter_gets_priority_after_realtime_frame():
    _reset_gpu_coordinator()
    assert gpu_coordinator.try_begin_realtime_detection("realtime")
    acquired: list[bool] = []
    waiter = threading.Thread(
        target=lambda: acquired.append(gpu_coordinator.begin_segmentation("semantic", 1.0))
    )
    waiter.start()
    time.sleep(0.05)
    assert not gpu_coordinator.try_begin_realtime_detection("new-frame")
    gpu_coordinator.finish_realtime_detection("realtime")
    waiter.join(timeout=1.0)
    try:
        assert acquired == [True]
        assert gpu_coordinator.status() == "segmentation"
    finally:
        gpu_coordinator.finish_segmentation("semantic")


def test_realtime_detection_http_endpoint(monkeypatch):
    app = FastAPI()
    app.include_router(realtime_detection.router, prefix="/api")
    monkeypatch.setattr(
        realtime_detection.realtime_detection_service,
        "detect",
        lambda _payload, frame_id: {
            "frame_id": frame_id,
            "width": 640,
            "height": 360,
            "inference_ms": 20.0,
            "detections": [{"category": "bottle", "score": 0.88, "bbox": [0.3, 0.2, 0.5, 0.75]}],
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/realtime/detect",
            data={"frame_id": "7"},
            files={"image": ("camera.jpg", b"fake-jpeg", "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["frame_id"] == 7
    assert response.json()["detections"][0]["category"] == "bottle"


def test_realtime_detection_http_rejects_bad_inputs():
    app = FastAPI()
    app.include_router(realtime_detection.router, prefix="/api")
    with TestClient(app) as client:
        bad_type = client.post(
            "/api/realtime/detect",
            data={"frame_id": "1"},
            files={"image": ("camera.txt", b"text", "text/plain")},
        )
        negative_frame = client.post(
            "/api/realtime/detect",
            data={"frame_id": "-1"},
            files={"image": ("camera.jpg", b"fake-jpeg", "image/jpeg")},
        )
    assert bad_type.status_code == 415
    assert negative_frame.status_code == 422
