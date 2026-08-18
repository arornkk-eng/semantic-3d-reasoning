"""Moving-camera YOLO preview endpoint."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.core.schemas import RealtimeDetectionResponse
from backend.detection.realtime_service import (
    RealtimeDetectionBusyError,
    RealtimeDetectionUnavailableError,
    realtime_detection_service,
)

router = APIRouter()


@router.post("/realtime/detect", response_model=RealtimeDetectionResponse)
async def detect_realtime(image: UploadFile = File(...), frame_id: int = Form(..., ge=0)):
    if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="检测截图格式不支持")
    payload = await image.read()
    if len(payload) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="检测截图过大")
    try:
        return await run_in_threadpool(realtime_detection_service.detect, payload, frame_id)
    except RealtimeDetectionBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RealtimeDetectionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="检测截图无效") from exc
