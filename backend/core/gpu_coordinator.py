"""Single-GPU exclusion between ZipSplat reconstruction and SAM segmentation."""

import threading
import time

_lock = threading.Lock()
_condition = threading.Condition(_lock)
_reconstruction_active = False
_segmentation_session_id: str | None = None
_realtime_detection_id: str | None = None
_segmentation_waiters = 0


def try_begin_reconstruction() -> bool:
    global _reconstruction_active
    with _lock:
        if (
            _reconstruction_active
            or _segmentation_session_id is not None
            or _realtime_detection_id
            or _segmentation_waiters
        ):
            return False
        _reconstruction_active = True
        return True


def finish_reconstruction() -> None:
    global _reconstruction_active
    with _lock:
        _reconstruction_active = False
        _condition.notify_all()


def try_begin_segmentation(session_id: str) -> bool:
    global _segmentation_session_id
    with _lock:
        if _reconstruction_active or _segmentation_session_id is not None or _realtime_detection_id:
            return False
        _segmentation_session_id = session_id
        return True


def begin_segmentation(session_id: str, timeout_seconds: float = 3.0) -> bool:
    """Give semantic work priority while waiting for an in-flight realtime frame."""
    global _segmentation_session_id, _segmentation_waiters
    deadline = time.monotonic() + timeout_seconds
    with _condition:
        _segmentation_waiters += 1
        try:
            while _realtime_detection_id is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                _condition.wait(remaining)
            if _reconstruction_active or _segmentation_session_id is not None:
                return False
            _segmentation_session_id = session_id
            return True
        finally:
            _segmentation_waiters -= 1


def finish_segmentation(session_id: str) -> None:
    global _segmentation_session_id
    with _lock:
        if _segmentation_session_id == session_id:
            _segmentation_session_id = None
            _condition.notify_all()


def try_begin_realtime_detection(request_id: str) -> bool:
    global _realtime_detection_id
    with _lock:
        if (
            _reconstruction_active
            or _segmentation_session_id is not None
            or _realtime_detection_id
            or _segmentation_waiters
        ):
            return False
        _realtime_detection_id = request_id
        return True


def finish_realtime_detection(request_id: str) -> None:
    global _realtime_detection_id
    with _lock:
        if _realtime_detection_id == request_id:
            _realtime_detection_id = None
            _condition.notify_all()


def status() -> str:
    with _lock:
        if _reconstruction_active:
            return "reconstruction"
        if _segmentation_session_id is not None:
            return "segmentation"
        if _realtime_detection_id is not None:
            return "realtime_detection"
        return "idle"
