"""Single-GPU exclusion between ZipSplat reconstruction and SAM segmentation."""

import threading

_lock = threading.Lock()
_reconstruction_active = False
_segmentation_session_id: str | None = None


def try_begin_reconstruction() -> bool:
    global _reconstruction_active
    with _lock:
        if _reconstruction_active or _segmentation_session_id is not None:
            return False
        _reconstruction_active = True
        return True


def finish_reconstruction() -> None:
    global _reconstruction_active
    with _lock:
        _reconstruction_active = False


def try_begin_segmentation(session_id: str) -> bool:
    global _segmentation_session_id
    with _lock:
        if _reconstruction_active or _segmentation_session_id is not None:
            return False
        _segmentation_session_id = session_id
        return True


def finish_segmentation(session_id: str) -> None:
    global _segmentation_session_id
    with _lock:
        if _segmentation_session_id == session_id:
            _segmentation_session_id = None


def status() -> str:
    with _lock:
        if _reconstruction_active:
            return "reconstruction"
        if _segmentation_session_id is not None:
            return "segmentation"
        return "idle"
