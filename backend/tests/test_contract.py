"""契约护栏：确保关键端点已声明 response_model（后端契约化）。

这是一条「回归防护」：防止有人把刚建立的 Pydantic 契约退回成裸 dict 返回。
新增端点时，请在此登记其 response_model 类型名。
"""

from backend.api import ply_viewer, segmentation, upload

EXPECTED = [
    (upload.router, "/upload", "UploadResponse"),
    (upload.router, "/upload-video", "VideoUploadResponse"),
    (ply_viewer.router, "/view-ply", "PlyUploadResponse"),
    (segmentation.router, "/segmentation/sessions", "SegmentationSessionResponse"),
    (segmentation.router, "/semantic/predict", "SemanticPredictResponse"),
    (
        segmentation.router,
        "/semantic/results/{result_id}/confirm",
        "SegmentationLayerResponse",
    ),
    (
        segmentation.router,
        "/segmentation/sessions/{session_id}/predict",
        "SegmentationPredictResponse",
    ),
    (segmentation.router, "/tasks/{task_id}/layers", "SegmentationLayerResponse"),
]


def test_endpoints_have_response_model():
    for router, path, model_name in EXPECTED:
        route = next(r for r in router.routes if r.path == path)
        assert route.response_model is not None, f"{path} 缺少 response_model"
        assert model_name in str(route.response_model), (
            f"{path} 的 response_model 类型不符，期望 {model_name}"
        )
