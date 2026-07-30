"""开集物体检测：使用 Grounding DINO 在输入帧中检测指定物体。

模型首次加载会从 HuggingFace Hub 下载 (~700MB)，后续使用缓存。
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 延迟加载的全局模型
_pipeline = None
_device = None


def _get_device() -> str:
    """自动选择最优设备。"""
    if torch.cuda.is_available():
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                     torch.cuda.memory_allocated(0)) / 1e9
        return "cuda" if free_mem > 2.0 else "cpu"
    return "cpu"


def _load_pipeline():
    """延迟加载 Grounding DINO pipeline。"""
    global _pipeline, _device
    if _pipeline is not None:
        return

    from transformers import pipeline

    _device = _get_device()
    logger.info(f"加载 Grounding DINO (device={_device})...")

    # 优先离线加载（模型已缓存时避免 SSL 错误）
    try:
        _pipeline = pipeline(
            model="IDEA-Research/grounding-dino-base",
            task="zero-shot-object-detection",
            device=_device,
            local_files_only=True,
        )
    except Exception:
        logger.info("离线加载失败，尝试联网…")
        _pipeline = pipeline(
            model="IDEA-Research/grounding-dino-base",
            task="zero-shot-object-detection",
            device=_device,
        )
    logger.info("Grounding DINO 加载完成")


def detect_objects(
    image_paths: list[Path],
    text_query: str,
    box_threshold: float = 0.25,
    use_clip: bool = True,
    use_sam: bool = True,
) -> list[dict]:
    """对多张图片运行开集物体检测。

    Args:
        image_paths: 图片文件路径列表
        text_query: 文字查询，如 "water cup" 或 "水杯"
        box_threshold: 检测置信度阈值
        use_clip: 是否启用 CLIP 语义打分（默认 True）
        use_sam: 是否启用 SAM 精确分割 mask（默认 True）

    Returns:
        [{frame_path, index, detections: [{label, score, bbox, color_hist, clip_score, mask_area}]}]
        bbox 格式: [x1, y1, x2, y2] (归一化 0-1)
    """
    _load_pipeline()

    # 预热 CLIP 文字缓存（首次调用时加载模型 + 编码文字，后续帧零推理开销）
    if use_clip:
        from backend.recognition.clip_scorer import _encode_text
        _encode_text(text_query)  # 预处理文字 embedding

    results = []
    for idx, img_path in enumerate(image_paths):
        if not img_path.exists():
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning(f"无法读取图片: {img_path}")
            continue

        h, w = image_bgr.shape[:2]

        try:
            # Grounding DINO pipeline 接受文件路径字符串或 PIL Image
            dets = _pipeline(
                str(img_path),
                candidate_labels=[text_query],
                threshold=box_threshold,
            )
        except Exception as e:
            logger.warning(f"检测失败 {img_path.name}: {e}")
            continue

        if not dets:
            continue

        frame_result = {
            "frame_path": str(img_path),
            "index": idx,
            "image_size": [w, h],
            "detections": [],
        }

        for det in dets:
            bbox = det["box"]  # [x1, y1, x2, y2] 绝对像素坐标
            # 归一化到 0-1
            bbox_norm = [
                bbox["xmin"] / w,
                bbox["ymin"] / h,
                bbox["xmax"] / w,
                bbox["ymax"] / h,
            ]

            x1, y1, x2, y2 = int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])

            # ---- SAM 精确 mask（优先）或 bbox 矩形作为降级 ----
            if use_sam:
                from backend.recognition.segmenter import generate_mask, extract_masked_histogram
                mask = generate_mask(image_bgr, (x1, y1, x2, y2))
                mask_area = int(mask.sum())
                color_hist = extract_masked_histogram(image_bgr, mask)
            else:
                mask_area = (x2 - x1) * (y2 - y1)
                roi = image_bgr[y1:y2, x1:x2]
                color_hist = _extract_hsv_histogram(roi)

            # CLIP 语义打分
            clip_score = 0.5  # 默认中性分数
            if use_clip:
                from backend.recognition.clip_scorer import score_detection
                clip_score = score_detection(image_bgr, (x1, y1, x2, y2), text_query)

            frame_result["detections"].append({
                "label": det["label"],
                "score": round(det["score"], 3),
                "bbox": [round(v, 4) for v in bbox_norm],
                "color_histogram": color_hist,
                "clip_score": round(clip_score, 4),
                "mask_area": mask_area,
            })

        if frame_result["detections"]:
            results.append(frame_result)

    logger.info(
        f"检测完成: {len(image_paths)} 张图片 → "
        f"{len(results)} 张命中, "
        f"总检测数 {sum(len(r['detections']) for r in results)}"
    )
    return results


def _extract_hsv_histogram(roi: np.ndarray) -> list[float]:
    """提取 ROI 区域的 HSV 颜色直方图 (H:16, S:8, V:8 = 1024 bins)。"""
    if roi.size == 0:
        return [0.0] * 1024

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [16, 8, 8], [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32).tolist()
