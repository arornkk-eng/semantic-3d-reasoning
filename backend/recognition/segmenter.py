"""SAM 精确分割：Grounding DINO bbox → SAM mask → 纯净颜色直方图。

使用 facebook/sam-vit-base（~360MB，GPU ~1.2GB VRAM）。
只统计 mask 内像素 → 直方图不受背景污染 → 2D→3D 映射更精准。
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---- 全局模型（延迟加载） ----
_sam_model = None
_sam_processor = None
_sam_device: str = "cpu"


def _load_sam():
    """延迟加载 SAM 模型。"""
    global _sam_model, _sam_processor, _sam_device
    if _sam_model is not None:
        return

    from transformers import SamModel, SamProcessor

    _sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载 SAM ViT-B (device={_sam_device})...")

    # 优先离线加载
    try:
        _sam_model = SamModel.from_pretrained(
            "facebook/sam-vit-base", local_files_only=True
        ).to(_sam_device)
        _sam_processor = SamProcessor.from_pretrained(
            "facebook/sam-vit-base", local_files_only=True
        )
    except Exception:
        logger.info("SAM 离线加载失败，尝试联网…")
        _sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(_sam_device)
        _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    _sam_model.eval()
    logger.info("SAM 加载完成")


def generate_mask(
    image_bgr: np.ndarray,
    bbox_abs: tuple[int, int, int, int],
) -> np.ndarray:
    """根据 bbox 提示生成 SAM 精确分割 mask。

    Args:
        image_bgr: OpenCV BGR 图像 (H, W, 3)
        bbox_abs: (xmin, ymin, xmax, ymax) 绝对像素坐标

    Returns:
        (H, W) numpy bool 数组，True = 物体区域
    """
    _load_sam()

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_abs

    # 边界裁剪
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)

    # BGR → RGB
    image_rgb = image_bgr[:, :, ::-1].copy()

    # SAM 的 bbox 输入格式：[[[x1, y1, x2, y2]]]（batch × prompts × coords）
    input_boxes = [[[float(x1), float(y1), float(x2), float(y2)]]]

    try:
        inputs = _sam_processor(
            images=image_rgb,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(_sam_device)

        with torch.no_grad():
            outputs = _sam_model(**inputs)

        # 后处理 mask — 兼容 SAM/SAM2 不同输出形状
        masks_list = _sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        masks = masks_list[0]  # 取第一个（唯一）batch

        # masks 可能形状: (num_masks, H, W) 或 (1, num_masks, H, W)
        if masks.ndim == 3 and masks.shape[0] > 1:
            # (num_masks, H, W) — 多个候选 mask
            iou_scores = outputs.iou_scores
            if iou_scores.ndim == 2:
                iou_scores = iou_scores[0]  # (num_masks,)
            best_idx = int(iou_scores.argmax().item())
            mask = masks[best_idx].numpy().astype(bool)
        elif masks.ndim == 3 and masks.shape[0] == 1:
            mask = masks[0].numpy().astype(bool)
        elif masks.ndim == 2:
            mask = masks.numpy().astype(bool)
        else:
            # 降级：全部当作物体
            logger.warning(f"SAM mask 意外形状 {masks.shape}，降级为 bbox 区域")
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True

    except Exception as e:
        logger.warning(f"SAM 分割失败，降级为 bbox 全区域: {e}")
        # 降级：用 bbox 矩形作为 mask
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True

    return mask


def extract_masked_histogram(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    h_bins: int = 16,
    s_bins: int = 8,
    v_bins: int = 8,
) -> list[float]:
    """从 mask 区域提取 HSV 直方图 — 仅统计 mask=True 的像素。

    相比 bbox 矩形区域，mask 去除了背景像素，直方图更纯净。
    """
    import cv2

    total_bins = h_bins * s_bins * v_bins

    n_masked = int(mask.sum())
    if n_masked < 10:
        return [0.0] * total_bins

    # 提取 mask 内像素
    masked_pixels = image_bgr[mask]  # (K, 3) BGR
    if len(masked_pixels) == 0:
        return [0.0] * total_bins

    # 转为 (1, K, 3) 以匹配 cv2.calcHist 输入要求
    hsv = cv2.cvtColor(masked_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1, 2], None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32).tolist()


def unload():
    """释放 SAM 模型以回收 GPU 显存。"""
    global _sam_model, _sam_processor, _sam_device
    if _sam_model is not None:
        _sam_model.cpu()
    _sam_model = None
    _sam_processor = None
    _sam_device = "cpu"
    logger.info("SAM 模型已卸载")
