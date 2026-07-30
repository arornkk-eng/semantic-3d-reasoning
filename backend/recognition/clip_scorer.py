"""CLIP 语义打分：对检测区域做图像-文字语义匹配。

使用 openai/clip-vit-base-patch32（151M 参数，GPU ~0.6GB VRAM）。
文字 embedding 自动缓存，同一 query 只编码一次。
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---- 全局模型（延迟加载） ----
_clip_model = None
_clip_processor = None
_device: Optional[str] = None

# 文字 embedding 缓存：{text_query: normalized_tensor}
_text_cache: dict[str, torch.Tensor] = {}


def _extract_tensor(output) -> torch.Tensor:
    """兼容不同版本 transformers：从输出中提取 tensor。

    CLIPModel.get_text_features / get_image_features 在不同版本可能返回：
      - 直接返回 torch.Tensor
      - BaseModelOutputWithPooling (包含 pooler_output 属性)
      - CLIPTextModelOutput / CLIPVisionModelOutput (包含 text_embeds / image_embeds)
    """
    if isinstance(output, torch.Tensor):
        return output
    # 尝试常见属性名
    for attr in ("pooler_output", "text_embeds", "image_embeds"):
        if hasattr(output, attr):
            return getattr(output, attr)
    # 最后的降级尝试：last_hidden_state 的第一位 (CLS token)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"无法从 {type(output).__name__} 提取 tensor，请检查 transformers 版本")


def _load_clip():
    """延迟加载 CLIP 模型（首次调用时自动加载）。"""
    global _clip_model, _clip_processor, _device
    if _clip_model is not None:
        return

    from transformers import CLIPModel, CLIPProcessor

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"加载 CLIP ViT-B/32 (device={_device})...")
    # 优先离线加载（模型已缓存时不需要联网）
    try:
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        ).to(_device)
        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
    except Exception:
        logger.info("离线加载失败，尝试联网...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
    logger.info("CLIP 加载完成")


def _encode_text(text: str) -> torch.Tensor:
    """编码文字查询为归一化 embedding（自动缓存）。"""
    if text in _text_cache:
        return _text_cache[text]

    _load_clip()
    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        features = _clip_model.get_text_features(**inputs)
        features = _extract_tensor(features)
        features = features / features.norm(dim=-1, keepdim=True)

    _text_cache[text] = features
    return features


def score_detection(
    image_bgr: np.ndarray,
    bbox_abs: tuple[int, int, int, int],
    text_query: str,
) -> float:
    """计算检测框区域与文字查询的 CLIP 语义相似度。

    Args:
        image_bgr: OpenCV BGR 格式图像 (H, W, 3)
        bbox_abs: (xmin, ymin, xmax, ymax) 绝对像素坐标
        text_query: 文字查询，如 "水杯"

    Returns:
        cosine similarity，典型范围 [-0.1, 0.5]，越高越语义匹配。
        完全无关的配对通常在 0.0 附近。
    """
    _load_clip()

    x1, y1, x2, y2 = bbox_abs
    h, w = image_bgr.shape[:2]

    # 确保 bbox 在图像范围内
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # 裁剪 ROI，BGR → RGB（.copy() 修复负步长问题）
    roi_bgr = image_bgr[y1:y2, x1:x2, :]
    roi_rgb = roi_bgr[:, :, ::-1].copy()

    # CLIP 预处理
    inputs = _clip_processor(images=roi_rgb, return_tensors="pt").to(_device)
    with torch.no_grad():
        image_features = _clip_model.get_image_features(**inputs)
        image_features = _extract_tensor(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    # 文字 embedding（缓存命中则零推理开销）
    text_features = _encode_text(text_query)

    similarity = float((image_features @ text_features.T).item())
    return similarity


def clear_text_cache():
    """清空文字 embedding 缓存（模型切换或内存不足时使用）。"""
    _text_cache.clear()
    logger.debug("CLIP 文字缓存已清空")


def unload():
    """释放 CLIP 模型以回收 GPU 显存。"""
    global _clip_model, _clip_processor, _device
    if _clip_model is not None:
        _clip_model.cpu()
    _clip_model = None
    _clip_processor = None
    _device = None
    _text_cache.clear()
    logger.info("CLIP 模型已卸载")
