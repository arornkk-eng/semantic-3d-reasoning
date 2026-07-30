"""CLIP 特征提升：2D 帧 → 3D 高斯球语义特征投影。

有相机位姿后，将每帧 SAM mask 区域的 CLIP patch 特征投影到 3D 高斯球上，
累积得到每个高斯球的语义特征向量。
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_gaussians_from_params(params_path: Path, device: str = "cuda"):
    """从 .pt 文件加载高斯球参数，重建 Gaussians 对象（用于渲染）。"""
    import sys
    zipsplat_root = Path(__file__).resolve().parents[2] / "ZipSplat-main"
    if str(zipsplat_root) not in sys.path:
        sys.path.insert(0, str(zipsplat_root))

    from zipsplat.gaussians import Gaussians

    data = torch.load(params_path, map_location=device, weights_only=True)
    gaussians = Gaussians.from_parameters(
        means=data["means"].to(device),
        scales=data["scales"].to(device),
        quats=data["quats"].to(device),
        opacities=data["opacities"].to(device),
        sh_coeffs=data["sh_coeffs"].to(device),
    )
    return gaussians


def extract_clip_patch_features(
    image_bgr: np.ndarray,
    bbox_abs: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """提取检测框区域的 CLIP dense patch 特征（49 tokens × 768 dim）。

    Returns:
        (49, 768) float32 — 7×7 spatial grid flattened
    """
    from transformers import CLIPModel, CLIPProcessor

    # 延迟加载 CLIP（全局复用）
    global _clip_model, _clip_processor
    if "_clip_model" not in globals() or _clip_model is None:
        _clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        ).to("cuda")
        _clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", local_files_only=True
        )
        _clip_model.eval()

    x1, y1, x2, y2 = bbox_abs
    h, w = image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((49, 768), dtype=np.float32)

    # 裁剪并转为 RGB
    roi = image_bgr[y1:y2, x1:x2, :]
    if mask is not None:
        roi_mask = mask[y1:y2, x1:x2]
        roi = roi.copy()
        roi[~roi_mask] = 0  # 背景置零

    roi_rgb = roi[:, :, ::-1].copy()  # BGR → RGB

    inputs = _clip_processor(images=roi_rgb, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = _clip_model.vision_model(**inputs)

    # (1, 50, 768) → skip CLS, keep 49 patch tokens
    features = outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # (49, 768)
    return features.astype(np.float32)


def project_features_to_gaussians(
    gaussians,
    poses_w2c: np.ndarray,       # (N, 4, 4)
    Ks: np.ndarray,               # (N, 3, 3)
    image_sizes: np.ndarray,      # (N, 2) [w, h]
    detections_per_frame: list[list[dict]],
    images_bgr: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """将 2D 检测的 CLIP 特征投影累积到 3D 高斯球上。

    对每帧的每个检测：
      1. 提取 bbox 区域 CLIP patch 特征
      2. 用相机位姿把高斯球投影到该帧上
      3. 投影落在 mask 内的高斯球 → 累积该检测的 CLIP 特征

    Args:
        gaussians: zipsplat Gaussians 对象
        poses_w2c: (N, 4, 4) 世界→相机矩阵
        Ks: (N, 3, 3) 相机内参
        image_sizes: (N, 2) 图像尺寸
        detections_per_frame: [[det1, det2, ...], ...] 每帧的检测列表
        images_bgr: [np.array, ...] 每帧的 BGR 图像

    Returns:
        {label: (num_gaussians, 768) float32} 每个标签的高斯球特征累积
    """
    import sys
    zipsplat_root = Path(__file__).resolve().parents[2] / "ZipSplat-main"
    if str(zipsplat_root) not in sys.path:
        sys.path.insert(0, str(zipsplat_root))

    from zipsplat.pose import Pose
    from zipsplat.camera import Camera

    device = gaussians.device
    num_gaussians = gaussians.num_gaussians
    means = gaussians.means.detach()  # (N_g, 3)
    means_homo = torch.cat([
        means, torch.ones(num_gaussians, 1, device=device)
    ], dim=-1)  # (N_g, 4)

    # 每个标签的特征累积器
    accumulators: dict[str, np.ndarray] = {}  # {label: (N_g, 768)}
    counts: dict[str, np.ndarray] = {}         # {label: (N_g,)}

    for frame_idx, detections in enumerate(detections_per_frame):
        if not detections:
            continue

        w, h = int(image_sizes[frame_idx][0]), int(image_sizes[frame_idx][1])
        K = torch.from_numpy(Ks[frame_idx]).float().to(device)
        w2c = torch.from_numpy(poses_w2c[frame_idx]).float().to(device)

        # 构建 Pose 和 Camera
        # w2c 是 4×4 world-to-cam 矩阵 → 求逆得 cam-to-world
        c2w = torch.linalg.inv(w2c)
        pose = Pose.from_4x4mat(c2w)
        camera = Camera.from_K(K, w, h)

        # 将高斯球投影到该帧
        gs_cam = (w2c @ means_homo.T).T  # (N_g, 4) 相机坐标系
        gs_cam = gs_cam[:, :3] / (gs_cam[:, 3:4] + 1e-8)  # 透视除法

        # 投影到像素坐标
        gs_px = (K[:3, :3] @ gs_cam.T).T  # (N_g, 3)
        gs_px = gs_px[:, :2] / (gs_px[:, 2:3] + 1e-8)  # (N_g, 2) pixel coords

        # 可见性：z>0 且在图像范围内
        visible = (gs_cam[:, 2] > 0.01) & \
                  (gs_px[:, 0] >= 0) & (gs_px[:, 0] < w) & \
                  (gs_px[:, 1] >= 0) & (gs_px[:, 1] < h)
        visible_idx = visible.nonzero(as_tuple=True)[0].cpu().numpy()
        px_coords = gs_px[visible].cpu().numpy()  # (M, 2)

        img_bgr = images_bgr[frame_idx]

        for det in detections:
            label = det["label"]
            bbox = det["bbox"]  # [x1, y1, x2, y2] normalized

            # 绝对 bbox
            x1, y1, x2, y2 = (
                int(bbox[0] * w), int(bbox[1] * h),
                int(bbox[2] * w), int(bbox[3] * h),
            )

            # 提取 CLIP patch 特征
            clip_feat = extract_clip_patch_features(img_bgr, (x1, y1, x2, y2))
            # clip_feat: (49, 768)

            # 对每个可见高斯球，判断是否在 bbox 内
            in_bbox = (px_coords[:, 0] >= x1) & (px_coords[:, 0] < x2) & \
                      (px_coords[:, 1] >= y1) & (px_coords[:, 1] < y2)

            if in_bbox.sum() < 10:
                continue

            # 简单策略：bbox 内所有高斯球获得该检测框的平均 CLIP 特征
            avg_feat = clip_feat.mean(axis=0)  # (768,)

            if label not in accumulators:
                accumulators[label] = np.zeros((num_gaussians, 768), dtype=np.float32)
                counts[label] = np.zeros(num_gaussians, dtype=np.int32)

            gs_indices = visible_idx[in_bbox]
            accumulators[label][gs_indices] += avg_feat
            counts[label][gs_indices] += 1

    # 归一化（平均）
    result = {}
    for label in accumulators:
        cnt = counts[label]
        acc = accumulators[label]
        mask = cnt > 0
        acc[mask] = acc[mask] / cnt[mask, np.newaxis]
        result[label] = acc

    logger.info(f"特征投影完成: {list(result.keys())}")
    return result
