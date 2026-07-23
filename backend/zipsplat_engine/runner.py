"""ZipSplat 3D 重建引擎 — 主进程内执行模型推理和 PLY 导出。

torch 和 zipsplat 在函数内懒加载，避免服务器启动时占用 GPU 显存。
"""

import logging
import sys
from pathlib import Path

from backend.core.config import OUTPUT_DIR, PROJECT_ROOT, UPLOAD_DIR

logger = logging.getLogger(__name__)

_ZIPSPLAT_ROOT = PROJECT_ROOT / "ZipSplat-main"
if str(_ZIPSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZIPSPLAT_ROOT))


def run_reconstruction(task_id: str) -> dict:
    """执行 ZipSplat 3D 重建：图片 → 高斯模型 → PLY。"""
    import torch
    from zipsplat import ZipSplat, load_image

    input_dir = UPLOAD_DIR / task_id
    output_dir = OUTPUT_DIR / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise FileNotFoundError(f"输入目录无图片: {input_dir}")

    logger.info(f"任务 {task_id}: {len(image_paths)} 张图片")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = ZipSplat(weights="zipsplat").to(device).eval()
    images = [load_image(p) for p in image_paths]

    with torch.no_grad():
        gaussians = model(images, compression=1.0)[0]

    num_gs = gaussians.num_gaussians
    logger.info(f"已生成 {num_gs:,} 个高斯球")

    # ---- 颜色后处理：除白 + 饱和度修正 ----
    # 3DGS 渲染是 alpha 混合：数千个半透明高斯球叠加。
    # 不透明度 boost 会急剧加速白色叠加 → 整体发白。
    # 策略：不透明度保持原值；SH1 归零防方向白化；DC 色做温和饱和度修正。
    _SH_C0 = 0.28209479177387814

    # 1) 诊断：打印原始 DC 颜色统计
    dc = gaussians.sh0.squeeze(-2)                          # (N, 3)
    rgb_raw = (dc * _SH_C0 + 0.5).clamp(0, 1)               # SH → RGB
    logger.info(
        f"原始 DC 颜色 — R: {rgb_raw[:,0].mean():.3f}  "
        f"G: {rgb_raw[:,1].mean():.3f}  "
        f"B: {rgb_raw[:,2].mean():.3f}  "
        f"(≈{rgb_raw.mean():.3f})"
    )

    # 2) 饱和度修正：温和提升（1.2×），不改亮度
    SAT_BOOST = 1.5
    lum = 0.299 * rgb_raw[:, 0] + 0.587 * rgb_raw[:, 1] + 0.114 * rgb_raw[:, 2]
    rgb_sat = torch.stack([
        lum + SAT_BOOST * (rgb_raw[:, 0] - lum),
        lum + SAT_BOOST * (rgb_raw[:, 1] - lum),
        lum + SAT_BOOST * (rgb_raw[:, 2] - lum),
    ], dim=-1).clamp(0, 1)
    gaussians.sh0.copy_(((rgb_sat - 0.5) / _SH_C0).unsqueeze(-2))

    # 3) SH1 → 衰减 90%（不完全归零，保留微弱方向感）
    gaussians.shN.mul_(0.1)

    # 4) 不透明度：只 clamp，不 boost（boost 是白色的根因）
    gaussians.opacities.clamp_(0, 1)
    logger.info(
        f"不透明度 — mean: {gaussians.opacities.mean():.4f}, "
        f"max: {gaussians.opacities.max():.4f}"
    )
    logger.info("颜色后处理完成")

    # ---- 背景剔除：透明度过滤 + DBSCAN 空间聚类 ----
    gaussians = _filter_background(gaussians)
    num_gs_clean = gaussians.num_gaussians

    ply_path = output_dir / "scene.ply"
    gaussians.save_ply(str(ply_path))
    ply_size = ply_path.stat().st_size
    logger.info(f"PLY 已保存: {ply_path} ({ply_size / 1024:.1f} KB)")

    del model, gaussians, images
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "num_gaussians": num_gs_clean,
        "ply_size": ply_size,
    }


def _filter_background(gaussians, alpha_threshold=0.05, eps_ratio=0.03, min_samples=12, top_n=3):
    """剔除背景/漂浮高斯球：透明度 + DBSCAN 空间聚类。

    Args:
        gaussians: Gaussians 对象
        alpha_threshold: 透明度阈值，低于此值的高斯球被删除
        eps_ratio: DBSCAN eps = 场景对角线 × eps_ratio
        min_samples: DBSCAN 核心点最小邻居数
        top_n: 保留前 N 大簇的并集（降低位姿误差导致的碎片丢弃）

    Returns:
        过滤后的 Gaussians 对象
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    import torch

    n_before = gaussians.num_gaussians

    # ============ Step 1: 透明度过滤 ============
    opacity = gaussians.opacities.detach()
    mask_alpha = opacity > alpha_threshold
    n_after_alpha = mask_alpha.sum().item()
    logger.info(
        f"透明度过滤: {n_before:,} → {n_after_alpha:,} "
        f"(删除 {n_before - n_after_alpha:,}, alpha ≤ {alpha_threshold})"
    )

    if n_after_alpha < min_samples:
        logger.warning("透明度过滤后剩余点数不足，跳过空间聚类")
        return gaussians

    # ============ Step 2: 提取过滤后的空间位置 ============
    means_all = gaussians.means.detach()
    means_filtered = means_all[mask_alpha].cpu().numpy()  # (M, 3)

    # ============ Step 3: DBSCAN 聚类 ============
    # eps 自适应：场景包围盒对角线 × eps_ratio
    bbox_min = means_filtered.min(axis=0)
    bbox_max = means_filtered.max(axis=0)
    scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
    eps = scene_diag * eps_ratio
    logger.info(
        f"场景范围: {scene_diag:.2f}, eps={eps:.3f}, min_samples={min_samples}"
    )

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(means_filtered)
    labels = clustering.labels_  # (M,)  -1=噪声, 0/1/2...=簇编号
    n_noise = int((labels == -1).sum())
    cluster_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    logger.info(
        f"DBSCAN: {len(cluster_ids)} 个簇, {n_noise} 个噪声点, "
        f"簇大小: {sorted(counts.tolist(), reverse=True)[:5]}"
    )

    # ============ Step 4: 保留前 top_n 大簇的并集 ============
    if len(cluster_ids) == 0:
        logger.warning("DBSCAN 未发现任何簇，保留透明度过滤后的全部高斯球")
        mask_cluster = np.ones(len(labels), dtype=bool)
    else:
        # 按簇大小降序，取前 top_n 个
        sorted_idx = counts.argsort()[::-1]                       # 降序排列
        top_k = min(top_n, len(cluster_ids))
        top_labels = cluster_ids[sorted_idx[:top_k]]
        top_sizes = counts[sorted_idx[:top_k]]
        mask_cluster = np.isin(labels, top_labels)
        logger.info(
            f"保留前 {top_k} 大簇: {top_sizes.tolist()}, "
            f"合计 {mask_cluster.sum():,} 个高斯球"
        )

    # 将簇 mask 映射回原始索引
    idx_filtered = torch.where(mask_alpha)[0]                    # 透明度过滤后的原始索引
    idx_keep = idx_filtered[torch.from_numpy(mask_cluster)]       # 保留簇的原始索引
    final_mask = torch.zeros(n_before, dtype=torch.bool, device=idx_keep.device)
    final_mask[idx_keep] = True
    n_final = final_mask.sum().item()

    # ============ Step 5: 构建过滤后的 Gaussians ============
    from zipsplat.gaussians import Gaussians

    gaussians = Gaussians.from_parameters(
        means=gaussians.means[final_mask],
        scales=gaussians.scales[final_mask],
        quats=gaussians.quats[final_mask],
        opacities=gaussians.opacities[final_mask],
        sh_coeffs=gaussians.sh_coeffs[final_mask],
    )
    logger.info(
        f"背景剔除完成: {n_before:,} → {n_final:,} "
        f"({100 * n_final / n_before:.1f}%, 删除 {n_before - n_final:,})"
    )

    return gaussians
