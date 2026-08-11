"""ZipSplat 3D 重建引擎 — 主进程内执行模型推理和 PLY 导出。

torch 和 zipsplat 在函数内懒加载，避免服务器启动时占用 GPU 显存。
"""

import logging
import sys

from backend.core.config import (
    DEFAULT_NUM_VIEWS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SCENE_ALPHA_THRESHOLD,
    SCENE_OUTLIER_PERCENTILE,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

_ZIPSPLAT_ROOT = PROJECT_ROOT / "ZipSplat-main"
if str(_ZIPSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ZIPSPLAT_ROOT))


def run_reconstruction(
    task_id: str, mode: str = "object", num_views: int = DEFAULT_NUM_VIEWS
) -> dict:
    """执行 ZipSplat 3D 重建：图片 → 高斯模型 → PLY。

    Args:
        task_id: 任务 ID
        mode: 重建模式 — \"object\"（物体模式，DBSCAN 剔除背景）
              \"scene\"（场景模式，保留全部高斯球）
        num_views: 自动视图选择目标数。输入图片超过该数时，先自动挑出
             最优组合（质量过滤 + SfM 位姿贪心/CLIP 回退）再重建；
             设为 0 跳过选择。
    """
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

    # ---- 自动视图选择:输入过多时挑出最优组合 ----
    view_sel = None
    if num_views > 0 and len(image_paths) > num_views:
        _sys = __import__("sys")
        script_dir = PROJECT_ROOT / "ZipSplat-Demo" / "scripts"
        if str(script_dir) not in _sys.path:
            _sys.path.insert(0, str(script_dir))
        from pick_views import select_best_views

        logger.info(f"任务 {task_id}: 输入 {len(image_paths)} 张 > {num_views}, 自动视图选择中...")
        try:
            sel = select_best_views(input_dir, num=num_views)
            chosen = sel["chosen"]
            if chosen:
                chosen_set = set(chosen)
                image_paths = [p for p in image_paths if p.name in chosen_set]
                view_sel = {
                    "method": sel["method"],
                    "total": sel["total"],
                    "passed": sel["passed"],
                    "registered": sel["registered"],
                    "selected": len(image_paths),
                }
                logger.info(
                    f"任务 {task_id}: 视图选择({sel['method']}) "
                    f"{sel['total']}→{sel['passed']}→{len(image_paths)} 张, "
                    f"SfM 注册 {sel['registered']}/{sel['passed']}"
                )
        except Exception as e:
            logger.warning(f"任务 {task_id}: 视图选择失败({e}), 回退使用全部 {len(image_paths)} 张")

    logger.info(f"任务 {task_id}: {len(image_paths)} 张图片")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    model = ZipSplat(weights="zipsplat").to(device).eval()
    images = [load_image(p) for p in image_paths]

    with torch.no_grad():
        gaussians = model(images, compression=1.0)[0]

    num_gs = gaussians.num_gaussians
    logger.info(f"已生成 {num_gs:,} 个高斯球")

    # ---- 颜色后处理：除白 + 饱和度修正 ----
    # 3DGS 渲染是 alpha 混合：数千个半透明高斯球叠加。
    # 不透明度 boost 会急剧加速白色叠加 → 整体发白。
    # 策略：不透明度保持原值；SH1 全保留保持方向感；DC 色做温和饱和度修正。
    _SH_C0 = 0.28209479177387814

    # 1) 诊断：打印原始 DC 颜色统计
    dc = gaussians.sh0.squeeze(-2)  # (N, 3)
    rgb_raw = (dc * _SH_C0 + 0.5).clamp(0, 1)  # SH → RGB
    logger.info(
        f"原始 DC 颜色 — R: {rgb_raw[:, 0].mean():.3f}  "
        f"G: {rgb_raw[:, 1].mean():.3f}  "
        f"B: {rgb_raw[:, 2].mean():.3f}  "
        f"(≈{rgb_raw.mean():.3f})"
    )

    # 2) 饱和度修正：温和提升（1.2×），不改亮度
    SAT_BOOST = 1.2
    lum = 0.299 * rgb_raw[:, 0] + 0.587 * rgb_raw[:, 1] + 0.114 * rgb_raw[:, 2]
    rgb_sat = torch.stack(
        [
            lum + SAT_BOOST * (rgb_raw[:, 0] - lum),
            lum + SAT_BOOST * (rgb_raw[:, 1] - lum),
            lum + SAT_BOOST * (rgb_raw[:, 2] - lum),
        ],
        dim=-1,
    ).clamp(0, 1)
    gaussians.sh0.copy_(((rgb_sat - 0.5) / _SH_C0).unsqueeze(-2))

    # 3) SH1 → 保留全部方向感
    logger.info(f"SH1 全保留 — mean: {gaussians.shN.mean():.4f}, std: {gaussians.shN.std():.4f}")

    # 4) 不透明度：只 clamp，不 boost
    gaussians.opacities.clamp_(0, 1)
    logger.info(
        f"不透明度 — mean: {gaussians.opacities.mean():.4f}, max: {gaussians.opacities.max():.4f}"
    )
    logger.info("颜色后处理完成")

    # ---- 后处理：去噪 ----
    gaussians = _filter_background(gaussians) if mode == "object" else _cleanup_scene(gaussians)
    num_gs_clean = gaussians.num_gaussians

    ply_path = output_dir / "scene.ply"
    gaussians.save_ply(str(ply_path))
    ply_size = ply_path.stat().st_size
    logger.info(f"PLY 已保存: {ply_path} ({ply_size / 1024:.1f} KB)")

    # 保存高斯球参数（供位姿估算 + 特征投影使用）
    params_path = output_dir / "gaussians.pt"
    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "scales": gaussians.scales.detach().cpu(),
            "quats": gaussians.quats.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
            "sh_coeffs": gaussians.sh_coeffs.detach().cpu(),
        },
        params_path,
    )
    logger.info(f"高斯参数已保存: {params_path}")

    del model, gaussians, images
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "num_gaussians": num_gs_clean,
        "ply_size": ply_size,
        "view_selection": view_sel,
    }


def _cleanup_scene(
    gaussians,
    alpha_threshold=SCENE_ALPHA_THRESHOLD,
    percentile=SCENE_OUTLIER_PERCENTILE,
):
    """场景模式轻量去噪：仅剔除极低透明度 + 空间分布的尾部离群点。

    只删最明显的漂浮碎片，不碰墙体/地面等大面积表面。
    使用百分位数而不是固定距离阈值来适应不同场景尺度。
    """
    import numpy as np
    import torch

    n_before = gaussians.num_gaussians

    # Step 1: 透明度过滤（极其宽松，只删几乎完全透明的）
    opacity = gaussians.opacities.detach()
    mask_alpha = opacity > alpha_threshold
    n_alpha = mask_alpha.sum().item()
    logger.info(
        f"透明度过滤: {n_before:,} → {n_alpha:,} "
        f"(删除 {n_before - n_alpha:,}, alpha ≤ {alpha_threshold})"
    )

    if n_alpha < 100:
        logger.warning("透明度过滤后高斯球太少，保留原始结果")
        return gaussians

    # Step 2: 空间尾部离群点过滤（只删最极端的孤立法）
    means = gaussians.means.detach()[mask_alpha].cpu()
    from scipy.spatial import cKDTree

    tree = cKDTree(means.numpy())
    # 只用最近邻距离（非平均距离），对大面积表面更友好
    dists, _ = tree.query(means.numpy(), k=2)
    nn_dist = dists[:, 1]  # 到最近邻的距离

    # 用高分位数做阈值：只删最远端的 percentile% 孤立法
    # 默认只删最远的 1%，降低合法边缘结构被误删的风险。
    cutoff = float(np.percentile(nn_dist, 100 - percentile))
    keep_local = nn_dist <= cutoff

    n_removed = int((~keep_local).sum())
    logger.info(
        f"孤立点过滤: 最近邻距离中位数={np.median(nn_dist):.4f}, "
        f"阈值(p{100 - percentile})={cutoff:.4f}, 删除 {n_removed:,} 个尾部离群点"
    )

    alpha_indices = torch.where(mask_alpha)[0]
    keep_indices = alpha_indices[torch.from_numpy(keep_local)]
    final_mask = torch.zeros(n_before, dtype=torch.bool, device=gaussians.means.device)
    final_mask[keep_indices] = True
    n_final = final_mask.sum().item()

    if n_final < 100:
        logger.warning("去噪后高斯球太少，保留原始结果")
        return gaussians

    from zipsplat.gaussians import Gaussians

    gaussians = Gaussians.from_parameters(
        means=gaussians.means[final_mask],
        scales=gaussians.scales[final_mask],
        quats=gaussians.quats[final_mask],
        opacities=gaussians.opacities[final_mask],
        sh_coeffs=gaussians.sh_coeffs[final_mask],
    )
    logger.info(
        f"场景去噪: {n_before:,} → {n_final:,} "
        f"({100 * n_final / n_before:.1f}%, 删除 {n_before - n_final:,})"
    )
    return gaussians


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
    import torch
    from sklearn.cluster import DBSCAN

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

    # 如果点数过多（>80K），随机采样以避免 DBSCAN 内存爆炸
    MAX_DBSCAN_POINTS = 80000
    sample_indices = None
    if len(means_filtered) > MAX_DBSCAN_POINTS:
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(means_filtered), MAX_DBSCAN_POINTS, replace=False)
        means_for_cluster = means_filtered[sample_indices]
        logger.info(f"DBSCAN 采样: {len(means_filtered):,} → {MAX_DBSCAN_POINTS:,}")
    else:
        means_for_cluster = means_filtered

    # ============ Step 3: DBSCAN 聚类 ============
    # eps 自适应：场景包围盒对角线 × eps_ratio
    bbox_min = means_for_cluster.min(axis=0)
    bbox_max = means_for_cluster.max(axis=0)
    scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
    eps = scene_diag * eps_ratio
    logger.info(f"场景范围: {scene_diag:.2f}, eps={eps:.3f}, min_samples={min_samples}")

    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1).fit(means_for_cluster)
    cluster_labels = clustering.labels_  # (M_sample,)  -1=噪声, 0/1/2...=簇编号
    n_noise = int((cluster_labels == -1).sum())
    cluster_ids, counts = np.unique(cluster_labels[cluster_labels >= 0], return_counts=True)
    logger.info(
        f"DBSCAN: {len(cluster_ids)} 个簇, {n_noise} 个噪声点, "
        f"簇大小: {sorted(counts.tolist(), reverse=True)[:5]}"
    )

    # ============ Step 4: 保留前 top_n 大簇的并集 ============
    if len(cluster_ids) == 0:
        logger.warning("DBSCAN 未发现任何簇，保留透明度过滤后的全部高斯球")
        mask_cluster = np.ones(len(means_filtered), dtype=bool)
    else:
        # 按簇大小降序，取前 top_n 个
        sorted_idx = counts.argsort()[::-1]
        top_k = min(top_n, len(cluster_ids))
        top_labels = cluster_ids[sorted_idx[:top_k]]
        top_sizes = counts[sorted_idx[:top_k]]
        top_in_cluster = np.isin(cluster_labels, top_labels)
        logger.info(
            f"保留前 {top_k} 大簇: {top_sizes.tolist()}, 合计 {top_in_cluster.sum():,} 个高斯球"
        )

        # 将聚类结果映射回 means_filtered 的空间
        if sample_indices is not None:
            # 从采样点映射回全量点：最近邻分配（只用欧氏距离，避免 KDTree）
            mask_cluster = np.zeros(len(means_filtered), dtype=bool)
            # 对每个全量点，找最近的采样点，继承其簇标签
            from scipy.spatial import cKDTree

            tree = cKDTree(means_for_cluster)
            _, nn_idx = tree.query(means_filtered, k=1, workers=1)
            mask_cluster = top_in_cluster[nn_idx]
        else:
            mask_cluster = top_in_cluster

    # 将簇 mask 映射回原始索引
    idx_filtered = torch.where(mask_alpha)[0]
    idx_keep = idx_filtered[torch.from_numpy(mask_cluster)]
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
