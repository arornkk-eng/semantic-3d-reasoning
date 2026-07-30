"""2D 检测 → 3D 高斯球映射：颜色匹配 + 空间聚类 + 多实例支持。

使用健壮的 ply_reader 模块读写 PLY（支持 binary/ASCII，按属性名查找）。
颜色相似度使用卡方距离比较 HSV 直方图（替代旧的单 bin 查表）。
"""

import json
import logging
from pathlib import Path

import numpy as np

from backend.recognition.ply_reader import read_ply

logger = logging.getLogger(__name__)

# HSV 量化参数
_H_BINS, _S_BINS, _V_BINS = 16, 8, 8
_HIST_BINS = _H_BINS * _S_BINS * _V_BINS  # 1024


def label_gaussians(
    ply_path: Path,
    detections: list[dict],
    output_path: Path,
    query_label: str = "",
    z_threshold: float = 2.0,
    min_frame_ratio: float = 0.15,
    top_n_clusters: int = 3,
) -> dict:
    """根据 2D 检测结果标记 3D 高斯球（跨帧投票版）。

    流程:
        每帧: 颜色相似度 → Z-score 阈值过滤 → DBSCAN 聚类 → 记录投票
        跨帧: 高斯球需在 ≥15% 帧中被选中才保留
        最终: 对投票通过的高斯球再做 DBSCAN → 输出标签

    Args:
        ply_path: scene.ply 文件路径
        detections: detector.detect_objects() 的返回结果
        output_path: labels.json 输出路径
        query_label: 查询的物体名称
        z_threshold: Z-score 阈值（默认 2.0，即均值+2σ）
        min_frame_ratio: 跨帧投票最小比例（默认 0.15 = 15%）
        top_n_clusters: 最终 DBSCAN 保留前 N 大簇
    """
    from sklearn.cluster import DBSCAN

    # ---- Step 1: 读取 PLY ----
    logger.info(f"读取 PLY: {ply_path}")
    ply_data = read_ply(ply_path)
    positions = ply_data.positions  # (N, 3)
    colors_rgb = ply_data.colors_rgb  # (N, 3) in [0, 1]
    n = ply_data.num_vertices
    logger.info(f"共 {n:,} 个高斯球")

    if n == 0:
        logger.error("PLY 为空")
        return {}

    # ---- Step 2: 预计算 HSV bin 索引 ----
    gaussian_hsv_bins = _rgb_to_hsv_bins_batch(colors_rgb)

    # ---- Step 3: 逐帧处理 + 累积投票 ----
    # 按 label 分组：{label: {votes, score_sum, frame_count}}
    label_state: dict[str, dict] = {}

    total_frames_with_detections = len(detections)

    for frame_result in detections:
        for det in frame_result["detections"]:
            label = det["label"]
            hist = np.array(det["color_histogram"], dtype=np.float32)

            # 3.1 计算颜色相似度（原始直方图频次，不做 max 归一化）
            color_scores = _histogram_similarity(gaussian_hsv_bins, hist)

            # 3.1b 融入 CLIP 语义权重
            # CLIP cosine 相似度典型范围 [-0.1, 0.5]，将其映射为 [0.2, 1.0] 的权重因子
            # clip ≥ 0.35 → 权重 1.0（完全信任）
            # clip = 0.0  → 权重 0.2（仅保留 20% 颜色信号）
            # clip < 0    → 权重 0.2（保底）
            clip_score = det.get("clip_score", 0.5)
            clip_weight = 0.2 + 0.8 * max(0.0, min(1.0, clip_score / 0.35))
            weighted_scores = color_scores * clip_weight

            # 3.2 Z-score 阈值过滤（使用 CLIP 加权后的分数）
            mean_s = float(weighted_scores.mean())
            std_s = float(weighted_scores.std())
            threshold = mean_s + z_threshold * std_s

            candidate_mask = weighted_scores >= threshold
            n_candidates = candidate_mask.sum()

            if n_candidates < 10:
                logger.debug(
                    f"Z-score 过滤后不足: '{label}' frame={frame_result['index']}, "
                    f"阈值={threshold:.5f} (mean={mean_s:.5f} σ={std_s:.5f}), "
                    f"候选={n_candidates}"
                )
                continue

            # 3.3 候选上限：只取分数最高的 max_candidates 个（防止 DBSCAN OOM）
            MAX_CANDIDATES = 10000
            if n_candidates > MAX_CANDIDATES:
                top_k_idx = np.argpartition(
                    -color_scores[candidate_mask], MAX_CANDIDATES
                )[:MAX_CANDIDATES]
                # 重建 mask：只保留 top-K
                candidate_indices_arr = np.where(candidate_mask)[0]
                keep_idx = candidate_indices_arr[top_k_idx]
                candidate_mask = np.zeros(n, dtype=bool)
                candidate_mask[keep_idx] = True
                n_candidates = MAX_CANDIDATES

            # 3.4 DBSCAN 空间聚类
            candidate_positions = positions[candidate_mask]
            candidate_indices = np.where(candidate_mask)[0]

            bbox_min = candidate_positions.min(axis=0)
            bbox_max = candidate_positions.max(axis=0)
            diag = float(np.linalg.norm(bbox_max - bbox_min))
            eps = max(diag * 0.05, 0.02)

            clustering = DBSCAN(eps=eps, min_samples=10, n_jobs=1).fit(candidate_positions)
            cluster_labels_arr = clustering.labels_

            unique_lbl, counts = np.unique(
                cluster_labels_arr[cluster_labels_arr >= 0], return_counts=True
            )
            if len(unique_lbl) == 0:
                continue

            # 保留最大簇（每帧只取最显著的一个簇投票）
            best_cluster = unique_lbl[counts.argmax()]
            best_mask = cluster_labels_arr == best_cluster
            voted_indices = candidate_indices[best_mask]

            # 初始化 label 状态
            if label not in label_state:
                label_state[label] = {
                    "votes": np.zeros(n, dtype=np.int32),
                    "score_sum": np.zeros(n, dtype=np.float32),
                    "frame_count": 0,
                }
            state = label_state[label]
            state["votes"][voted_indices] += 1
            state["score_sum"][voted_indices] += weighted_scores[voted_indices]
            state["frame_count"] += 1

            logger.debug(
                f"投票 '{label}' frame={frame_result['index']}: "
                f"CLIP={clip_score:.3f}→w={clip_weight:.2f} "
                f"Z>{z_threshold}σ → {candidate_mask.sum()} 候选 → "
                f"DBSCAN {len(unique_lbl)} 簇 → 最大簇 {len(voted_indices)} 个高斯球"
            )

    # ---- Step 4: 跨帧投票 + 最终聚类 ----
    all_labels: dict[str, dict] = {}
    min_frames = max(1, int(total_frames_with_detections * min_frame_ratio))

    for label, state in label_state.items():
        votes = state["votes"]
        frame_count = state["frame_count"]
        min_votes = max(1, int(frame_count * min_frame_ratio))

        voted_mask = votes >= min_votes
        n_voted = int(voted_mask.sum())

        logger.info(
            f"'{label}': {frame_count} 帧参与投票, "
            f"需 ≥{min_votes} 票 → 通过 {n_voted:,} 个高斯球 "
            f"({100 * n_voted / n:.1f}%)"
        )

        if n_voted < 10:
            logger.warning(f"'{label}' 跨帧投票后不足 10 个高斯球，丢弃")
            continue

        # 最终 DBSCAN 聚类（在投票通过的高斯球上）
        voted_positions = positions[voted_mask]
        voted_indices = np.where(voted_mask)[0]

        bbox_min = voted_positions.min(axis=0)
        bbox_max = voted_positions.max(axis=0)
        diag = float(np.linalg.norm(bbox_max - bbox_min))
        eps = max(diag * 0.05, 0.02)

        final_clustering = DBSCAN(eps=eps, min_samples=10, n_jobs=1).fit(voted_positions)
        final_labels_arr = final_clustering.labels_

        unique_lbl, counts = np.unique(
            final_labels_arr[final_labels_arr >= 0], return_counts=True
        )
        if len(unique_lbl) == 0:
            logger.warning(f"'{label}' 最终聚类失败，所有点为噪声")
            continue

        # 保留前 N 大簇
        sorted_idx = counts.argsort()[::-1]
        top_k = min(top_n_clusters, len(unique_lbl))
        top_ids = unique_lbl[sorted_idx[:top_k]]

        clusters_info = []
        all_matched = []
        for cid in top_ids:
            cmask = final_labels_arr == cid
            matched = voted_indices[cmask]
            all_matched.append(matched)
            cpos = positions[matched]
            clusters_info.append({
                "indices": matched.tolist(),
                "count": int(len(matched)),
                "center_3d": cpos.mean(axis=0).tolist(),
                "bbox_3d": {
                    "min": cpos.min(axis=0).tolist(),
                    "max": cpos.max(axis=0).tolist(),
                },
                "avg_votes": round(float(votes[matched].mean()), 1),
            })

        final_indices = np.concatenate(all_matched)
        final_positions = positions[final_indices]

        all_labels[label] = {
            "indices": final_indices.tolist(),
            "count": int(len(final_indices)),
            "center_3d": final_positions.mean(axis=0).tolist(),
            "bbox_3d": {
                "min": final_positions.min(axis=0).tolist(),
                "max": final_positions.max(axis=0).tolist(),
            },
            "score": round(float(final_indices.size / max(1, n_voted)), 3),
            "clusters": clusters_info,
        }

        logger.info(
            f"'{label}' 最终: {len(final_indices):,} 高斯球, "
            f"{len(clusters_info)} 个簇 "
            f"({[c['count'] for c in clusters_info]})"
        )

    # ---- Step 5: 保存标签 ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"标签已保存: {output_path} ({len(all_labels)} 个物体)")
    return all_labels


# ================================================================
# 内部实现：颜色相似度（卡方距离 + NumPy 向量化）
# ================================================================

def _rgb_to_hsv_bins_batch(colors_rgb: np.ndarray) -> np.ndarray:
    """批量 RGB→HSV→量化 bin 索引 (N,)，纯 NumPy 无 Python 循环。

    Args:
        colors_rgb: (N, 3) in [0, 1]

    Returns:
        (N,) int32 bin indices in [0, 1023]
    """
    # RGB → HSV (NumPy 向量化)
    r, g, b = colors_rgb[:, 0], colors_rgb[:, 1], colors_rgb[:, 2]
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    delta = max_val - min_val

    # Hue
    h = np.zeros(len(colors_rgb), dtype=np.float32)
    mask_r = (max_val == r) & (delta > 0)
    mask_g = (max_val == g) & (delta > 0)
    mask_b = (max_val == b) & (delta > 0)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h / 360.0  # → [0, 1]

    # Saturation
    s = np.where(max_val > 0, delta / (max_val + 1e-8), 0.0)

    # Value
    v = max_val

    # 量化到 bin
    h_bins = np.clip((h * _H_BINS).astype(np.int32), 0, _H_BINS - 1)
    s_bins = np.clip((s * _S_BINS).astype(np.int32), 0, _S_BINS - 1)
    v_bins = np.clip((v * _V_BINS).astype(np.int32), 0, _V_BINS - 1)

    return h_bins * (_S_BINS * _V_BINS) + s_bins * _V_BINS + v_bins


def _histogram_similarity(
    gaussian_bins: np.ndarray,
    query_hist: np.ndarray,
) -> np.ndarray:
    """基于直方图比较的相似度 (N,) — 返回原始频次，不做归一化。

    对每个高斯球：取 HSV bin 在检测框直方图中的频次 + 6 邻域加权。
    分数范围取决于直方图集中度（~0.001~0.1），由上层 Z-score 阈值过滤。

    Args:
        gaussian_bins: (N,) int32, 每个高斯球的 HSV bin 索引
        query_hist: (1024,) float32, 检测框的 HSV 直方图（已归一化到 sum=1）

    Returns:
        (N,) float32 原始相似度分数
    """
    # 基础分数：该 bin 在检测框中的频次
    scores = query_hist[gaussian_bins]

    # 加入邻域平滑（±1 bin 在 H, S, V 各维度）
    h_stride = _S_BINS * _V_BINS  # 128
    s_stride = _V_BINS            # 8
    v_stride = 1

    h_idx = gaussian_bins // h_stride
    s_idx = (gaussian_bins % h_stride) // s_stride
    v_idx = gaussian_bins % s_stride

    # 对 H 维度做 ±1 邻域（环形，因为 Hue 是环状的）
    for dh in (-1, 1):
        nh = (h_idx + dh) % _H_BINS
        neighbor_bins = nh * h_stride + s_idx * s_stride + v_idx
        scores += query_hist[neighbor_bins] * 0.5

    # 对 S 维度做 ±1 邻域（非环形）
    for ds in (-1, 1):
        ns = np.clip(s_idx + ds, 0, _S_BINS - 1)
        neighbor_bins = h_idx * h_stride + ns * s_stride + v_idx
        scores += query_hist[neighbor_bins] * 0.5

    # 对 V 维度做 ±1 邻域（非环形）
    for dv in (-1, 1):
        nv = np.clip(v_idx + dv, 0, _V_BINS - 1)
        neighbor_bins = h_idx * h_stride + s_idx * s_stride + nv
        scores += query_hist[neighbor_bins] * 0.5

    # 不做 max 归一化：保留原始直方图频次，让 Z-score 阈值自然区分
    # 邻域平滑后的分数范围 ≈ [0, ~0.2]，取决于直方图集中度
    return scores.astype(np.float32)
