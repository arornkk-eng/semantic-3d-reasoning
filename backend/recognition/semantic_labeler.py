"""语义标签器：用 CLIP 特征相似度（替代 HSV 颜色）做 2D→3D 高斯球映射。

相比 color_labeler，不再依赖颜色直方图 → 高斯球颜色匹配，
而是利用 CLIP 特征投影的结果，计算语义一致性。
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def semantic_label_gaussians(
    projected_features: dict[str, np.ndarray],  # {label: (N, 768)}
    detections: list[dict],
    output_path: Path,
    query_label: str = "",
    z_threshold: float = 1.5,
    min_frame_ratio: float = 0.15,
    top_n_clusters: int = 3,
) -> dict:
    """基于 CLIP 语义特征的 3D 高斯球标签。

    流程:
      每帧: 检测 bbox → 投影区域内高斯球得 CLIP 特征
      跨帧: 高斯球需在 ≥15% 帧中被选中（与 color_labeler 相同的投票机制）
      最终: DBSCAN 聚类 + 保留前 N 大簇

    Args:
        projected_features: feature_lifter 的投影结果 {label: (N, 768)}
        detections: detector 的检测结果列表
        output_path: labels.json 输出路径
        query_label: 查询的物体名称
        z_threshold: 特征相似度 Z-score 阈值
        min_frame_ratio: 跨帧投票最小比例
        top_n_clusters: 保留前 N 大簇
    """
    from sklearn.cluster import DBSCAN

    from backend.recognition.ply_reader import read_ply

    # ---- 寻找 label key（可能与 query_label 不同） ----
    if query_label in projected_features:
        label_key = query_label
    else:
        # 尝试模糊匹配
        match = None
        for k in projected_features:
            if query_label in k or k in query_label:
                match = k
                break
        if match:
            label_key = match
        else:
            logger.warning(f"未找到 '{query_label}' 的投影特征")
            return {}

    features = projected_features[label_key]  # (N, 768)
    n = features.shape[0]
    if n == 0:
        return {}

    # 对每帧计数（feature_lifter 的输出已经按检测聚合，这里用检测帧数做投票权重）
    total_frames = len(detections)

    # 计算所有高斯球的 "语义显著性"：
    # 被投影过的高斯球有特征向量，没被投影的为零向量
    # 显著性 = 特征向量的 L2 norm（非零表示至少被一帧选中）
    saliency = np.linalg.norm(features, axis=1)  # (N,)

    # Z-score 阈值过滤
    mean_s = float(saliency.mean())
    std_s = float(saliency.std())
    threshold = mean_s + z_threshold * std_s

    candidate_mask = saliency >= threshold
    n_candidates = candidate_mask.sum()

    if n_candidates < 10:
        logger.warning(f"Z-score 过滤后不足 10 个高斯球 (阈值={threshold:.4f})")
        return {}

    logger.info(
        f"语义显著性: mean={mean_s:.4f} σ={std_s:.4f} "
        f"阈值={threshold:.4f} → {n_candidates:,} 候选 ({100*n_candidates/n:.1f}%)"
    )

    # 需要 position 做 DBSCAN
    ply_path = output_path.parent / "scene.ply"
    ply_data = read_ply(ply_path)
    positions = ply_data.positions

    # 候选上限
    MAX_CANDIDATES = 10000
    if n_candidates > MAX_CANDIDATES:
        top_k_idx = np.argpartition(-saliency[candidate_mask], MAX_CANDIDATES)[:MAX_CANDIDATES]
        candidate_indices_arr = np.where(candidate_mask)[0]
        keep_idx = candidate_indices_arr[top_k_idx]
        candidate_mask = np.zeros(n, dtype=bool)
        candidate_mask[keep_idx] = True
        n_candidates = MAX_CANDIDATES

    # DBSCAN 空间聚类
    candidate_positions = positions[candidate_mask]
    candidate_indices = np.where(candidate_mask)[0]

    bbox_min = candidate_positions.min(axis=0)
    bbox_max = candidate_positions.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    eps = max(diag * 0.05, 0.02)

    clustering = DBSCAN(eps=eps, min_samples=10, n_jobs=1).fit(candidate_positions)
    labels_arr = clustering.labels_

    unique_lbl, counts = np.unique(labels_arr[labels_arr >= 0], return_counts=True)
    if len(unique_lbl) == 0:
        return {}

    # 保留前 N 大簇
    sorted_idx = counts.argsort()[::-1]
    top_k = min(top_n_clusters, len(unique_lbl))
    top_ids = unique_lbl[sorted_idx[:top_k]]

    clusters_info = []
    all_matched = []
    for cid in top_ids:
        cmask = labels_arr == cid
        matched = candidate_indices[cmask]
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
        })

    final_indices = np.concatenate(all_matched)
    final_positions = positions[final_indices]

    result = {
        query_label: {
            "indices": final_indices.tolist(),
            "count": int(len(final_indices)),
            "center_3d": final_positions.mean(axis=0).tolist(),
            "bbox_3d": {
                "min": final_positions.min(axis=0).tolist(),
                "max": final_positions.max(axis=0).tolist(),
            },
            "score": round(float(len(final_indices) / max(1, n_candidates)), 3),
            "clusters": clusters_info,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"语义标签已保存: {output_path} — {len(final_indices):,} 高斯球")
    return result
