"""3D 自监督语义发现：不依赖 2D 检测和相机位姿，直接在 3D 高斯球上聚类发现物体。

原理：
  1. 对高斯球采样 → 计算空间+颜色+尺度联合相似度
  2. 构建稀疏 k-NN 相似度图 → Spectral Clustering
  3. 标签传播到全部高斯球 → 返回结构化的物体簇
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def discover_objects(
    ply_path: Path,
    output_path: Path,
    n_clusters: int = 5,
    n_samples: int = 5000,
    pos_weight: float = 1.0,
    col_weight: float = 0.5,
    min_cluster_size: int = 100,
) -> list[dict]:
    """在 3D 高斯球上做无监督聚类发现物体/区域。

    Args:
        ply_path: scene.ply 路径
        output_path: 结果 JSON 输出路径
        n_clusters: 期望发现的物体/区域数量
        n_samples: 采样点数（控制聚类速度，5K 约 1-3 秒）
        pos_weight: 空间邻近度权重（大值 → 更强调空间连续性）
        col_weight: 颜色相似度权重（大值 → 更强调外观一致性）
        min_cluster_size: 最小簇大小（小于此值的簇被丢弃）

    Returns:
        [{cluster_id, count, center_3d, bbox_3d, dominant_color}]
    """
    from scipy.sparse import csr_matrix
    from sklearn.cluster import SpectralClustering
    from sklearn.neighbors import NearestNeighbors

    from backend.recognition.ply_reader import read_ply

    # ---- Step 1: 读取 PLY ----
    logger.info(f"读取 PLY: {ply_path}")
    ply = read_ply(ply_path)
    positions = ply.positions  # (N, 3)
    colors = ply.colors_rgb    # (N, 3) in [0, 1]
    n = ply.num_vertices

    if n == 0:
        return []

    if n_clusters > n_samples:
        n_clusters = min(n_clusters, n // min_cluster_size)
        logger.warning(f"n_clusters 超过样本数，调整为 {n_clusters}")

    # ---- Step 2: 提取辅助特征 ----
    # 平均尺度（exp 因为 ZipSplat 存 log-scale）
    try:
        s0, s1, s2 = ply.col("scale_0"), ply.col("scale_1"), ply.col("scale_2")
        avg_scale = np.exp(np.column_stack([s0, s1, s2])).mean(axis=1)
    except KeyError:
        avg_scale = np.ones(n)

    # 不透明度（sigmoid 激活）
    try:
        opacity = ply.col("opacity")
        opacity = 1.0 / (1.0 + np.exp(-opacity))
    except KeyError:
        opacity = np.ones(n)

    # ---- Step 3: 特征归一化 ----
    pos_norm = (positions - positions.mean(axis=0)) / (positions.std(axis=0) + 1e-8)
    col_norm = colors  # 已在 [0, 1]，直接使用
    scale_norm = (avg_scale - avg_scale.mean()) / (avg_scale.std() + 1e-8)
    opacity_norm = (opacity - opacity.mean()) / (opacity.std() + 1e-8)

    # ---- Step 4: 降采样 ----
    if n > n_samples:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(n, n_samples, replace=False)
    else:
        sample_idx = np.arange(n)
        n_samples = n

    sample_pos = pos_norm[sample_idx]
    sample_col = col_norm[sample_idx]
    logger.info(f"采样 {n_samples:,} / {n:,} 个高斯球")

    # ---- Step 5: k-NN 稀疏相似度图 ----
    n_neighbors = min(50, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=1)
    nn.fit(sample_pos)
    distances, indices = nn.kneighbors(sample_pos)

    # 自适应 sigma：用 k-NN 中位数距离
    sigma_pos = float(np.median(distances[:, 1:])) + 1e-8

    row_idx, col_idx, edge_vals = [], [], []
    for i in range(n_samples):
        for k in range(1, n_neighbors):  # 跳过自身 (k=0)
            j = indices[i, k]
            if j <= i:  # 去重：只保留 i < j 的上三角
                continue

            # 空间相似度
            d_pos = distances[i, k]
            sim_pos = np.exp(-d_pos ** 2 / (2 * sigma_pos ** 2))

            # 颜色相似度
            d_col = float(np.linalg.norm(sample_col[i] - sample_col[j]))
            sim_col = np.exp(-d_col ** 2 / 0.1)

            # 综合
            sim = pos_weight * sim_pos + col_weight * sim_col

            if sim > 0.01:
                row_idx.extend([i, j])
                col_idx.extend([j, i])
                edge_vals.extend([sim, sim])

    affinity = csr_matrix(
        (edge_vals, (row_idx, col_idx)),
        shape=(n_samples, n_samples),
    )
    n_edges = len(edge_vals) // 2
    logger.info(
        f"相似度图: {n_edges:,} 条边 "
        f"({100 * n_edges / (n_samples * (n_samples - 1) / 2):.1f}% 密度, "
        f"σ_pos={sigma_pos:.3f})"
    )

    # ---- Step 6: Spectral Clustering ----
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
        n_jobs=1,
    )
    sample_labels = clustering.fit_predict(affinity)
    logger.info(
        f"谱聚类完成: {n_clusters} 个簇, "
        f"分布={np.bincount(sample_labels[sample_labels >= 0])}"
    )

    # ---- Step 7: 标签传播（最近邻映射回全量高斯球） ----
    nn_all = NearestNeighbors(n_neighbors=1, n_jobs=1)
    nn_all.fit(pos_norm[sample_idx])
    _, all_nn_idx = nn_all.kneighbors(pos_norm)
    all_labels = sample_labels[all_nn_idx.flatten()]

    # ---- Step 8: 整理结果 ----
    clusters = []
    for cid in range(n_clusters):
        mask = all_labels == cid
        count = int(mask.sum())

        if count < min_cluster_size:
            logger.info(f"  簇 {cid}: {count:,} 高斯球 → 太小，丢弃")
            continue

        cpos = positions[mask]
        ccol = colors[mask]
        clusters.append({
            "cluster_id": cid,
            "count": count,
            "ratio": round(count / n, 4),
            "center_3d": cpos.mean(axis=0).tolist(),
            "bbox_3d": {
                "min": cpos.min(axis=0).tolist(),
                "max": cpos.max(axis=0).tolist(),
            },
            "dominant_color_rgb": ccol.mean(axis=0).tolist(),
            "indices": mask.nonzero()[0].tolist()[:1000],  # 只存前 1000 个索引，避免 JSON 膨胀
        })
        logger.info(
            f"  簇 {cid}: {count:,} 高斯球 ({100*count/n:.1f}%), "
            f"中心={cpos.mean(axis=0).round(3)}"
        )

    # 按大小降序
    clusters.sort(key=lambda c: c["count"], reverse=True)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "spectral_clustering",
        "n_clusters_requested": n_clusters,
        "n_clusters_found": len(clusters),
        "total_vertices": n,
        "clusters": clusters,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"发现 {len(clusters)} 个物体/区域 → {output_path}")

    # ---- 自动命名 ----
    try:
        clusters = _auto_label_clusters(clusters, positions)
        result["clusters"] = clusters
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"自动命名失败（CLIP 不可用？）: {e}")

    return clusters


# ================================================================
# 自动命名：CLIP 文字相似度匹配常见物体
# ================================================================

_COMMON_OBJECTS = [
    # 地面/结构
    "floor", "wall", "ceiling", "ground", "carpet", "rug", "door", "window",
    # 家具
    "chair", "table", "desk", "sofa", "bed", "bookshelf", "cabinet", "drawer", "nightstand",
    # 小物体
    "cup", "water cup", "bottle", "book", "laptop", "phone", "lamp", "plant", "vase",
    "bowl", "plate", "bag", "backpack", "pillow",
    # 电子
    "tv", "monitor", "keyboard", "mouse", "speaker",
    # 人物/动物
    "person", "cat", "dog",
    # 户外
    "tree", "grass", "road", "sky", "building", "car",
]


def _rgb_to_color_name(rgb: list[float]) -> str:
    """RGB → 中文颜色名。"""
    r, g, b = rgb
    # 灰度检测
    if max(r, g, b) - min(r, g, b) < 0.05:
        v = (r + g + b) / 3
        if v < 0.2: return "black"
        if v < 0.45: return "dark gray"
        if v < 0.7: return "gray"
        if v < 0.9: return "light gray"
        return "white"
    # 简单颜色判定
    if r > g and r > b:
        if g > 0.4: return "yellow" if r > 0.7 else "brown"
        return "red" if r - max(g, b) > 0.15 else "brown"
    if g > r and g > b:
        return "green"
    if b > r and b > g:
        return "blue"
    if r > 0.6 and g > 0.5: return "orange"
    return "brown"


def _auto_label_clusters(clusters: list[dict], positions: np.ndarray) -> list[dict]:
    """用 CLIP 文字相似度为每个聚类匹配最可能的物体名称。

    为每个聚类生成属性描述文字 → 与常见物体名称做 CLIP 文字匹配 → 选最高分。
    """
    from backend.recognition.clip_scorer import _encode_text, _load_clip

    _load_clip()
    scene_diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    z_all = positions[:, 2]
    z_min, z_max = float(z_all.min()), float(z_all.max())

    for c in clusters:
        rgb = c["dominant_color_rgb"]
        bmin = np.array(c["bbox_3d"]["min"])
        bmax = np.array(c["bbox_3d"]["max"])
        center = np.array(c["center_3d"])
        extent = bmax - bmin

        # 位置描述
        z_norm = (center[2] - z_min) / (z_max - z_min + 1e-8)
        if z_norm < 0.1: pos = "on the floor"
        elif z_norm < 0.35: pos = "near the ground"
        elif z_norm < 0.65: pos = "at mid height"
        elif z_norm < 0.9: pos = "high up"
        else: pos = "on the ceiling"

        # 大小描述
        diag = float(np.linalg.norm(extent))
        ratio = diag / (scene_diag + 1e-8)
        if ratio < 0.05: sz = "tiny"
        elif ratio < 0.15: sz = "small"
        elif ratio < 0.4: sz = "medium-sized"
        else: sz = "large"

        # 形状描述
        xy = float(np.linalg.norm(extent[:2]))
        z_ext = float(extent[2])
        if z_ext < xy * 0.15: shape = "flat"
        elif z_ext > xy * 2.5: shape = "tall and narrow"
        elif abs(xy - z_ext) < xy * 0.3: shape = "compact"
        else: shape = "irregular"

        # 颜色
        color_name = _rgb_to_color_name(rgb)

        # 生成描述 → CLIP 匹配
        desc_en = f"a {sz} {shape} {color_name} object {pos}"
        desc_cn = f"一个{sz}的{shape}的{color_name}物体{pos}"

        try:
            desc_emb = _encode_text(desc_en)

            best_obj = None
            best_sim = -999.0
            for obj in _COMMON_OBJECTS:
                obj_emb = _encode_text(obj)
                sim = float((desc_emb @ obj_emb.T).item())
                if sim > best_sim:
                    best_sim = sim
                    best_obj = obj

            # 置信度映射：CLIP text-text cosine 典型范围 [0.0, 0.45]
            confidence = round(max(0.1, min(1.0, best_sim / 0.45)), 3)
        except Exception:
            best_obj = "object"
            confidence = 0.0

        # 如果 CLIP 匹配结果是 generic 类别，附上颜色以区分不同区域
        generic_labels = {"floor", "wall", "ceiling", "ground", "carpet", "rug"}
        if best_obj in generic_labels:
            c["suggested_label"] = f"{color_name} {best_obj}"
        else:
            c["suggested_label"] = best_obj
        c["label_confidence"] = confidence

        logger.debug(
            f"  区域 {c['cluster_id']}: \"{desc_en}\" → {c['suggested_label']} ({confidence:.2f})"
        )

    return clusters
