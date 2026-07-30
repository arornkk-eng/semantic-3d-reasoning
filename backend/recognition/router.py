"""物体识别 API 端点。

三合一语义识别系统：
  1. POST /recognize  — 文字查询 → Grounding DINO + CLIP + SAM → 3D 标签
  2. POST /discover   — 无查询 → 3D 自监督聚类 → 发现场景中的物体
  3. GET  /labels     — 获取已保存的标签
  4. GET  /clusters   — 获取已发现的物体簇
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.core.config import OUTPUT_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)
router = APIRouter()


# ================================================================
# POST /recognize — 文字查询驱动识别
# ================================================================

@router.post("/task/{task_id}/recognize")
async def recognize_objects(task_id: str, body: dict):
    """对已完成的重建任务执行语义物体识别。

    Body:
        {
            "objects": ["水杯", "桌子"],
            "box_threshold": 0.25,    # Grounding DINO 置信度阈值
            "use_clip": true,          # 启用 CLIP 语义验证（推荐）
            "use_sam": true            # 启用 SAM 精确分割 mask（推荐）
        }

    Returns:
        {"task_id": ..., "status": "completed", "labels": {"水杯": {"count": N, "score": ...}}}
    """
    # 校验任务状态
    from backend.storage.file_manager import get_task_meta

    meta = get_task_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    if meta["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成（当前状态: {meta['status']}），请等待重建完成后再识别",
        )

    objects = body.get("objects", [])
    if not objects:
        raise HTTPException(status_code=400, detail="请至少指定一个物体名称，如 {\"objects\": [\"水杯\"]}")

    box_threshold = float(body.get("box_threshold", 0.15))
    z_threshold = float(body.get("z_threshold", 1.5))
    use_clip = bool(body.get("use_clip", True))
    use_sam = bool(body.get("use_sam", True))

    ply_path = OUTPUT_DIR / task_id / "scene.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail=f"PLY 文件不存在: {ply_path}")

    # 获取输入帧
    input_dir = UPLOAD_DIR / task_id
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise HTTPException(status_code=400, detail="未找到输入帧，请先完成重建")

    logger.info(
        f"开始识别: task={task_id}, objects={objects}, "
        f"frames={len(image_paths)}, CLIP={use_clip}, SAM={use_sam}"
    )

    import cv2

    def _run():
        from backend.recognition.detector import detect_objects
        from backend.recognition.labeler import label_gaussians
        from backend.recognition.semantic_labeler import semantic_label_gaussians
        from backend.recognition.translator import translate_query
        from backend.recognition.pose_estimation import estimate_poses
        from backend.recognition.feature_lifter import (
            load_gaussians_from_params, project_features_to_gaussians,
        )

        # ---- Step 0: 加载高斯球参数 ----
        params_path = OUTPUT_DIR / task_id / "gaussians.pt"
        if not params_path.exists():
            logger.warning("无 gaussians.pt，回退到颜色匹配模式")
            return _run_color_mode(objects, image_paths, ply_path, task_id,
                                   box_threshold, z_threshold, use_clip, use_sam)

        gaussians = load_gaussians_from_params(params_path)
        logger.info(f"已加载 {gaussians.num_gaussians:,} 个高斯球")

        # ---- Step 1: 位姿估算 ----
        poses_w2c, Ks, image_sizes = estimate_poses(
            gaussians, image_paths, num_steps=80
        )
        logger.info(f"位姿估算完成: {len(poses_w2c)} 帧")

        # ---- Step 2: 检测 + 特征投影 ----
        # 加载所有帧为 BGR
        images_bgr = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                raise ValueError(f"无法读取图片: {p}")
            images_bgr.append(img)

        all_detections = {}
        all_labels = {}
        for query in objects:
            en_candidates = translate_query(query)
            logger.info(f"查询 '{query}' → 候选 {en_candidates}")

            best_labels = {}
            best_count = 0
            for en_query in en_candidates[:3]:
                detections = detect_objects(
                    image_paths,
                    text_query=en_query,
                    box_threshold=box_threshold,
                    use_clip=use_clip,
                    use_sam=use_sam,
                )
                if not detections:
                    continue

                # 提取每帧检测列表（匹配 feature_lifter 格式）
                det_per_frame = _group_detections_by_frame(detections, len(image_paths))

                # CLIP 特征投影
                projected = project_features_to_gaussians(
                    gaussians, poses_w2c, Ks, image_sizes,
                    det_per_frame, images_bgr,
                )

                if not projected:
                    continue

                # 语义标签（用 CLIP 特征替代 HSV 颜色）
                labels_path = OUTPUT_DIR / task_id / "labels.json"
                labels = semantic_label_gaussians(
                    projected, detections, labels_path,
                    query_label=query, z_threshold=z_threshold,
                )
                if labels:
                    total = sum(v["count"] for v in labels.values())
                    if total > best_count:
                        best_count = total
                        best_labels = labels

            if best_labels:
                for en_key, val in best_labels.items():
                    all_labels[query] = val
                    break

        del gaussians
        torch_cleanup()

        final_path = OUTPUT_DIR / task_id / "labels.json"
        final_path.write_text(
            json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return all_labels


    def _run_color_mode(objects, image_paths, ply_path, task_id,
                         box_threshold, z_threshold, use_clip, use_sam):
        """回退模式：无 gaussians.pt 时用原来的颜色匹配。"""
        from backend.recognition.detector import detect_objects
        from backend.recognition.labeler import label_gaussians
        from backend.recognition.translator import translate_query

        all_labels = {}
        for query in objects:
            en_candidates = translate_query(query)
            best_labels, best_count = {}, 0
            for en_query in en_candidates[:3]:
                detections = detect_objects(image_paths, text_query=en_query,
                    box_threshold=box_threshold, use_clip=use_clip, use_sam=use_sam)
                labels_path = OUTPUT_DIR / task_id / "labels.json"
                labels = label_gaussians(ply_path, detections, labels_path,
                    query_label=query, z_threshold=z_threshold)
                if labels:
                    total = sum(v["count"] for v in labels.values())
                    if total > best_count:
                        best_count, best_labels = total, labels
            if best_labels:
                for en_key, val in best_labels.items():
                    all_labels[query] = val
                    break
        final_path = OUTPUT_DIR / task_id / "labels.json"
        final_path.write_text(json.dumps(all_labels, ensure_ascii=False, indent=2), encoding="utf-8")
        return all_labels


    def _group_detections_by_frame(detections: list[dict], n_frames: int) -> list[list[dict]]:
        """将检测结果按帧分组。"""
        groups = [[] for _ in range(n_frames)]
        for det in detections:
            idx = det.get("index", 0)
            if 0 <= idx < n_frames:
                groups[idx].extend(det.get("detections", []))
        return groups


    def torch_cleanup():
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.error(f"识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

    # 读取结果
    labels_path = OUTPUT_DIR / task_id / "labels.json"
    labels = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))

    return {
        "task_id": task_id,
        "status": "completed",
        "labels": {
            k: {"count": v["count"], "score": v.get("score", 0)}
            for k, v in labels.items()
        },
    }


# ================================================================
# POST /discover — 无查询自监督发现
# ================================================================

@router.post("/task/{task_id}/discover")
async def discover_objects(task_id: str, body: dict | None = None):
    """不依赖文字查询，直接在 3D 高斯球上做无监督聚类，自动发现场景中的物体/区域。

    Body (可选):
        {
            "n_clusters": 5,          # 期望发现的物体数量
            "n_samples": 5000,         # 采样点数（越大越精确但也越慢）
            "pos_weight": 1.0,         # 空间邻近度权重
            "col_weight": 0.5,         # 颜色相似度权重
            "min_cluster_size": 100    # 最小簇大小
        }

    Returns:
        {"task_id": ..., "clusters": [{cluster_id, count, center_3d, dominant_color_rgb}]}
    """
    from backend.storage.file_manager import get_task_meta

    meta = get_task_meta(task_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    if meta["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成（当前状态: {meta['status']}），请等待重建完成后再分析",
        )

    ply_path = OUTPUT_DIR / task_id / "scene.ply"
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail=f"PLY 文件不存在: {ply_path}")

    if body is None:
        body = {}

    n_clusters = int(body.get("n_clusters", 5))
    n_samples = int(body.get("n_samples", 5000))
    pos_weight = float(body.get("pos_weight", 1.0))
    col_weight = float(body.get("col_weight", 0.5))
    min_cluster_size = int(body.get("min_cluster_size", 100))

    logger.info(
        f"开始 3D 语义发现: task={task_id}, n_clusters={n_clusters}, "
        f"n_samples={n_samples}"
    )

    def _run():
        from backend.recognition.cluster_3d import discover_objects as do
        output_path = OUTPUT_DIR / task_id / "clusters.json"
        return do(
            ply_path=ply_path,
            output_path=output_path,
            n_clusters=n_clusters,
            n_samples=n_samples,
            pos_weight=pos_weight,
            col_weight=col_weight,
            min_cluster_size=min_cluster_size,
        )

    try:
        loop = asyncio.get_running_loop()
        clusters = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.error(f"3D 语义发现失败: {e}")
        raise HTTPException(status_code=500, detail=f"语义发现失败: {str(e)}")

    return {
        "task_id": task_id,
        "status": "completed",
        "n_clusters_found": len(clusters),
        "clusters": [
            {
                "cluster_id": c["cluster_id"],
                "count": c["count"],
                "ratio": c.get("ratio", 0),
                "center_3d": c["center_3d"],
                "bbox_3d": c["bbox_3d"],
                "dominant_color_rgb": c["dominant_color_rgb"],
                "suggested_label": c.get("suggested_label", "object"),
                "label_confidence": c.get("label_confidence", 0),
            }
            for c in clusters
        ],
    }


# ================================================================
# GET /export — 导出识别物体的高斯球为独立 PLY
# ================================================================

@router.get("/task/{task_id}/export/{object_name}")
async def export_object_ply(task_id: str, object_name: str):
    """导出识别物体的高斯球为独立 PLY 文件。"""
    import json

    labels_path = OUTPUT_DIR / task_id / "labels.json"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="请先执行识别")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if object_name not in labels:
        raise HTTPException(status_code=404, detail=f"未找到 '{object_name}'，已识别: {list(labels.keys())}")

    indices = labels[object_name].get("indices", [])
    if not indices:
        raise HTTPException(status_code=400, detail="无索引数据")

    ply_path = OUTPUT_DIR / task_id / "scene.ply"
    export_path = OUTPUT_DIR / task_id / f"{object_name}.ply"

    import asyncio

    def _run():
        from backend.recognition.ply_reader import read_ply, extract_vertices
        data = read_ply(ply_path)
        extract_vertices(data, indices, export_path)
        return export_path

    try:
        loop = asyncio.get_running_loop()
        result_path = await loop.run_in_executor(None, _run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {str(e)}")

    from fastapi.responses import FileResponse
    return FileResponse(
        path=str(result_path),
        filename=f"{object_name}.ply",
        media_type="application/octet-stream",
    )


# ================================================================
# GET /indices — 获取识别物体的高斯球索引
# ================================================================

@router.get("/task/{task_id}/indices/{object_name}")
async def get_object_indices(task_id: str, object_name: str):
    """返回识别物体的高斯球索引列表（供编辑器选中 API 使用）。"""
    import json

    labels_path = OUTPUT_DIR / task_id / "labels.json"
    if not labels_path.exists():
        raise HTTPException(status_code=404, detail="请先执行识别")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if object_name not in labels:
        raise HTTPException(status_code=404, detail=f"未找到 '{object_name}'")

    return {
        "object_name": object_name,
        "indices": labels[object_name].get("indices", []),
    }


# ================================================================
# GET /labels & GET /clusters — 结果查询
# ================================================================

@router.get("/task/{task_id}/labels")
async def get_labels(task_id: str):
    """获取文字查询识别的物体标签（含 3D 位置和聚类信息）。"""
    labels_path = OUTPUT_DIR / task_id / "labels.json"
    if not labels_path.exists():
        return {"task_id": task_id, "objects": {}}

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return {
        "task_id": task_id,
        "objects": {
            k: {
                "count": v["count"],
                "score": v.get("score", 0),
                "center_3d": v.get("center_3d"),
                "bbox_3d": v.get("bbox_3d"),
            }
            for k, v in labels.items()
        },
    }


@router.get("/task/{task_id}/clusters")
async def get_clusters(task_id: str):
    """获取自监督发现的物体簇（含 indices 用于高亮）。"""
    clusters_path = OUTPUT_DIR / task_id / "clusters.json"
    if not clusters_path.exists():
        return {"task_id": task_id, "clusters": []}

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    return {
        "task_id": task_id,
        "method": clusters.get("method"),
        "n_clusters_found": clusters.get("n_clusters_found", 0),
        "total_vertices": clusters.get("total_vertices", 0),
        "clusters": clusters.get("clusters", []),
    }
