"""视频帧智能提取：逐帧采样 → 质量评分 → 视角多样性筛选。

对多段 MP4 视频逐帧采样，按清晰度+曝光+纹理综合评分，
最后用 ORB 特征匹配做贪心视角多样性筛选。
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ORB 特征检测器（全局单例延迟初始化）
_orb = None


def _get_orb():
    global _orb
    if _orb is None:
        _orb = cv2.ORB.create(nfeatures=500, scaleFactor=1.2, nlevels=8)
    return _orb


# ================================================================
# 公开 API
# ================================================================

def extract_and_select(
    video_paths: list[Path],
    output_dir: Path,
    max_frames: int = 25,
    sample_interval: float = 1.0,
    quality_threshold: float = 0.25,
) -> dict:
    """从多段视频提取帧 → 质量筛选 → 视角多样性选择 → 保存 JPG。

    Args:
        video_paths: 视频文件路径列表
        output_dir: 选中帧的输出目录（JPG）
        max_frames: 最终保留的最大帧数
        sample_interval: 采样间隔（秒）
        quality_threshold: 质量分阈值，低于此值的帧丢弃

    Returns:
        { total_extracted, quality_passed, selected, frame_paths }
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: 从全部视频提取帧 ----
    all_frames: list[dict] = []  # [{image, timestamp, video_idx, quality_score}]
    total_extracted = 0

    for vid_idx, vp in enumerate(video_paths):
        if not vp.exists():
            logger.warning(f"视频不存在，跳过: {vp}")
            continue
        frames = _extract_frames(vp, vid_idx, sample_interval)
        total_extracted += len(frames)
        logger.info(f"视频 {vid_idx+1}/{len(video_paths)} ({vp.name}): 提取 {len(frames)} 帧")

        # ---- Step 2: 质量评分 ----
        for f in frames:
            score = _quality_score(f["image"])
            if score >= quality_threshold:
                f["quality"] = score
                all_frames.append(f)

    quality_passed = len(all_frames)
    logger.info(
        f"质量筛选: {total_extracted} → {quality_passed} "
        f"(阈值={quality_threshold}, 丢弃 {total_extracted - quality_passed})"
    )

    if not all_frames:
        logger.warning("无帧通过质量筛选，降低阈值重试")
        return {"total_extracted": total_extracted, "quality_passed": 0,
                "selected": 0, "frame_paths": []}

    # ---- Step 3: 视角多样性筛选 ----
    if len(all_frames) <= max_frames:
        selected = all_frames
        logger.info(f"候选帧 ≤ {max_frames}，全部保留")
    else:
        selected = _diversity_select(all_frames, max_frames)
        logger.info(f"多样性筛选: {quality_passed} → {len(selected)}")

    # ---- Step 4: 保存选中帧 (imencode 兼容非 ASCII 路径) ----
    saved = []
    for i, f in enumerate(selected):
        filename = f"frame_{i+1:04d}_v{f['video_idx']}_t{f['timestamp']:.1f}s.jpg"
        out_path = output_dir / filename
        # imencode + tofile 比 imwrite 更好地处理 Windows Unicode 路径
        success, encoded = cv2.imencode(".jpg", f["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        if success:
            encoded.tofile(str(out_path))
            saved.append(str(out_path))

    logger.info(f"保存 {len(saved)} 帧到 {output_dir}")

    return {
        "total_extracted": total_extracted,
        "quality_passed": quality_passed,
        "selected": len(selected),
        "frame_paths": saved,
    }


# ================================================================
# 内部实现
# ================================================================

def _extract_frames(video_path: Path, video_idx: int, interval_sec: float) -> list[dict]:
    """从单个视频按间隔提取帧，返回 [{image, timestamp, video_idx}]。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"无法打开视频: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # 兜底

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = max(1, int(fps * interval_sec))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            frames.append({
                "image": frame,
                "timestamp": round(timestamp, 1),
                "video_idx": video_idx,
            })

        frame_idx += 1

    cap.release()
    logger.debug(f"  提取 {len(frames)}/{total_frames} 帧 @ {fps:.1f}fps, "
                 f"间隔 {frame_interval} 帧 ({interval_sec}s), 时长 {duration:.1f}s")
    return frames


def _quality_score(image: np.ndarray) -> float:
    """综合质量评分 (0-1)：清晰度 40% + 曝光 30% + 纹理 30%。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- 清晰度：Laplacian 方差 ----
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()
    # 经验归一化：Laplacian 方差 0-500 映射到 0-1，超过 500 = 满分
    sharp_score = min(sharpness / 500.0, 1.0)

    # ---- 曝光：亮度直方图均衡度 ----
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化
    # 避免过曝 (>240) 和欠曝 (<15) 的像素占比
    overexposed = hist[240:].sum()
    underexposed = hist[:15].sum()
    # 中间调占比越高越好
    midtones = hist[60:200].sum()
    exposure_score = midtones * (1.0 - overexposed) * (1.0 - underexposed)
    exposure_score = max(0.0, min(1.0, exposure_score))

    # ---- 纹理丰富度：ORB 特征点数量 ----
    orb = _get_orb()
    kp = orb.detect(gray, None)
    # 经验归一化：100 个特征点 = 满分
    texture_score = min(len(kp) / 100.0, 1.0)

    # 综合
    score = 0.4 * sharp_score + 0.3 * exposure_score + 0.3 * texture_score
    return float(score)


def _diversity_select(frames: list[dict], max_frames: int) -> list[dict]:
    """按视频均衡分配 + 时间均匀采样 + 质量优先。

    1. 统计视频数，每段视频分配 frames_per_video = max_frames / n_videos
    2. 每段视频内按时序分段，每段选质量最高的帧
    3. 确保所有视角均匀覆盖
    """
    n_videos = len(set(f["video_idx"] for f in frames))
    frames_per_video = max(1, max_frames // n_videos)

    # 按视频分组
    by_video: dict[int, list[dict]] = {}
    for f in frames:
        by_video.setdefault(f["video_idx"], []).append(f)

    selected = []
    for vid_idx, vid_frames in by_video.items():
        # 按时序排序
        vid_frames = sorted(vid_frames, key=lambda f: f["timestamp"])

        if len(vid_frames) <= frames_per_video:
            # 帧数不足，全部保留
            selected.extend(vid_frames)
        else:
            # 将时间轴均匀分成 frames_per_video 段，每段选质量最高帧
            n_seg = frames_per_video
            seg_size = len(vid_frames) / n_seg
            for seg_i in range(n_seg):
                start = int(seg_i * seg_size)
                end = int((seg_i + 1) * seg_size)
                if start >= len(vid_frames):
                    break
                seg_frames = vid_frames[start:max(start + 1, end)]
                # 段内选质量最高
                best = max(seg_frames, key=lambda f: f.get("quality", 0))
                selected.append(best)

        logger.info(f"  视频 {vid_idx}: {len(vid_frames)} → {min(len(vid_frames), frames_per_video)} 帧")

    logger.info(f"均衡选择完成: {len(frames)} → {len(selected)} 帧 ({n_videos} 段视频 × ~{frames_per_video})")
    return selected
