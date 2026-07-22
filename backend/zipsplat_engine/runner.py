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

    ply_path = output_dir / "scene.ply"
    gaussians.save_ply(str(ply_path))
    ply_size = ply_path.stat().st_size
    logger.info(f"PLY 已保存: {ply_path} ({ply_size / 1024:.1f} KB)")

    del model, gaussians, images
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "num_gaussians": num_gs,
        "ply_size": ply_size,
    }
