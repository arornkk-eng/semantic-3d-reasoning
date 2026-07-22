"""
从多角度拍摄的物体照片重建 3D 模型 (Gaussian Splatting)

使用方法:
    1. 将物体的多角度照片放入 image/input/ 文件夹
    2. 运行: python scripts/reconstruct.py
    3. 输出:
       - output/scene.ply         → 3D 高斯点云（可用 SuperSplat 等查看器打开）
       - output/turntable.mp4     → 360° 旋转预览视频

支持的图片格式: jpg, jpeg, png, bmp, webp
建议: 5-24 张不同角度的照片，覆盖物体的各个面，效果最佳
"""

import argparse
import math
from pathlib import Path
import sys

import torch

# 将 zipsplat 包加入路径
REPO_ROOT = Path(__file__).resolve().parents[2] / "ZipSplat-main"
sys.path.insert(0, str(REPO_ROOT))

from zipsplat import ZipSplat, Camera, Pose, load_image, viz


def collect_images(input_dir: Path) -> list[Path]:
    """收集 input_dir 下所有图片文件，按文件名排序。"""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in extensions],
        key=lambda p: p.name,
    )
    if not images:
        raise FileNotFoundError(f"在 {input_dir} 中没有找到图片文件（支持: {extensions}）")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="ZipSplat: 从多角度照片重建物体 3D 模型"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="包含多角度照片的文件夹（默认: image/input/）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件夹（默认: output/）",
    )
    parser.add_argument(
        "--compression",
        type=float,
        default=1.0,
        help="压缩比例 (0-1]，越小高斯球越少、文件越小但质量越低（默认: 1.0 最高质量）",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="zipsplat",
        help="模型权重: zipsplat (默认) / 本地路径 / URL",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成旋转预览视频",
    )
    args = parser.parse_args()

    # 确定路径
    demo_dir = Path(__file__).resolve().parents[1]
    input_dir = (args.input or demo_dir / "image" / "input").resolve()
    output_dir = (args.output or demo_dir / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 收集图片 ----------
    print(f"[1/5] Scanning input: {input_dir}")
    image_paths = collect_images(input_dir)
    print(f"      Found {len(image_paths)} photos:")
    for p in image_paths:
        print(f"       - {p.name}")

    # ---------- 2. Load model ----------
    print(f"\n[2/5] Loading model weights: {args.weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("      WARNING: No CUDA GPU detected, using CPU (slow)")

    model = ZipSplat(weights=args.weights).to(device).eval()
    print("      Model loaded OK")

    # ---------- 3. Reconstruct ----------
    print(f"\n[3/5] Reconstructing 3D scene (compression={args.compression})...")
    images = [load_image(p) for p in image_paths]

    with torch.no_grad():
        gaussians = model(images, compression=args.compression)[0]

    num_gaussians = gaussians.num_gaussians
    print(f"      Done! Generated {num_gaussians:,} Gaussians")

    # ---------- 4. Export PLY ----------
    ply_path = output_dir / "scene.ply"
    print(f"\n[4/5] Exporting PLY: {ply_path}")
    gaussians.save_ply(str(ply_path))
    print(f"      File size: {ply_path.stat().st_size / 1024:.1f} KB")

    # ---------- 5. Turntable video ----------
    if not args.no_video:
        video_path = output_dir / "turntable.mp4"
        print(f"\n[5/5] Generating turntable preview video: {video_path}")
        viz.turntable(gaussians, str(video_path), sweep_deg=None)
        print(f"      Video saved")

    # ---------- 6. Done ----------
    print(f"\n{'='*50}")
    print(f"ALL DONE! Output files:")
    print(f"   3D Model: {ply_path}")
    if not args.no_video:
        print(f"   Preview:  {video_path}")
    print(f"\nTip: Open PLY with SuperSplat: https://playcanvas.com/supersplat/editor")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
