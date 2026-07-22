"""生成 360° 旋转展示视频。

基于已生成的高斯模型，渲染完整的 360° 旋转 turntable 视频。
"""

import sys
from pathlib import Path
import torch

# 添加 zipsplat 到路径
REPO_ROOT = Path(__file__).resolve().parents[2] / "ZipSplat-main"
sys.path.insert(0, str(REPO_ROOT))

from zipsplat import ZipSplat, viz, load_image


def main():
    demo_dir = Path(__file__).resolve().parents[1]
    input_dir = demo_dir / "image" / "input"
    output_dir = demo_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载图片 (支持 jpg / png / bmp / webp)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )
    print(f"加载 {len(image_paths)} 张图片: {[p.name for p in image_paths]}")

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ZipSplat(weights="zipsplat").to(device).eval()

    # 重建高斯模型
    images = [load_image(p) for p in image_paths]
    with torch.no_grad():
        gaussians = model(images, compression=1.0)[0]
    print(f"已生成 {gaussians.num_gaussians:,} 个高斯球")

    # 生成 360° 旋转视频
    video_path = output_dir / "turntable_360.mp4"
    print(f"正在渲染 360° 旋转视频 → {video_path} ...")
    viz.turntable(
        gaussians,
        str(video_path),
        fov_deg=55.0,
        render_size=512,
        num_frames=360,          # 360 帧 = 12 秒 @ 30fps，每帧 1°
        fps=30,
        sweep_deg=360.0,         # ← 完整 360° 旋转
        elevation_deg=5.0,       # 略微俯视
    )
    print(f"视频已保存: {video_path} ({video_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
