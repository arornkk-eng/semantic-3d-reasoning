"""子进程执行的 ZipSplat 3D 重建任务。

由 backend/zipsplat_engine/runner.py 通过 subprocess 调用。
子进程继承父进程的 MSVC + CUDA 环境变量，因此 gsplat JIT 编译正常工作。
进程结束后 GPU 显存自动释放，保证任务间隔离。

用法:
    python worker_job.py --task-id <id> --input <dir> --output <dir>
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# ---- 添加 ZipSplat 到 sys.path ----
REPO_ROOT = Path(__file__).resolve().parents[2] / "ZipSplat-main"
sys.path.insert(0, str(REPO_ROOT))

from zipsplat import ZipSplat, load_image, viz


def main():
    parser = argparse.ArgumentParser(description="ZipSplat 3D 重建子进程")
    parser.add_argument("--task-id", required=True, help="任务 ID")
    parser.add_argument("--input", required=True, help="输入图片目录")
    parser.add_argument("--output", required=True, help="输出目录")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载图片 ----
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in exts],
        key=lambda p: p.name,
    )

    if not image_paths:
        print(json.dumps({"error": "输入目录中未找到图片文件"}), flush=True)
        sys.exit(1)

    print(f"[worker_job] 加载 {len(image_paths)} 张图片: {[p.name for p in image_paths]}", flush=True)

    # ---- 加载模型 ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[worker_job] 设备: {device}", flush=True)
    if device == "cuda":
        print(f"[worker_job] GPU: {torch.cuda.get_device_name(0)}, "
              f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

    model = ZipSplat(weights="zipsplat").to(device).eval()

    # ---- 重建高斯模型 ----
    images = [load_image(p) for p in image_paths]
    print(f"[worker_job] 正在重建...", flush=True)
    with torch.no_grad():
        gaussians = model(images, compression=1.0)[0]  # [0] 解批次

    num_gs = gaussians.num_gaussians
    print(f"[worker_job] 已生成 {num_gs:,} 个高斯球", flush=True)

    # ---- 导出 PLY ----
    ply_path = output_dir / "scene.ply"
    gaussians.save_ply(str(ply_path))
    ply_size = ply_path.stat().st_size
    print(f"[worker_job] PLY 已保存: {ply_path} ({ply_size / 1024:.1f} KB)", flush=True)

    # ---- 生成 360° 旋转视频 ----
    video_path = output_dir / "turntable_360.mp4"
    print(f"[worker_job] 正在渲染 360° 旋转视频...", flush=True)
    viz.turntable(
        gaussians,
        str(video_path),
        fov_deg=55.0,
        render_size=512,
        num_frames=360,
        fps=30,
        sweep_deg=360.0,
        elevation_deg=5.0,
    )
    video_size = video_path.stat().st_size
    print(f"[worker_job] 视频已保存: {video_path} ({video_size / 1024:.1f} KB)", flush=True)

    # ---- 清理 GPU 显存 ----
    del gaussians, model, images
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- 输出统计信息（JSON，最后一行被父进程解析） ----
    stats = {
        "num_gaussians": num_gs,
        "ply_size": ply_size,
        "video_size": video_size,
    }
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
