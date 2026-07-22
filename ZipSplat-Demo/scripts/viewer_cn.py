"""中文本地化 ZipSplat 3D 交互查看器。

基于官方 zipsplat.viewer，将界面元素翻译为中文。
不修改官方源码，仅做 UI 层汉化包装。

使用方式:
    python scripts/viewer_cn.py --input image/input/ --port 8080
"""

import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import viser
from nerfview import Viewer
from zipsplat.camera import Camera
from zipsplat.pose import Pose
from zipsplat.predictor import ZipSplat
from zipsplat.utils import load_image, load_video

_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run(model: ZipSplat, images: List[torch.Tensor], port: int = 8080) -> None:
    """启动中文本地化 Viser 查看器。"""
    group_size = model.model.conf["gaussians_per_token"]
    state = {"gaussians": None, "colored": None, "by_token": False}

    def recompute(compression: float) -> None:
        g = model(images, compression=compression)[0]
        state["gaussians"], state["colored"] = g, g.color_by_group(group_size)

    recompute(1.0)

    server = viser.ViserServer(port=port, verbose=False)

    # --- 汉化 UI 控件 ---
    gs_count = server.gui.add_number("高斯球数量", 0, disabled=True)
    compression = server.gui.add_slider(
        "压缩比例", min=0.05, max=1.0, step=0.05, initial_value=1.0
    )
    color_toggle = server.gui.add_checkbox("按 Token 着色", initial_value=False)

    @torch.no_grad()
    def render_fn(camera_state, render_tab_state):
        w, h = render_tab_state.viewer_width, render_tab_state.viewer_height
        K = torch.from_numpy(camera_state.get_K((w, h))).float()
        c2w = torch.from_numpy(camera_state.c2w).float()
        scene = state["colored"] if state["by_token"] else state["gaussians"]
        rgb, _ = scene.render(Camera.from_K(K, w=w, h=h), Pose.from_4x4mat(c2w), mode="RGB")
        gs_count.value = scene.num_gaussians
        return rgb[0].clamp(0, 1).moveaxis(0, -1).cpu().numpy()

    viewer = Viewer(server, render_fn, output_dir=None, mode="rendering")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        client.camera.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        client.camera.position = np.array([0.0, 0.0, 0.0])

    @compression.on_update
    def _(_):
        recompute(compression.value)
        viewer.rerender(None)

    @color_toggle.on_update
    def _(_):
        state["by_token"] = color_toggle.value
        viewer.rerender(None)

    print(f"查看器已启动: http://localhost:{port} — Ctrl+C 退出")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


def _load_inputs(path: str, num_frames: int) -> List[torch.Tensor]:
    """加载图片目录、glob 匹配或视频文件。"""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES:
        return load_video(p, num_frames=num_frames)
    files = (
        sorted(f for f in p.iterdir() if p.is_dir() and f.suffix.lower() in _IMAGE_SUFFIXES)
        if p.is_dir()
        else sorted(Path().glob(path))
    )
    if not files:
        raise ValueError(f"在 {path!r} 中未找到图片或视频。")
    return [load_image(f) for f in files]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="中文本地化 ZipSplat 3D 交互查看器")
    ap.add_argument("input", help="图片目录、glob 匹配或视频文件路径")
    ap.add_argument("--weights", default="zipsplat", help="模型权重（默认: zipsplat）")
    ap.add_argument("--num-frames", type=int, default=24, help="从视频中采样的帧数")
    ap.add_argument("--port", type=int, default=8080, help="Web 服务端口（默认: 8080）")
    args = ap.parse_args()

    images = _load_inputs(args.input, args.num_frames)
    print(f"已加载 {len(images)} 张图片。")
    model = ZipSplat(weights=args.weights).cuda().eval()
    run(model, images, port=args.port)


if __name__ == "__main__":
    main()
