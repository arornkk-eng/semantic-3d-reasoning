"""相机位姿估算：重建后通过逆渲染恢复每帧的相机位姿。

原理：
  1. 用高斯球位置创建轨道初始位姿（围绕场景中心）
  2. 对每帧做位姿优化：渲染高斯球 → 对比输入帧 → 反向传播修正位姿
  3. 保存位姿供后续 2D→3D 特征投影使用
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


def estimate_poses(
    gaussians,          # zipsplat Gaussians 对象
    image_paths: list[Path],
    num_steps: int = 80,
    rot_lr: float = 0.003,
    trans_lr: float = 0.003,
    render_size: int = 252,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """为每帧估算相机位姿。

    Args:
        gaussians: ZipSplat Gaussians 对象（已 detach）
        image_paths: 输入帧路径列表
        num_steps: 优化步数（80 步约 3~5 秒/帧）
        rot_lr: 旋转学习率
        trans_lr: 平移学习率
        render_size: 渲染分辨率（ZipSplat 用 252×252）
        device: 计算设备

    Returns:
        (poses_4x4, Ks_3x3, image_sizes)
        poses_4x4: (N, 4, 4) world-to-cam 矩阵
        Ks_3x3: (N, 3, 3) 相机内参矩阵
        image_sizes: (N, 2) [w, h] 图像尺寸
    """
    from torchvision.io import read_image
    from torchvision.transforms import functional as VF
    import sys
    from pathlib import Path as _Path

    # 确保 zipsplat 在 path 中
    zipsplat_root = _Path(__file__).resolve().parents[2] / "ZipSplat-main"
    if str(zipsplat_root) not in sys.path:
        sys.path.insert(0, str(zipsplat_root))

    from zipsplat.pose import Pose
    from zipsplat.camera import Camera

    gaussians = gaussians.detach()
    n_frames = len(image_paths)

    # ---- Step 1: 计算场景中心和半径 ----
    means = gaussians.means.detach()
    center = means.median(dim=0).values
    radius = float((means - center).norm(dim=-1).quantile(0.9).item() * 1.2)

    logger.info(f"场景中心: {center.tolist()}, 半径: {radius:.2f}, 帧数: {n_frames}")

    # ---- Step 2: 创建初始轨道位姿 ----
    azimuths = torch.linspace(0, 360, n_frames + 1, device=device)[:n_frames]
    elevations = torch.full((n_frames,), 15.0, device=device)  # 略微俯视
    init_poses = Pose.orbit(center.to(device), radius, azimuths, elevations)

    # ---- Step 3: 推断相机内参 ----
    # 假设主点在图像中心，焦距 ≈ 0.8 × max(w,h)（典型手机）
    first_img = read_image(str(image_paths[0])).to(device)
    _, h, w = first_img.shape
    f_est = max(w, h) * 0.8
    camera = Camera.from_focal(f_est, f_est, w, h, w / 2, h / 2)
    logger.info(f"图像尺寸: {w}×{h}, 估计焦距: {f_est:.0f}")

    # ---- Step 4: 加载并预处理所有帧 ----
    target_rgbs = []
    actual_sizes = []
    for p in image_paths:
        img = read_image(str(p)).float().to(device) / 255.0  # (3, H, W) [0,1]
        _, ih, iw = img.shape
        actual_sizes.append((iw, ih))
        # Resize 到渲染分辨率
        img_252 = VF.resize(img, [render_size, render_size], antialias=True)
        target_rgbs.append(img_252)
    target_rgbs = torch.stack(target_rgbs, dim=0)  # (N, 3, 252, 252)

    # 调整 camera 到渲染分辨率
    scale_w = render_size / w
    scale_h = render_size / h
    render_camera = camera.scale((scale_w, scale_h))

    # ---- Step 5: 逐帧优化位姿 ----
    optimized_poses = []
    for i in range(n_frames):
        logger.info(f"优化位姿 {i+1}/{n_frames}...")
        pose = _optimize_single_pose(
            init_pose=init_poses[i],
            gaussians=gaussians,
            camera=render_camera,
            target_rgb=target_rgbs[i],
            num_steps=num_steps,
            rot_lr=rot_lr,
            trans_lr=trans_lr,
        )
        optimized_poses.append(pose)
        logger.debug(f"  帧 {i}: t={pose.t.tolist()}")

    # ---- Step 6: 构建输出 ----
    # 世界→相机矩阵（gsplat viewmats 格式）
    poses_c2w = Pose(torch.stack([p.data_ for p in optimized_poses]))
    poses_w2c = poses_c2w.inv().Rt  # (N, 4, 4)

    # 内参矩阵 K（原始分辨率）
    K = camera.K.cpu().numpy()  # (3, 3)

    return (
        poses_w2c.cpu().numpy(),
        np.tile(K[np.newaxis, ...], (n_frames, 1, 1)),
        np.array(actual_sizes, dtype=np.int32),
    )


def _optimize_single_pose(
    init_pose,
    gaussians,
    camera,
    target_rgb: torch.Tensor,  # (3, H, W)
    num_steps: int = 80,
    rot_lr: float = 0.003,
    trans_lr: float = 0.003,
) -> "Pose":
    """优化单个相机位姿使其渲染结果接近目标图像。"""
    from zipsplat.pose import Pose

    device = target_rgb.device
    target_rgb = target_rgb.unsqueeze(0)  # (1, 3, H, W)

    # 优化变量：旋转 + 平移增量（axis-angle 表示）
    rot_delta = nn.Parameter(torch.zeros(1, 3, device=device))
    trans_delta = nn.Parameter(torch.zeros(1, 3, device=device))

    optimizer = torch.optim.AdamW([
        {"params": [rot_delta], "lr": rot_lr},
        {"params": [trans_delta], "lr": trans_lr},
    ])

    pose = init_pose.clone()
    prev_loss = None
    patience = 0

    for step in range(num_steps):
        optimizer.zero_grad()

        # 从 axis-angle 构建姿态增量
        pose_update = Pose.from_aa(rot_delta, trans_delta)
        current_pose = pose @ pose_update

        # 渲染
        renderings, info = gaussians.render(
            cameras=camera[None], poses=current_pose[None], mode="RGB"
        )
        # renderings: (1, 1, 3, H, W)
        rendered = renderings[0, 0]  # (3, H, W)

        # 前景 mask（高斯覆盖区域）
        alpha = info["alphas"][0, 0]  # (1, H, W)
        mask = (alpha > 0.05).float()

        # L1 损失（仅前景区域）
        l1 = (F.l1_loss(rendered, target_rgb[0], reduction="none") * mask).sum()
        l1 = l1 / (mask.sum() * 3 + 1e-8)

        l1.backward()
        optimizer.step()

        # 更新位姿并重置增量
        with torch.no_grad():
            pose = pose @ Pose.from_aa(rot_delta, trans_delta)
            rot_delta.data.zero_()
            trans_delta.data.zero_()

        # 早停
        loss_val = l1.item()
        if prev_loss is not None:
            if abs(loss_val - prev_loss) < 1e-5:
                patience += 1
                if patience >= 5:
                    break
            else:
                patience = 0
        prev_loss = loss_val

        if step % 20 == 0:
            logger.debug(f"    step {step}: loss={loss_val:.4f}")

    return pose.detach()


# Monkey-patch: Pose.from_aa 方法（axis-angle → Rotation matrix）
def _from_aa(aa: torch.Tensor, t: torch.Tensor) -> "Pose":
    """从 axis-angle 旋转 + 平移构建 Pose。"""
    from zipsplat.pose import Pose

    angle = aa.norm(dim=-1, keepdim=True)
    axis = aa / (angle + 1e-8)
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    one_minus_cos = 1 - cos_a

    R = torch.stack([
        cos_a + x*x*one_minus_cos, x*y*one_minus_cos - z*sin_a, x*z*one_minus_cos + y*sin_a,
        y*x*one_minus_cos + z*sin_a, cos_a + y*y*one_minus_cos, y*z*one_minus_cos - x*sin_a,
        z*x*one_minus_cos - y*sin_a, z*y*one_minus_cos + x*sin_a, cos_a + z*z*one_minus_cos,
    ], dim=-1).reshape(*aa.shape[:-1], 3, 3)

    return Pose.from_Rt(R, t)


import types as _types
import zipsplat.pose as _pose_module
_pose_module.Pose.from_aa = staticmethod(_from_aa)
