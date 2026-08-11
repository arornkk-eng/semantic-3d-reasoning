"""PLY → .splat 格式转换器。

.splat 是 Web GL 原生高斯点云格式，每 splat 固定 32 字节：
    float32[3]  position (xyz)
    float32[3]  scale (xyz)
    uint8[4]    color (RGBA)
    uint8[4]    rotation (wxyz quaternion, 编码为 (val+1)*127.5)

比 PLY 小 3-5 倍，且在浏览器中直接由 GPU 读取，无需解析。
"""

import struct
from pathlib import Path

import numpy as np

from backend.core.config import SPLAT_SCALE_FACTOR

_SH_C0 = 0.28209479177387814


def ply_to_splat(ply_path: Path, splat_path: Path) -> dict:
    """将标准 3DGS PLY 文件转换为 .splat 格式。

    Returns:
        dict: {"splat_count": int, "ply_size": int, "splat_size": int}
    """
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    vert = ply["vertex"]
    n = len(vert)

    # 1. 位置 (直接使用)
    x = vert["x"].astype(np.float32)
    y = vert["y"].astype(np.float32)
    z = vert["z"].astype(np.float32)

    # 2. 尺度。PLY 中为 log-scale，转换只做 exp，不改变模型几何。
    scale_x = np.exp(vert["scale_0"].astype(np.float32)) * SPLAT_SCALE_FACTOR
    scale_y = np.exp(vert["scale_1"].astype(np.float32)) * SPLAT_SCALE_FACTOR
    scale_z = np.exp(vert["scale_2"].astype(np.float32)) * SPLAT_SCALE_FACTOR

    # 3. 颜色 — SH DC → RGB
    sh0_r = vert["f_dc_0"].astype(np.float32)
    sh0_g = vert["f_dc_1"].astype(np.float32)
    sh0_b = vert["f_dc_2"].astype(np.float32)
    r = np.clip(sh0_r * _SH_C0 + 0.5, 0.0, 1.0)
    g = np.clip(sh0_g * _SH_C0 + 0.5, 0.0, 1.0)
    b = np.clip(sh0_b * _SH_C0 + 0.5, 0.0, 1.0)

    # 4. 不透明度 — logit → sigmoid
    opacity = vert["opacity"].astype(np.float32)
    alpha = 1.0 / (1.0 + np.exp(-opacity))

    # 5. 旋转四元数 (wxyz 顺序)
    rot_w = vert["rot_0"].astype(np.float32)
    rot_x = vert["rot_1"].astype(np.float32)
    rot_y = vert["rot_2"].astype(np.float32)
    rot_z = vert["rot_3"].astype(np.float32)

    # 归一化四元数
    rot_norm = np.sqrt(rot_w**2 + rot_x**2 + rot_y**2 + rot_z**2)
    rot_norm = np.where(rot_norm > 0, rot_norm, 1.0)
    rot_w /= rot_norm
    rot_x /= rot_norm
    rot_y /= rot_norm
    rot_z /= rot_norm

    # 写入 .splat 文件
    with open(splat_path, "wb") as f:
        for i in range(n):
            # position (float32 × 3)
            f.write(struct.pack("<fff", x[i], y[i], z[i]))
            # scale (float32 × 3)
            f.write(struct.pack("<fff", scale_x[i], scale_y[i], scale_z[i]))
            # color (uint8 × 4, RGBA)
            f.write(
                struct.pack(
                    "<BBBB",
                    int(np.clip(r[i] * 255, 0, 255)),
                    int(np.clip(g[i] * 255, 0, 255)),
                    int(np.clip(b[i] * 255, 0, 255)),
                    int(np.clip(alpha[i] * 255, 0, 255)),
                )
            )
            # rotation (uint8 × 4, wxyz, 编码 (val+1)*127.5)
            f.write(
                struct.pack(
                    "<BBBB",
                    int(np.clip((rot_w[i] + 1.0) * 127.5, 0, 255)),
                    int(np.clip((rot_x[i] + 1.0) * 127.5, 0, 255)),
                    int(np.clip((rot_y[i] + 1.0) * 127.5, 0, 255)),
                    int(np.clip((rot_z[i] + 1.0) * 127.5, 0, 255)),
                )
            )

    ply_size = ply_path.stat().st_size
    splat_size = splat_path.stat().st_size

    return {
        "splat_count": n,
        "ply_size": ply_size,
        "splat_size": splat_size,
    }
