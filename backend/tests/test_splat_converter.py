"""splat_converter 的字节级契约测试。

验证 PLY → .splat 转换的固定 32 字节/splat 布局，以及坐标/颜色/旋转
字段的解码正确性。这是前端 WebGL 直接消费的格式，契约不能破。
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from backend.zipsplat_engine.splat_converter import ply_to_splat

# 2 个顶点的 ASCII PLY。
# 顶点0：坐标(1,2,3) 尺度0 颜色DC0 不透明度0 旋转(1,0,0,0)
# 顶点1：坐标(4,5,6) 尺度log=0 颜色DC=0.5 不透明度0 旋转近似单位四元数
_PLY_TEXT = """\
ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
1 2 3 0 0 0 0 0 0 0 1 0 0 0
4 5 6 0 0 0 0.5 0.5 0.5 0 0.7071 0.7071 0 0
"""


@pytest.fixture
def ply_file(tmp_path) -> Path:
    p = tmp_path / "scene.ply"
    p.write_text(_PLY_TEXT, encoding="utf-8")
    return p


def test_splat_size_and_count(ply_file, tmp_path):
    splat_path = tmp_path / "scene.splat"
    result = ply_to_splat(ply_file, splat_path)
    assert result["splat_count"] == 2
    # 固定 32 字节 / splat
    assert splat_path.stat().st_size == 2 * 32
    assert result["splat_size"] == 2 * 32


def test_splat_byte_layout(ply_file, tmp_path):
    splat_path = tmp_path / "scene.splat"
    ply_to_splat(ply_file, splat_path)
    data = splat_path.read_bytes()

    # ---- 顶点0 ----
    # position (float32 ×3)
    x, y, z = struct.unpack("<fff", data[0:12])
    assert (x, y, z) == (1.0, 2.0, 3.0)
    # scale = exp(0) = 1.0。格式转换不得隐式修改模型几何。
    sx, sy, sz = struct.unpack("<fff", data[12:24])
    assert sx == pytest.approx(1.0, abs=1e-6)
    assert sy == pytest.approx(1.0, abs=1e-6)
    assert sz == pytest.approx(1.0, abs=1e-6)
    # color: dc=0 → 0*C0+0.5 = 0.5 → *255 → 127
    r, g, b, a = data[24:28]
    assert (r, g, b, a) == (127, 127, 127, 127)
    # rotation (1,0,0,0) 归一化后 (val+1)*127.5
    rw, rx, ry, rz = data[28:32]
    assert (rw, rx, ry, rz) == (255, 127, 127, 127)

    # ---- 顶点1：坐标 ----
    x1, y1, z1 = struct.unpack("<fff", data[32:44])
    assert (x1, y1, z1) == (4.0, 5.0, 6.0)
    # color: dc=0.5 → 0.5*C0+0.5 ≈ 0.641 → *255 ≈ 163
    r1, _, _, a1 = data[56:60]
    assert abs(r1 - 163) <= 2  # 容差：SH→RGB 近似与截断
    assert a1 == 127  # 不透明度 0 → sigmoid(0)=0.5 → 127
