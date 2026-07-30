"""健壮的 PLY 读写：解析 header 获取字段名/类型/偏移，支持 ASCII 和 binary。

ZipSplat PLY 格式 (binary_little_endian):
    x y z | nx ny nz | f_dc_0 f_dc_1 f_dc_2 | f_rest_0..8 | opacity | scale_0..2 | rot_0..3
    共 26 个 float32 = 104 bytes/vertex

颜色从 f_dc_0/1/2 (SH DC 系数) 转换: rgb = SH_C0 * f_dc + 0.5
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814


# ================================================================
# PLY 读取
# ================================================================

class PlyData:
    """PLY 数据容器：header 元信息 + vertex NumPy 数组 (N, num_props)。"""

    def __init__(self):
        self.format: str = ""          # "ascii" | "binary_little_endian" | "binary_big_endian"
        self.num_vertices: int = 0
        self.props: list[dict] = []    # [{name, dtype}, ...]
        self.header_text: str = ""
        self.vertices: Optional[np.ndarray] = None  # (N, num_props) float32
        self._prop_index: dict[str, int] = {}       # name → column index

    def col(self, name: str) -> np.ndarray:
        """按属性名获取列数据 (N,)。"""
        if name not in self._prop_index:
            available = list(self._prop_index.keys())
            raise KeyError(f"PLY 中无属性 '{name}'，可用: {available}")
        return self.vertices[:, self._prop_index[name]]

    @property
    def positions(self) -> np.ndarray:
        return self.vertices[:, [self._prop_index["x"],
                                  self._prop_index["y"],
                                  self._prop_index["z"]]]

    @property
    def colors_rgb(self) -> np.ndarray:
        """获取 RGB 颜色 (N,3) 范围 [0,1]。

        优先从 f_dc_0/1/2 (SH DC) 转换；若不存在则回退到 red/green/blue。
        """
        if "f_dc_0" in self._prop_index:
            dc = self.vertices[:, [self._prop_index["f_dc_0"],
                                    self._prop_index["f_dc_1"],
                                    self._prop_index["f_dc_2"]]]
            rgb = dc * _SH_C0 + 0.5
            return np.clip(rgb, 0.0, 1.0)
        elif "red" in self._prop_index:
            rgb = self.vertices[:, [self._prop_index["red"],
                                     self._prop_index["green"],
                                     self._prop_index["blue"]]]
            # 判断范围: 0-255 vs 0-1
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            return np.clip(rgb, 0.0, 1.0)
        else:
            raise KeyError("PLY 中无颜色属性 (f_dc_0 或 red)")

    @property
    def opacities(self) -> Optional[np.ndarray]:
        if "opacity" in self._prop_index:
            return self.vertices[:, self._prop_index["opacity"]]
        return None


def read_ply(path: Path) -> PlyData:
    """读取 PLY 文件（自动识别 ASCII / binary）。"""
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            header_bytes += line
            if line.strip() == b"end_header":
                break
            if not line:
                raise ValueError(f"PLY 文件损坏，找不到 end_header: {path}")

    header_text = header_bytes.decode("utf-8", errors="ignore")
    data = _parse_header(header_text)

    # 数据起始偏移
    data_offset = len(header_bytes)
    vertex_bytes = data.num_vertices * data._vertex_stride()

    with open(path, "rb") as f:
        f.seek(data_offset)
        raw = f.read(vertex_bytes)

    if len(raw) < vertex_bytes:
        raise ValueError(f"PLY 数据不完整: 期望 {vertex_bytes} bytes, 实际 {len(raw)} bytes")

    _read_vertices(data, raw)
    logger.info(f"PLY 读取完成: {data.num_vertices:,} vertices, "
                f"{len(data.props)} props, format={data.format}")
    return data


# ================================================================
# 内部实现
# ================================================================

_NP_DTYPE_MAP = {
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
    "int": np.int32,
    "int32": np.int32,
    "uint8": np.uint8,
    "uchar": np.uint8,
}


def _parse_header(text: str) -> PlyData:
    """解析 PLY header 文本。"""
    data = PlyData()
    data.header_text = text
    lines = text.strip().split("\n")

    if lines[0].strip() != "ply":
        raise ValueError("不是有效的 PLY 文件（首行应为 'ply'）")

    for line in lines[1:]:
        line = line.strip()
        if line == "end_header":
            break
        if line.startswith("format "):
            data.format = line.split()[1]  # ascii | binary_little_endian | binary_big_endian
        elif line.startswith("element vertex "):
            data.num_vertices = int(line.split()[2])
        elif line.startswith("property "):
            parts = line.split()
            # "property float x" 或 "property list uchar int vertex_indices"
            if parts[1] == "list":
                continue  # 跳过 list 属性
            prop_name = parts[2]
            prop_type = parts[1]
            data.props.append({"name": prop_name, "dtype": prop_type})

    # 建立 name → column index 映射
    for i, p in enumerate(data.props):
        data._prop_index[p["name"]] = i

    return data


def _read_vertices(data: PlyData, raw: bytes) -> None:
    """读取顶点数据到 NumPy 数组。"""
    n = data.num_vertices
    p = len(data.props)

    if data.format.startswith("binary"):
        fmt_char = "<" if "little" in data.format else ">"
        np_dtype = np.dtype([
            (f"p{i}", _NP_DTYPE_MAP.get(prop["dtype"], np.float32))
            for i, prop in enumerate(data.props)
        ])
        structured = np.frombuffer(raw, dtype=np_dtype, count=n)
        # 转为 (N, P) float32
        data.vertices = np.zeros((n, p), dtype=np.float32)
        for i in range(p):
            data.vertices[:, i] = structured[f"p{i}"].astype(np.float32)

    else:
        # ASCII
        lines = raw.decode("utf-8", errors="ignore").strip().split("\n")
        data.vertices = np.zeros((n, p), dtype=np.float32)
        for row_idx, line in enumerate(lines):
            if row_idx >= n:
                break
            parts = line.strip().split()
            for col_idx in range(min(p, len(parts))):
                try:
                    data.vertices[row_idx, col_idx] = float(parts[col_idx])
                except ValueError:
                    pass


def _vertex_stride(self: PlyData) -> int:
    """每顶点字节数。"""
    type_sizes = {
        "float": 4, "float32": 4, "double": 8, "float64": 8,
        "int": 4, "int32": 4, "uint8": 1, "uchar": 1,
    }
    return sum(type_sizes.get(p["dtype"], 4) for p in self.props)


# Monkey-patch: 将 _vertex_stride 绑定到类（避免循环引用）
PlyData._vertex_stride = _vertex_stride


def extract_vertices(data: PlyData, indices: list[int], output_path: Path) -> Path:
    """从 PLY 中提取指定顶点，写入新 PLY 文件（仅包含选中的高斯球）。

    用于将识别结果导出为独立 PLY，直接在编辑器中打开 = 全选状态。
    """
    idx = np.array(indices, dtype=np.int64)
    data.vertices = data.vertices[idx, :]
    data.num_vertices = len(idx)

    header_lines = data.header_text.strip().split("\n")
    out_lines = []
    for line in header_lines:
        if line.strip() == "end_header":
            break
        if line.startswith("element vertex "):
            out_lines.append(f"element vertex {len(idx)}")
        else:
            out_lines.append(line)
    out_lines.append("end_header\n")
    data.header_text = "\n".join(out_lines)

    with open(output_path, "wb") as f:
        f.write(data.header_text.encode("utf-8"))
        for row in data.vertices:
            for val in row:
                f.write(np.float32(val).tobytes())

    return output_path
