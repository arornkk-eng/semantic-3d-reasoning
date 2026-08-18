# Open3D TSDF 原型

该原型把编辑器捕获的多视角 RGB、线性深度和相机矩阵融合为三角网格。它不直接把 Gaussian 椭球转成面，而是重建可见表面的 TSDF 零等值面。

## 环境

Open3D 0.19 在 Windows 不支持项目当前 Python 3.13，因此使用 Python 3.12 隔离环境：

```powershell
D:\pqg\Ae study\python\uploads\sam3.1\python312\python.exe -m venv venv-open3d
venv-open3d\Scripts\python.exe -m pip install -r requirements-open3d.txt
```

如果 Windows 未启用长路径，Open3D 的 Jupyter 可选依赖可能安装失败。TSDF 路径不需要 Jupyter，可先安装 `open3d==0.19.0 --no-deps`，再安装 `numpy Pillow plotly dash flask nbformat configargparse werkzeug`。

## 输入格式

目录内放置 `metadata.json`、PNG 彩色图和 Float32 little-endian 深度。深度沿用编辑器定义：

```text
normalized_depth = (view_z - near) / (far - near)
```

无效深度写 NaN。可选 coverage 文件同为 Float32，可选 mask 为灰度 PNG。`metadata.json` 示例：

```json
{
  "views": [{
    "color": "color-0.png",
    "depth": "depth-0.f32",
    "coverage": "coverage-0.f32",
    "mask": "mask-0.png",
    "width": 1280,
    "height": 767,
    "near": 0.01,
    "far": 1000.0,
    "projection": "perspective",
    "view_matrix": [16个PlayCanvas列主序数值],
    "projection_matrix": [16个PlayCanvas列主序数值]
  }]
}
```

## 执行

```powershell
venv-open3d\Scripts\python.exe tools\open3d_tsdf.py INPUT_DIR OUTPUT.ply --voxel-size 0.005
venv-open3d\Scripts\python.exe tools\open3d_tsdf_smoke.py
```

`voxel-size` 使用场景世界单位。初始值建议取目标包围盒最长边的 0.5% 至 1%。`sdf-trunc` 默认是体素尺寸的 4 倍。

当前语义捕获只有中心和正负 6 度三个近邻视角，适合语义验证，不足以获得完整闭合网格。实际物体 TSDF 需要围绕隔离图层采集约 12 至 24 个有位移的视角，并在每张深度图上应用该物体 mask。
