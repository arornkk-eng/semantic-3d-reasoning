# ZipSplat Object Reconstruction Demo

本地多视角图片或视频到 3D Gaussian Splatting 模型的 Web 应用。

## 环境基线

当前验证环境：

* Windows 11
* Python 3.13.3
* Node.js 24.16.0
* NVIDIA CUDA Toolkit 12.8
* PyTorch 2.11.0+cu128
* NVIDIA GPU，建议至少 6 GB 显存
* MSVC C++ Build Tools，用于首次编译 gsplat CUDA 扩展

模型权重采用 CC BY-NC 4.0，仅限非商业用途。项目代码与模型权重不是同一许可证。

## 后端安装

在仓库根目录执行：

```powershell
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次重建可能编译 CUDA 扩展。执行前确保 `nvcc --version` 可用，MSVC 开发者环境已加载。

模型默认从 Hugging Face 下载。离线运行时需提前把权重放入本地缓存，或按 ZipSplat 接口传入本地权重路径。

### SAM 2.1 Tiny

交互分割使用官方 SAM 2.1 Hiera Tiny。Windows 环境跳过可选 CUDA 后处理扩展：

```powershell
git clone https://github.com/facebookresearch/sam2.git third_party/sam2
$env:SAM2_BUILD_CUDA="0"
venv\Scripts\python.exe -m pip install --no-build-isolation -e third_party/sam2
```

权重保存为 `models/sam2.1_hiera_tiny.pt`。可通过 `SAM2_CHECKPOINT` 和 `SAM2_MODEL_CONFIG` 覆盖路径与配置。

## 前端安装

```powershell
npm --prefix frontend ci
npm --prefix frontend run build
npm --prefix supersplat-editor ci
npm --prefix supersplat-editor run deploy:local
```

`package-lock.json` 锁定前端依赖。`deploy:local` 把 SuperSplat 生产包发布到 `frontend/public/splat-editor`，不携带 source map。开发服务器通过 Vite 把 `/api` 代理到 `http://localhost:8000`。

## 启动

一键启动全部功能并自动打开浏览器：

```powershell
.\run_all.bat
```

也可以双击仓库根目录中的 `一键启动全部功能.bat`。

双击或在终端运行：

```powershell
.\run_backend.bat
.\run_frontend.bat
```

访问地址：

* 前端：http://localhost:5173
* 后端：http://localhost:8000
* API 文档：http://localhost:8000/docs

可选环境变量：

* `CUDA_HOME`：CUDA Toolkit 路径。未设置时启动脚本使用 CUDA 12.8 默认安装路径。
* `TORCH_CUDA_ARCH_LIST`：目标 GPU 架构。未设置时启动脚本使用 `8.9`，对应 RTX 40 系列。
* `ZIPSPLAT_CORS_ORIGINS`：逗号分隔的前端来源白名单。
* `REALTIME_YOLO_MODEL`：实时预览检测权重，默认使用仓库根目录 `yolo26n-objv1-150.pt`（Objects365 类别集）。权重不入 Git，也可通过该环境变量指定本地路径。
* `REALTIME_YOLO_CONFIDENCE`：实时检测置信度阈值，默认 `0.35`。
* `REALTIME_YOLO_IMAGE_SIZE`：YOLO 推理尺寸，默认 `640`。
* `REALTIME_YOLO_MAX_DETECTIONS`：每帧最大框数，默认 `15`。

## 数据目录

运行时数据写入以下目录，不应提交版本库：

* `data/uploads`
* `data/tasks`
* `data/outputs`
* `models`
* `results`

删除任务会删除对应上传和输出文件。实验结果应先复制到独立归档目录。

## 质量检查

```powershell
venv\Scripts\python.exe -m pytest backend/tests
venv\Scripts\python.exe -m ruff check backend tools
npm --prefix frontend run build
npm --prefix supersplat-editor run lint
npm --prefix supersplat-editor run deploy:local
```

## 主要流程

图片或视频上传后，后端将任务写入磁盘队列。单 Worker 使用独立 Python 子进程执行视图筛选、ZipSplat 推理、Gaussian 后处理及 PLY 和 splat 导出。单任务结束后子进程退出，释放 GPU 显存。

当前生产参数集中在 `backend/core/config.py`：默认 6 个重建视图，scene alpha 阈值 0.02，最近邻尾点过滤 1%，splat 尺度保持 1.0。参数来源及对照数据见 `evaluation/experiments/README.md`。

## 编辑器语义分割

相机移动时，编辑器以 `640×360` 低分辨率截图调用 Objects365 YOLO26n，仅显示实时预览框。左上角“YOLO实时检测”按钮可关闭该功能，设置保存在浏览器本地。相机停止后不自动启动高成本模型；用户点击“分割”时，YOLO 暂停并清框，再由 Grounding DINO 与 SAM 2.1 执行正式实例分割。最终语义图层不使用 YOLO 结果。

正式分割支持 32 类常见室内物体。每次按置信度最多保留 7 个识别实例，可通过 `MAX_SEMANTIC_INSTANCES` 环境变量调整。编辑器从当前焦点捕获中心及两个辅助轨道视角的 RGB、深度和覆盖率，融合逐视角实例 mask 后可执行：

1. `投射到3D`：按相机矩阵、深度和透明度选择可见 Gaussian。
2. `补全3D`：在 mask 证据和局部 Gaussian 邻接约束内补全破损或遮挡部分。
3. `生成独立3D图层`：从原 Splat 分离选区，支持撤销和重做。
4. `保存所选图层`：保存 2D mask、相机信息和 `uint32-le` Gaussian 索引 sidecar。
5. `补全已有图层`：把新视角投射得到的新增 Gaussian 合并到用户指定的已有图层。
6. `删除已有图层`：确认影响后删除当前会话的 mask、索引和观测记录，原始 PLY 不受影响。
7. `生成物理代理`：直接从实例 Gaussian 的中心、尺度和旋转生成闭合 OBB、圆柱体、凸包或支撑面代理，用于后续静态刚体分析。
8. `分析支撑关系`：用支撑面校准重力方向，通过 PyBullet 稳定模拟、接触力和移除支撑物实验生成 `supported_by` 世界关系。
9. `设为地面`：用户指定已有 Gaussian 图层作为地面，系统拟合平面；用户可翻转法线并确认，支撑分析只读取已确认标定。

浏览器仅缓存未提交编辑状态。当前会话图层暂存于 `data/layers/{task_id}/{layer_id}`；打开编辑器时清理上次残留，页面关闭或刷新时再清理。
