ZipSplat 3D 重建 Web Demo — 项目总体汇报
一、项目概述
将 ZipSplat（DA3-Giant/ViT-G 前馈式 3D 高斯泼溅模型）封装为一个本地 Web Demo，用户通过浏览器上传多视角照片，后端 GPU 推理生成 3D 高斯点云（PLY 格式），前端提供下载和 SuperSplat 在线预览。

运行环境: NVIDIA RTX 4050 Laptop (6.4 GB VRAM), CUDA 12.8, Windows 11

二、项目架构

┌──────────────┐     HTTP/REST      ┌─────────────────┐     multiprocessing      ┌────────────────┐
│  React 前端   │ ◄────────────────► │  FastAPI 后端    │ ◄──────────────────────► │  ZipSplat 引擎  │
│  (Vite + TS)  │   轮询 / 上传 /    │  (Python 3.11)  │   子进程隔离 + 可终止     │  (torch + CUDA) │
│               │   下载 / 取消      │                 │                          │                │
└──────────────┘                    └─────────────────┘                          └────────────────┘
      │                                    │                                            │
      │ localStorage                       │ JSON 文件元数据                             │ PLY 导出
      │ 任务历史                           │ data/tasks/*.json                          │ data/outputs/
      ▼                                    ▼                                            ▼
  跨标签页同步                        无数据库依赖                                   scene.ply
关键设计决策
决策	选择	原因
任务元数据存储	JSON 文件（非 SQLite）	零依赖，原子写入（tmp → rename），调试友好
任务执行隔离	multiprocessing.Process（非 threading）	Process.terminate() 真正杀死任务 + GPU 显存由 CUDA 驱动自动回收
前端状态同步	2 秒轮询 + localStorage	简单可靠，无需 WebSocket
模型加载	函数内懒加载（非启动时预加载）	避免空跑时占用 6+ GB 显存
三、各环节技术点（附关键代码）
3.1 推理引擎 — runner.py
核心技术栈: ZipSplat (DA3-Giant) → torch.no_grad() 单次前向传播 → 3D Gaussian Splatting


# [runner.py:43-47] 模型懒加载 + 单次推理
model = ZipSplat(weights="zipsplat").to(device).eval()
images = [load_image(p) for p in image_paths]
with torch.no_grad():
    gaussians = model(images, compression=1.0)[0]
技术点:

ViT-G 隐式位姿估计: 无需用户提供相机外参，Cross-Attention 自动建模多视图几何关系
sh_degree=1: 输出 4 个 SH 系数/通道（DC + SH1 三方向），共 12 个系数
compression=1.0: 不压缩，全精度推理
3.2 颜色后处理 — 本项目最大技术难点
3DGS 的颜色存储在 Spherical Harmonics (球谐函数) 系数中。DC 分量到 RGB 的转换公式：

$$rgb = SH_C0 \times sh0 + 0.5, \quad SH_C0 = 0.28209479177387814$$

关键代码（runner.py:52-87）:


# 1) DC → RGB 转换
dc = gaussians.sh0.squeeze(-2)
rgb_raw = (dc * _SH_C0 + 0.5).clamp(0, 1)

# 2) 饱和度修正：亮度分离 + 饱和度缩放（SAT_BOOST = 1.5）
lum = 0.299 * rgb_raw[:, 0] + 0.587 * rgb_raw[:, 1] + 0.114 * rgb_raw[:, 2]
rgb_sat = torch.stack([
    lum + SAT_BOOST * (rgb_raw[:, 0] - lum),
    lum + SAT_BOOST * (rgb_raw[:, 1] - lum),
    lum + SAT_BOOST * (rgb_raw[:, 2] - lum),
], dim=-1).clamp(0, 1)

# 3) SH1 衰减 90%（不完全归零，保留微弱方向感）
gaussians.shN.mul_(0.1)

# 4) 不透明度只 clamp，不 boost（boost 是白色的根因）
gaussians.opacities.clamp_(0, 1)
技术原理: 亮度分离使用 ITU-R BT.601 标准权重（0.299R + 0.587G + 0.114B）——这与人类视觉感知一致。

3.3 背景剔除 — DBSCAN 空间聚类
管线（runner.py:108-202）:


透明度过滤 (alpha > 0.05) → 空间坐标提取 → DBSCAN → 保留前 N 大簇并集 → 重建 Gaussians 对象
关键代码:


# [runner.py:146-151] eps 自适应：基于场景包围盒对角线
bbox_min = means_filtered.min(axis=0)
bbox_max = means_filtered.max(axis=0)
scene_diag = float(np.linalg.norm(bbox_max - bbox_min))
eps = scene_diag * eps_ratio       # eps_ratio = 0.03

# [runner.py:164-178] 多簇保留：前 top_n 大簇并集（关键创新）
sorted_idx = counts.argsort()[::-1]
top_k = min(top_n, len(cluster_ids))
top_labels = cluster_ids[sorted_idx[:top_k]]
mask_cluster = np.isin(labels, top_labels)

# [runner.py:190-196] 从过滤后索引重建 Gaussians 对象
gaussians = Gaussians.from_parameters(
    means=gaussians.means[final_mask],
    scales=gaussians.scales[final_mask],
    quats=gaussians.quats[final_mask],
    opacities=gaussians.opacities[final_mask],
    sh_coeffs=gaussians.sh_coeffs[final_mask],
)
技术亮点: 保留前 N 大簇的并集（而非仅最大簇），解决了相机位姿误差导致物体背部/侧面被 DBSCAN 误判为噪声的问题。这个设计从 single-cluster 升级为 multi-cluster，高斯球保留量提升了 65%。

3.4 任务终止 — multiprocessing 隔离

# [worker.py:36-71] 终止运行中的子进程
def cancel_current(task_id: str) -> bool:
    proc.terminate()           # SIGTERM（优雅终止）
    proc.join(timeout=10)
    if proc.is_alive():
        proc.kill()             # SIGKILL（强制终止）
        proc.join(timeout=5)
队列层取消（queue_manager.py:44-62）:


def cancel(self, task_id: str) -> bool:
    # 等待中 → 标记 cancelled + 加入跳过集合
    # dequeuer 自动跳过 _cancelled 集合中的任务
技术要点:

multiprocessing.Process（非 threading.Thread）—— Process.terminate() 是真正的 OS 级终止
双保险：先 terminate()，10 秒不响应则 kill()
子进程终止后 CUDA 驱动自动回收该进程持有的 GPU 显存
3.5 API 设计
端点	方法	功能	文件
/api/upload	POST	多图上传 + 创建任务	upload.py
/api/task/{id}	GET	查询任务状态	task.py:11-23
/api/task/{id}/cancel	POST	取消等待中/运行中任务	task.py:26-60
/api/task/{id}	DELETE	删除任务 + 清理文件	task.py:63-73
/api/result/{id}	GET	列出输出文件	result.py
/api/result/{id}/{file}	GET	下载输出文件	result.py:35-51
/api/health	GET	健康检查 + GPU 信息	main.py:65-86
3.6 前端 — React + TypeScript + Vite
状态管理: useTask Hook 驱动 2 秒轮询 → 终态自动停止


// [useTask.ts:19-20] 终态判断（含 cancelled）
const isTerminal = (t: TaskMeta | null) =>
    t?.status === "completed" || t?.status === "failed" || t?.status === "cancelled";
任务卡片 UI（TaskCard.tsx）:

hover 显示终止按钮（StopIcon）和删除按钮（TrashIcon）
删除前二次确认
cancelled 状态琥珀色提示
结果面板（ResultPanel.tsx）:

三列统计卡片（高斯球数 / PLY 大小 / 文件名）
下载按钮 + SuperSplat 3D 预览链接（一键直达）
3.7 PLY → .splat 转换（新增模块）

# [splat_converter.py] 32 字节/高斯的紧凑二进制格式
# position(float32*3) + scale(float32*3) + color(uint8*4) + rotation(uint8*4)
比 PLY 小 3-5×，浏览器端可直接由 GPU 读取，无需解析文本格式。

四、难点分析
难点 1：颜色发白（耗时占比 ~35%）
表象: 重建后渲染全部偏白，几乎不可辨认物体颜色。

根因探究过程:

最初以为是 SH 系数转换公式有误 → 验证 rgb = SH_C0 × sh0 + 0.5 无误
然后怀疑 SH1 方向分量引入白色 → 归零 SH1，效果不显著
接着尝试提高饱和度 → 颜色看起来更浓但整体仍然发白
最终定位: 原始代码 ALPHA_BOOST = 3.0 将不透明度放大 3 倍
物理原因: 3DGS 的 alpha 混合公式为 $\alpha_i = 1 - e^{-opacity \times weight}$。数千个灰白色高斯球在 boost 后全部趋于完全不透明，沿视线方向累积混合 → RGB 三个通道全部逼近 1.0（白色）。

教训: 参数调整要追根溯源，不能头痛医头。饱和度、SH、亮度修正都只是症状缓解，根因是不透明度的异常放大。

难点 2：DBSCAN 参数敏感度（耗时占比 ~25%）
同一模型、同一批 8 张照片，仅因选片角度不同，高斯球数量从 19,747 掉到 12,772（-35%）。

根因: DBSCAN 的 eps 对点云密度极度敏感：

均匀间隔视角 → 深度信号弱 → 高斯球空间分散 → DBSCAN 碎片化
大跳变视角 → 强视差 → 高斯球空间集中 → DBSCAN 聚类效果好
解决:

eps = 场景对角线 × eps_ratio（自适应）
经过多轮网格搜索：eps_ratio: 0.02 → 0.04 → 0.03；min_samples: 15 → 10 → 12
多簇保留（top_n=3）替代单簇保留
难点 3：模型黑盒 — 视角选择的隐性影响（耗时占比 ~15%）
ZipSplat 是端到端 black-box，内部位姿估计、跨视角融合完全不可控。

实验发现:

选片策略	高斯球数	质量
15+ 张，均匀间距	~12,772	❌ 最差
8 张，大跳变 + 极端视角	~21,112	✅ 最优
规律: ZipSplat 训练于 2-6 张控制间距的视角；手机拍摄的 15+ 张照片引入曝光/白平衡不一致，反而降低模型置信度。

难点 4：任务终止的进程隔离（耗时占比 ~20%）
挑战: threading 方案下，Thread 无法从外部强制终止，GPU 显存也无法回收。

方案: 整个 Worker 从 threading 重构为 multiprocessing：

子进程执行推理（独立 CUDA context）
父进程主线程通过 Process.terminate() 终止
终止后 CUDA 驱动自动回收显存
注意点: 子进程直接写 JSON 元数据文件（而非通过父进程通信），避免管道序列化大数据的问题。

五、时间耗费分布
阶段	耗时占比	主要工作
Phase 0: 环境搭建	5%	ZipSplat 模型 + 依赖安装，CUDA 环境验证
Phase 1: 后端 API	15%	FastAPI 路由、上传、任务队列、文件管理
Phase 2: 前端	10%	React 组件、轮询 Hook、拖拽上传
颜色发白调试	35%	从 SH → 饱和度 → 不透明度的排查链
任务终止功能	20%	threading → multiprocessing 重构
DBSCAN 参数调优	15%	选片策略分析 + 参数网格搜索
总开发时间: 约 2 天（含多轮测试-调试迭代）

六、最终成果

原始模型输出  →  31,104 高斯球（含背景噪声 + 发白）
优化后最终版  →  21,112 高斯球（主体纯净 + 颜色正常）
指标	优化前	优化后
高斯球保留率	100%（含大量噪声）	68%（干净主体）
颜色	整体发白，不可辨认	饱和度+50%，纹理可见
单任务重建耗时	~2 分钟	~2 分钟
任务可终止	❌	✅ 等待中/运行中均可终止
前端体验	基础轮询	拖拽上传 + 取消 + 3D 预览
