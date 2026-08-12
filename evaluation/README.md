# 重建评估基线

`baseline_config.json` 固定任务、输入、输出和当前推理参数。执行：

```powershell
venv\Scripts\python.exe tools\evaluate_baseline.py evaluation\baseline_config.json
```

产物写入 `evaluation/baselines/<name>/`：

* `manifest.json`：配置、文件 SHA256、尺寸和生成时间
* `metrics.json`：Gaussian 数量、包围盒、透明度、尺度、最近邻距离统计
* `preview.png`：输入缩略图及点云 XY、XZ、YZ 正交投影视图

后续实验复制配置并修改 `name`、`ply_path` 与参数。相同输入必须保持 SHA256 一致。统计用于发现规模、裁剪和离群点变化，预览用于快速目检，不替代 NVS 的 PSNR、SSIM 指标。
