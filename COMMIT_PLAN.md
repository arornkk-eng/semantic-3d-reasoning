# 提交边界

## 建议提交

### 运行与依赖

* `README.md`
* `requirements.txt`
* `requirements-dev.txt`
* `run_backend.bat`
* `run_frontend.bat`
* `pyproject.toml`
* `.pre-commit-config.yaml`
* `.github/`

### AI 与失效识别功能清理

* `backend/main.py`
* `backend/core/config.py`
* `frontend/src/App.tsx`
* `frontend/src/api/client.ts`
* `frontend/src/types/index.ts`
* 删除 `backend/recognition/`

### 重建参数与视图选择

* `backend/zipsplat_engine/runner.py`
* `backend/zipsplat_engine/splat_converter.py`
* `ZipSplat-Demo/scripts/pick_views.py`
* `backend/tests/`

### 评估基础设施

* `evaluation/` 中 JSON、Markdown、PNG
* `tools/`

PLY、splat、PyTorch 张量由 `.gitignore` 排除，可通过配置和脚本重新生成。

## 提交前单独确认

以下改动在本轮审计前已存在，不能自动判断是否应随本次提交：

* 删除 `ZipSplat-Demo/image/input/1.png` 到 `3.png`
* 删除 `ZipSplat-Demo/log/conclusion.md`
* `backend/api/ply_viewer.py`
* `backend/api/result.py`
* `backend/api/upload.py`
* `backend/core/queue_manager.py`
* `backend/core/worker.py`
* `backend/storage/file_manager.py`
* `backend/video_processor/extractor.py`
* `frontend/public/` 内嵌查看器改动
* `frontend/vite.config.ts`

## 外部源码

`ZipSplat-main/` 是后端运行依赖，应明确选择 vendor 提交或 Git submodule。当前不是 submodule。

`supersplat-editor/` 是带独立 `.git` 的外部仓库。不要直接作为普通目录整体提交。若仍需维护源码，使用 Git submodule；当前前端运行使用 `frontend/public/` 中的构建产物。

## 不提交

* `data/`
* `results/`
* `models/`
* 虚拟环境和缓存
* `.claude/`
* `.workbuddy/`
* `supersplat-editor/node_modules/`
* `supersplat-editor/dist/`
* `ZipSplat-main/assets/`
