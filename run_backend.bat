@echo off
cd /d "d:\pqg\Ae study\python\uploads\ZipSplat-Object-Reconstruction-Demo"

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "PYTHONUTF8=1"

echo ZipSplat-Demo Backend
echo http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
