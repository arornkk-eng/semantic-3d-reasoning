@echo off
setlocal
cd /d "%~dp0"

if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%CUDA_HOME%\bin;%PATH%"
if not defined TORCH_CUDA_ARCH_LIST set "TORCH_CUDA_ARCH_LIST=8.9"
set "PYTHONUTF8=1"

if not exist "venv\Scripts\python.exe" (
    echo Missing venv\Scripts\python.exe
    echo See README.md for installation.
    exit /b 1
)

echo ZipSplat-Demo Backend
echo http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
