@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44 > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo vcvars64.bat failed
    exit /b 1
)
cd /d "d:\pqg\Ae study\python\uploads\ZipSplat-Object-Reconstruction-Demo"
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=8.9
set PYTHONUTF8=1
venv\Scripts\python.exe ZipSplat-Demo\scripts\reconstruct.py %*
