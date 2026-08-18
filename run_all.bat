@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"

set "APP_ROOT=%~dp0"
set "BACKEND_URL=http://127.0.0.1:8000"
set "FRONTEND_URL=http://127.0.0.1:5173"
set "OPEN3D_PYTHON=%APP_ROOT%venv-open3d\Scripts\python.exe"
set "LOG_DIR=%APP_ROOT%logs"

echo [ZipSplat] Checking runtime...
if not exist "%APP_ROOT%venv\Scripts\python.exe" (
    echo [ERROR] Missing venv\Scripts\python.exe
    pause
    exit /b 1
)
if not exist "%APP_ROOT%frontend\node_modules" (
    echo [ERROR] Missing frontend\node_modules
    echo Run npm --prefix frontend ci first.
    pause
    exit /b 1
)
if not exist "%OPEN3D_PYTHON%" (
    echo [WARNING] Open3D runtime missing. Mesh generation will be unavailable.
) else (
    echo [OK] Open3D runtime found.
)

if /i "%~1"=="check" (
    echo [OK] Main runtime is ready.
    exit /b 0
)

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%BACKEND_URL%/docs' ^| Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo [ZipSplat] Starting backend...
    powershell -NoProfile -Command ^
      "$env:CUDA_HOME='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8';" ^
      "$env:CUDA_PATH=$env:CUDA_HOME;" ^
      "$env:PATH=($env:CUDA_HOME + '\bin;' + $env:PATH);" ^
      "$env:TORCH_CUDA_ARCH_LIST='8.9'; $env:PYTHONUTF8='1'; $env:OPEN3D_PYTHON='%OPEN3D_PYTHON%';" ^
      "$arguments=@('-m','uvicorn','backend.main:app','--host','0.0.0.0','--port','8000');" ^
      "Start-Process -FilePath '%APP_ROOT%venv\Scripts\python.exe' -ArgumentList $arguments -WorkingDirectory '%APP_ROOT%' -WindowStyle Hidden -RedirectStandardOutput '%LOG_DIR%\backend.log' -RedirectStandardError '%LOG_DIR%\backend-error.log'"
) else (
    echo [ZipSplat] Backend is already running.
)

powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 '%FRONTEND_URL%' ^| Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo [ZipSplat] Starting frontend...
    powershell -NoProfile -Command ^
      "$arguments=@('run','dev');" ^
      "Start-Process -FilePath 'npm.cmd' -ArgumentList $arguments -WorkingDirectory '%APP_ROOT%frontend' -WindowStyle Hidden -RedirectStandardOutput '%LOG_DIR%\frontend.log' -RedirectStandardError '%LOG_DIR%\frontend-error.log'"
) else (
    echo [ZipSplat] Frontend is already running.
)

echo [ZipSplat] Waiting for services...
powershell -NoProfile -Command ^
  "$deadline=(Get-Date).AddSeconds(90);" ^
  "$backend=$false; $frontend=$false;" ^
  "while((Get-Date)-lt $deadline -and -not($backend -and $frontend)){" ^
  "  try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%BACKEND_URL%/docs'|Out-Null;$backend=$true}catch{};" ^
  "  try{Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 '%FRONTEND_URL%'|Out-Null;$frontend=$true}catch{};" ^
  "  if(-not($backend -and $frontend)){Start-Sleep -Seconds 1}" ^
  "};" ^
  "if($backend -and $frontend){exit 0}else{Write-Host ('Backend=' + $backend + ' Frontend=' + $frontend);exit 1}"

if errorlevel 1 (
    echo [ERROR] Services did not become ready within 90 seconds.
    echo Check logs in: %LOG_DIR%
    pause
    exit /b 1
)

echo [OK] Backend: %BACKEND_URL%
echo [OK] API docs: %BACKEND_URL%/docs
echo [OK] Frontend: %FRONTEND_URL%
start "" "%FRONTEND_URL%"
exit /b 0
