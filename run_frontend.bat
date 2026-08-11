@echo off
setlocal
cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo Missing frontend\node_modules
    echo Run: npm --prefix frontend ci
    exit /b 1
)

echo ZipSplat-Demo Frontend
echo PC:   http://localhost:5173
echo Phone: http://<PC-LAN-IP>:5173  (run "ipconfig" to find your PC LAN IP)
echo.
echo Make sure backend is running: http://localhost:8000
echo.
npm run dev
pause
