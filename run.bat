@echo off
setlocal enabledelayedexpansion

title ResoGenre System Launcher
color 0B

echo ===============================================================================
echo                ResoGenre: Deep-Learning Audio Genre Classifier
echo ===============================================================================
echo.

:: Anchor script to its own directory
cd /d "%~dp0"

echo [*] Checking prerequisites...

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10+ and add it to your System PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo  [+] Detected: !PYTHON_VER!

:: 2. Check Node & NPM
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js / NPM is not found in PATH!
    echo Please install Node.js (v18+) to run the Next.js frontend.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version 2^>^&1') do set NODE_VER=%%i
echo  [+] Detected: Node !NODE_VER!

:: 3. Check Best Model Checkpoint
if not exist "models\resogenre_cnn_best.pth" (
    echo  [!] Warning: models\resogenre_cnn_best.pth not found.
    echo      Make sure you have trained or placed a model checkpoint in models/.
) else (
    echo  [+] Model checkpoint verified: models\resogenre_cnn_best.pth
)

echo.
echo ===============================================================================
echo [*] Launching Servers in separate dedicated windows...
echo ===============================================================================
echo.

:: Launch FastAPI backend
echo  [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "ResoGenre Backend (FastAPI)" cmd /k "cd /d "%~dp0" && title ResoGenre Backend && python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"

:: Launch Next.js frontend
echo  [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "ResoGenre Frontend (Next.js)" cmd /k "cd /d "%~dp0frontend" && title ResoGenre Frontend && npm run dev"

echo.
echo [*] Waiting 4 seconds for servers to initialize...
timeout /t 4 /nobreak >nul 2>&1

echo [*] Opening ResoGenre Web Application in default browser...
start http://localhost:3000

echo.
echo ===============================================================================
echo  [SUCCESS] Both servers are now running!
echo ===============================================================================
echo  - Frontend UI:    http://localhost:3000
echo  - Backend API:    http://localhost:8000
echo  - API Swagger:    http://localhost:8000/docs
echo.
echo  Tip: To stop the servers, simply close the respective command prompt windows.
echo ===============================================================================
echo.
pause
