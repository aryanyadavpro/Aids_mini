@echo off
setlocal enabledelayedexpansion

title ResoGenre System Launcher
color 0B

echo ===============================================================================
echo                ResoGenre: Deep-Learning Audio Genre Classifier
echo ===============================================================================
echo.

:: Anchor script to its own directory and strip trailing backslash
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
cd /d "%ROOT_DIR%"

echo [*] Checking prerequisites...

:: 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in PATH!
    echo Please install Python 3.10 or higher and add it to your System PATH.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VER=%%i
echo  [+] Detected: !PYTHON_VER!

:: 2. Check Node & NPM
where npm >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node.js or NPM is not found in PATH!
    echo Please install Node.js v18 or higher to run the Next.js frontend.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version 2^>^&1') do set NODE_VER=%%i
echo  [+] Detected: Node !NODE_VER!

:: 3. Check Best Model Checkpoint
if not exist "%ROOT_DIR%\backend\models\resogenre_cnn_best.pth" (
    echo  [!] Warning: backend\models\resogenre_cnn_best.pth not found.
    echo      Make sure a trained model checkpoint is in backend\models folder.
) else (
    echo  [+] Model checkpoint verified: backend\models\resogenre_cnn_best.pth
)

echo.
echo [*] Freeing ports 8000 and 3000 (if left occupied by previous runs)...
python -c "import subprocess; [subprocess.run(f'taskkill /F /T /PID {l.split()[-1]}', shell=True, capture_output=True) for l in subprocess.getoutput('netstat -ano').splitlines() if (':8000' in l or ':3000' in l) and 'LISTENING' in l]"

echo.
echo ===============================================================================
echo [*] Launching Servers in separate dedicated windows...
echo ===============================================================================
echo.

:: Launch FastAPI backend in dedicated window with working directory set to backend
echo  [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "ResoGenre Backend" /D "%ROOT_DIR%\backend" cmd /k "title ResoGenre Backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: Launch Next.js frontend in dedicated window with working directory set to frontend
echo  [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "ResoGenre Frontend" /D "%ROOT_DIR%\frontend" cmd /k "title ResoGenre Frontend && npm run dev"

echo.
echo [*] Waiting for servers to initialize...
ping 127.0.0.1 -n 5 >nul

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
