@echo off
title Crisis-AI Assistant (Offline Launcher)
color 0B

echo ===================================================
echo   CRISIS-AI ASSISTANT - OFFLINE LAUNCHER
echo ===================================================
echo.

:: 1. Verify python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system!
    echo Please install Python and add it to your PATH.
    pause
    exit /b
)

:: 2. Check if Ollama is running offline
echo [INFO] Checking local Ollama instance...
tasklist /fi "imagename eq ollama.exe" | findstr /i "ollama.exe" >nul
if %errorlevel% neq 0 (
    echo [WARNING] Ollama is not running! Local AI answers might be disabled.
    echo Please start Ollama before continuing for full offline AI support.
    echo.
) else (
    echo [SUCCESS] Ollama is running and ready.
)

:: 3. Start Flask server in a new terminal window
echo [INFO] Starting the local Crisis-AI server...
start "Crisis-AI Backend" cmd /c "python -u run.py --web"

:: 4. Wait for 3 seconds to let the server bind
echo [INFO] Waiting for server to initialize...
timeout /t 3 /nobreak >nul

:: 5. Open loopback IP in browser (bypasses DNS so it works perfectly offline)
echo [SUCCESS] Opening Crisis-AI at http://127.0.0.1:5000...
start http://127.0.0.1:5000

echo.
echo ===================================================
echo   Crisis-AI is running offline. Keep this window open.
echo   To close the server, close the "Crisis-AI Backend" window.
echo ===================================================
echo.
pause
