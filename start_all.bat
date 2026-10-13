@echo off
title MACONFLIC Platform Launcher
echo ========================================================
echo   Launching MACONFLIC Platform (Backend + Frontend)
echo ========================================================
echo.
echo [1/2] Starting Backend in new window (http://localhost:8000)...
start "MACONFLIC Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend in new window (http://localhost:3000)...
start "MACONFLIC Frontend" cmd /k "cd /d "%~dp0web\frontend" && npm run dev"

echo.
echo ========================================================
echo  Both services launched!
echo  Backend:  http://localhost:8000 (API Docs: http://localhost:8000/docs)
echo  Frontend: http://localhost:3000
echo ========================================================
echo.
pause
