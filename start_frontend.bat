@echo off
title MACONFLIC Frontend (Next.js)
echo ===================================================
echo Starting MACONFLIC Frontend on http://localhost:3000
echo ===================================================
cd /d "%~dp0web\frontend"
npm run dev
pause
