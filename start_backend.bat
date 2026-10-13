@echo off
title MACONFLIC Backend (FastAPI)
echo ===================================================
echo Starting MACONFLIC Backend on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ===================================================
cd /d "%~dp0"
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
