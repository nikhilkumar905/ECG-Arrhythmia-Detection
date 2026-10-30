@echo off
cd /d "%~dp0"
start "CardioScan API" cmd /k "cd backend && .venv\Scripts\python -m uvicorn main:app --port 8001"
start "CardioScan UI" cmd /k "cd frontend && npm run dev"
timeout /t 4 >nul
start http://localhost:5173
