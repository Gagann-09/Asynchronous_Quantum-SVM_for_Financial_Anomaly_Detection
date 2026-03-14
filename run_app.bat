@echo off
TITLE Q-SVM Application Launcher
ECHO Starting Q-SVM Components...

:: Port 6379 for Redis, 8000 for Backend, 3000 for Frontend

:: 1. Start Redis Server
ECHO [1/4] Starting Redis Server...
start "Redis Server" cmd /k "cd redis && redis-server.exe redis.windows.conf"

:: 2. Start FastAPI Backend
ECHO [2/4] Starting FastAPI Backend...
start "Backend (FastAPI)" cmd /k "call venv\Scripts\activate && cd backend && uvicorn app.main:app --reload --port 8000"

:: 3. Start Celery Worker
:: Note: -P solo is recommended for Celery on Windows
ECHO [3/4] Starting Celery Worker...
start "Celery Worker" cmd /k "call venv\Scripts\activate && cd backend && celery -A app.worker.celery_app worker --loglevel=info -P solo"

:: 4. Start Next.js Frontend
ECHO [4/4] Starting Next.js Frontend...
start "Frontend (Next.js)" cmd /k "cd frontend && npm run dev"

ECHO All components are starting in separate windows.
ECHO Please keep those windows open while using the app.
PAUSE
