@echo off
echo ========================================
echo    NLP CRM System - Server Starter
echo ========================================
echo.

cd /d "D:\nlpcrmzip (4)\nlpcrmzip\nlpcrm"
echo Current directory: %CD%
echo.

echo Activating virtual environment...
call .venv\Scripts\activate
echo Virtual environment activated
echo.

echo Testing Python and dependencies...
python -c "import sys; print('Python:', sys.executable)"
python -c "import fastapi; print('FastAPI: OK')"
python -c "import uvicorn; print('Uvicorn: OK')"
python -c "import api_server; print('API Server: OK')"
echo.

echo Starting NLP CRM API Server...
echo Server will be available at: http://127.0.0.1:8000
echo CRM Dashboard: http://127.0.0.1:8000/crm
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload

echo.
echo Server stopped.
pause








