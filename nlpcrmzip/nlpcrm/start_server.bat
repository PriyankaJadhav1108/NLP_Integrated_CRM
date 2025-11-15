@echo off
echo Starting NLP CRM API Server...
echo.
cd /d "D:\nlpcrmzip (4)\nlpcrmzip\nlpcrm"
call .venv\Scripts\activate
echo Virtual environment activated
echo.
echo Starting server on http://127.0.0.1:8000
echo CRM Dashboard: http://127.0.0.1:8000/crm
echo.
echo Press Ctrl+C to stop the server
echo.
python run_server.py
pause








