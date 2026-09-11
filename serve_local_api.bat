@echo off
title APEX CODER - FASTAPI OPENAI SERVER (PORT 8000)
echo ===========================================================
echo       APEX CODER - FASTAPI LOCAL SERVER (PORT 8000)
echo ===========================================================
echo Swagger UI Docs : http://127.0.0.1:8000/docs
echo OpenAI Endpoint : http://127.0.0.1:8000/v1/chat/completions
echo.
echo Starting server... Please keep this window open.
echo.

if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)
python "%~dp0serve_api.py"

pause
