@echo off
if exist "E:\hf_cache" (
    set HF_HOME=E:\hf_cache
    set UV_CACHE_DIR=E:\uv_cache
    set TORCH_HOME=E:\hf_cache\torch
)
set PYTHONUNBUFFERED=1

if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0run_chat.py" %*
) else (
    python "%~dp0run_chat.py" %*
)
