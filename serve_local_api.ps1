# Apex Coder - Local OpenAI-Compatible API Server Runner (FastAPI + Uvicorn)
# Accessible at: http://127.0.0.1:8000/v1
# Swagger UI Docs: http://127.0.0.1:8000/docs
# Compatible with VS Code Continue.dev, Cursor, and standard OpenAI API clients

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "       APEX CODER - FASTAPI OPENAI SERVER (PORT 8000)      " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "Swagger UI Docs : http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "OpenAI Endpoint : http://127.0.0.1:8000/v1/chat/completions" -ForegroundColor Green
Write-Host "-----------------------------------------------------------"
Write-Host "VS Code Continue.dev config.json:"
Write-Host '  "models": [{'
Write-Host '    "title": "Apex Coder (3B)",'
Write-Host '    "provider": "openai",'
Write-Host '    "model": "apex-coder-3b",'
Write-Host '    "apiBase": "http://127.0.0.1:8000/v1"'
Write-Host '  }]'
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "`nStarting server... Keep this window open.`n"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $scriptDir "venv\Scripts\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe (Join-Path $scriptDir "serve_api.py")
} else {
    & python (Join-Path $scriptDir "serve_api.py")
}
