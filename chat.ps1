$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path "E:\hf_cache") {
    $env:HF_HOME = "E:\hf_cache"
    $env:UV_CACHE_DIR = "E:\uv_cache"
    $env:TORCH_HOME = "E:\hf_cache\torch"
}
$env:PYTHONUNBUFFERED = "1"

$pythonExe = Join-Path $scriptDir "venv\Scripts\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe (Join-Path $scriptDir "run_chat.py") @args
} else {
    & python (Join-Path $scriptDir "run_chat.py") @args
}
