$env:HF_HOME = "E:\hf_cache"
$env:UV_CACHE_DIR = "E:\uv_cache"
$env:TORCH_HOME = "E:\hf_cache\torch"
$env:PYTHONUNBUFFERED = "1"

$logFile = "E:\apex-coder\training\train.log"
$errFile = "E:\apex-coder\training\train_err.log"
if (Test-Path $logFile) { Remove-Item $logFile }
if (Test-Path $errFile) { Remove-Item $errFile }

Write-Host "Launching Apex Coder 4GB QLoRA Training in background..." -ForegroundColor Green
$process = Start-Process `
    -FilePath "E:\apex-coder\venv\Scripts\python.exe" `
    -ArgumentList "E:\apex-coder\training\train_unsloth.py", `
                  "--model", "Qwen/Qwen2.5-Coder-3B-Instruct", `
                  "--dataset", "E:\apex-coder\data\clean\train_2k.jsonl", `
                  "--output", "E:\apex-coder\outputs\apex_coder_3b_lora", `
                  "--seq-len", "512", `
                  "--rank", "16", `
                  "--alpha", "32", `
                  "--epochs", "1", `
                  "--lr", "2e-4", `
                  "--grad-accum", "4", `
                  "--save-every", "50" `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru

Write-Host "Training process started with PID: $($process.Id)" -ForegroundColor Cyan
Write-Host "Logs redirecting to: $logFile" -ForegroundColor Cyan
