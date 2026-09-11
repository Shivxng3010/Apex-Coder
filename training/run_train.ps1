# Apex Coder - GPU Training Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      APEX CODER - CUDA GPU TRAINING LAUNCHER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ImageName = "apex-coder-train:latest"
$WorkspacePath = (Get-Item -Path "apex-coder").FullName

# Check if image exists
$imageExists = docker images -q $ImageName
if (-not $imageExists) {
    Write-Host "Building CUDA training image ($ImageName)..." -ForegroundColor Yellow
    docker build -t $ImageName -f apex-coder/docker/Dockerfile.train apex-coder
}

Write-Host "Starting QLoRA training inside GPU container with RTX 3050 pass-through..." -ForegroundColor Green
docker run --rm -it `
    --gpus all `
    --ipc=host `
    --memory 12g `
    -v "${WorkspacePath}:/workspace" `
    $ImageName `
    python training/train_unsloth.py `
        --model "Qwen/Qwen2.5-Coder-3B-Instruct" `
        --dataset "data/clean/train_2k.jsonl" `
        --output "outputs/apex_coder_3b_lora" `
        --seq-len 1536 `
        --rank 16 `
        --alpha 32 `
        --epochs 1 `
        --lr 2e-4
