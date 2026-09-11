# Build Python Sandbox Image
Write-Host "Building Python verifier image..." -ForegroundColor Cyan
docker build -t coder-verifier-py:latest -f docker/Dockerfile.python .

# Build TypeScript Sandbox Image
Write-Host "Building TypeScript verifier image..." -ForegroundColor Cyan
docker build -t coder-verifier-ts:latest -f docker/Dockerfile.typescript .

Write-Host "Verifier images built successfully!" -ForegroundColor Green
