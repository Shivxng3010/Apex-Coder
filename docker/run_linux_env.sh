#!/bin/bash
set -e

echo "=== Initializing Linux CUDA Environment with uv ==="
apt-get update -qq && apt-get install -y -qq curl python3 python3-venv git build-essential > /dev/null

if [ ! -f "/root/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="/root/.local/bin:$PATH"

if [ ! -d "/workspace/venv_linux" ]; then
    uv venv /workspace/venv_linux --python 3.10
fi

echo "=== Installing PyTorch with CUDA 12.1 ==="
uv pip install torch torchvision --find-links /root/.cache/uv --index-url https://download.pytorch.org/whl/cu121 --python /workspace/venv_linux/bin/python

echo "=== Installing Unsloth, TRL, PEFT, bitsandbytes ==="
uv pip install transformers datasets peft trl bitsandbytes accelerate sentencepiece protobuf --python /workspace/venv_linux/bin/python
uv pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" --python /workspace/venv_linux/bin/python || true

echo "=== Verifying CUDA in Linux VirtualEnv ==="
/workspace/venv_linux/bin/python -c "import torch; print('LINUX CUDA READY! Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
