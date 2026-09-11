#!/bin/bash
set -e

echo "=========================================================="
echo "      APEX CODER - WSL2 ENVIRONMENT SETUP (CUDA & UNSLOTH)"
echo "=========================================================="

# Update and install build dependencies
sudo apt-get update
sudo apt-get install -y git curl build-essential python3-dev python3-pip python3-venv

# Create a clean virtual environment
python3 -m venv venv_apex
source venv_apex/bin/activate

# Install PyTorch with CUDA 12.1+ support
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Unsloth and training dependencies
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes datasets transformers

# Verify installation
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python3 -c "import unsloth; print('Unsloth Version:', unsloth.__version__)"

echo "WSL2 environment ready! Run training with:"
echo "source venv_apex/bin/activate && python apex-coder/training/train_unsloth.py"
