import os
import sys
import time
from pathlib import Path

log_file = Path("E:/apex-coder/training/train.log")

if not log_file.exists():
    print("Train log not found yet.")
    sys.exit(0)

lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
recent = [l for l in lines if "[Step" in l or "Training" in l or "Loss:" in l]

print(f"=== Apex Coder Training Progress (Total log lines: {len(lines)}) ===")
for l in recent[-15:]:
    print(l)

lora_dir = Path("E:/apex-coder/outputs/apex_coder_3b_lora")
if lora_dir.exists():
    files = list(lora_dir.glob("*"))
    print(f"\nAdapter Artifacts ({len(files)} files): {[f.name for f in files]}")
