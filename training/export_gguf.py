import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")


def resolve_model_path(model_name: str) -> str:
    snapshot_dir = Path("E:/hf_cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")
    if snapshot_dir.exists() and "Qwen2.5-Coder-3B" in model_name:
        return str(snapshot_dir.resolve())
    return model_name


def merge_and_export_lora(
    base_model_path: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    lora_adapter_path: str = "E:/apex-coder/outputs/apex_coder_3b_lora",
    merged_output_dir: str = "E:/apex-coder/outputs/apex_coder_3b_merged",
):
    print("===========================================================")
    print("      APEX CODER - LoRA MERGE & FP16 EXPORT")
    print("===========================================================")
    resolved_base = resolve_model_path(base_model_path)
    
    # Check if final output exists, else check latest checkpoint
    adapter_dir = Path(lora_adapter_path)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        checkpoints = sorted(adapter_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            adapter_dir = checkpoints[-1]
            print(f"Using latest available checkpoint: {adapter_dir.name}")
        else:
            raise FileNotFoundError(f"No adapter found in {lora_adapter_path}")

    print(f"Base Model: {resolved_base}")
    print(f"LoRA Adapter: {adapter_dir}")
    print(f"Merged Target: {merged_output_dir}")
    print("-----------------------------------------------------------\n", flush=True)

    print("[1/3] Loading base model in FP16 on CPU...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(resolved_base, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Base model loaded in {time.time()-t0:.2f}s", flush=True)

    print("[2/3] Merging LoRA adapter weights...", flush=True)
    t1 = time.time()
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = model.merge_and_unload()
    print(f"Adapter fused into base model in {time.time()-t1:.2f}s!", flush=True)

    print(f"[3/3] Saving fused 16-bit model to {merged_output_dir}...", flush=True)
    out_dir = Path(merged_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Merged model saved successfully to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA and export FP16 weights")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--lora-dir", type=str, default="E:/apex-coder/outputs/apex_coder_3b_lora")
    parser.add_argument("--output-dir", type=str, default="E:/apex-coder/outputs/apex_coder_3b_merged")

    args = parser.parse_args()

    merge_and_export_lora(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_dir,
        merged_output_dir=args.output_dir,
    )
