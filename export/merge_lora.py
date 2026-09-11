import os
import sys
import time
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")


def resolve_base_model(model_name: str) -> str:
    snapshot_dir = Path("E:/hf_cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")
    if snapshot_dir.exists() and "Qwen2.5-Coder-3B" in model_name:
        return str(snapshot_dir.resolve())
    return model_name


def merge_lora_to_fp16(
    base_model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    lora_dir: str = "E:/apex-coder/outputs/apex_coder_3b_lora_v3",
    output_dir: str = "E:/apex-coder/outputs/apex_coder_3b_merged",
):
    print("===========================================================")
    print("      APEX CODER - LoRA MERGE & FP16 FUSION ENGINE")
    print("===========================================================")
    resolved_base = resolve_base_model(base_model_name)
    lora_path = Path(lora_dir)
    if not lora_path.exists():
        lora_path = Path("E:/apex-coder/outputs/apex_coder_3b_lora_v2")

    print(f"Base Model   : {resolved_base}")
    print(f"LoRA Adapter : {lora_path.resolve()}")
    print(f"Output Target: {output_dir}")
    print("-----------------------------------------------------------\n", flush=True)

    print("[1/4] Loading base model in FP16 on CPU RAM...", flush=True)
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

    print("[2/4] Attaching trained LoRA adapter weights...", flush=True)
    t1 = time.time()
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    print(f"LoRA attached in {time.time()-t1:.2f}s", flush=True)

    print("[3/4] Fusing weights: W = W_0 + (B * A) * (alpha / r)...", flush=True)
    t2 = time.time()
    merged_model = model.merge_and_unload()
    print(f"Weights fused successfully in {time.time()-t2:.2f}s!", flush=True)

    print(f"[4/4] Saving fused FP16 model to {output_dir}...", flush=True)
    t3 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(out_path), safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(str(out_path))
    print(f"Export completed in {time.time()-t3:.2f}s at: {out_path.resolve()}")
    print("\n[OK] Merge complete! Total model size: ~6.17 GB\n")
    print("===========================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA into FP16 HuggingFace Model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--lora-dir", type=str, default="E:/apex-coder/outputs/apex_coder_3b_lora_v3")
    parser.add_argument("--output-dir", type=str, default="E:/apex-coder/outputs/apex_coder_3b_merged")
    args = parser.parse_args()

    merge_lora_to_fp16(
        base_model_name=args.base_model,
        lora_dir=args.lora_dir,
        output_dir=args.output_dir,
    )
