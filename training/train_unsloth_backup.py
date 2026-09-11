import os
import sys
import time
import json
import math
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")


def get_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {peak:.2f}GB | Total: {total:.2f}GB"
    return "CUDA not active"


def resolve_model_path(model_name: str) -> str:
    unsloth_bnb_dir = Path("E:/hf_cache/hub/models--unsloth--Qwen2.5-Coder-3B-Instruct-bnb-4bit/snapshots/84f4805873850233fe5b2b156d3e3435425b4098")
    if unsloth_bnb_dir.exists() and "Qwen2.5-Coder-3B" in model_name:
        return str(unsloth_bnb_dir.resolve())

    snapshot_dir = Path("E:/hf_cache/hub/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5")
    if snapshot_dir.exists() and "Qwen2.5-Coder-3B" in model_name:
        return str(snapshot_dir.resolve())

    return model_name


def train_apex_model(
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    dataset_path: str = "E:/apex-coder/data/clean/train_2k.jsonl",
    output_dir: str = "E:/apex-coder/outputs/apex_coder_3b_lora",
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 1,
    learning_rate: float = 2e-4,
    grad_accum_steps: int = 4,
    save_every: int = 50,
):
    resolved_model = resolve_model_path(model_name)
    data_file = Path(dataset_path)

    print("===========================================================")
    print("      APEX CODER - 4GB VRAM GUARDED QLoRA TRAINING")
    print("===========================================================")
    print(f"Base Model: {resolved_model}")
    print(f"Dataset: {data_file}")
    print(f"Max Seq Length: {max_seq_length} tokens")
    print(f"LoRA Rank: {lora_r} | Alpha: {lora_alpha}")
    print(f"Effective Batch Size: {grad_accum_steps}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Initial GPU State: {get_gpu_memory_info()}")
    print("-----------------------------------------------------------\n", flush=True)

    # 1. Load Model & Tokenizer
    print("[1/4] Loading 4-bit base model into GPU memory...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        device_map="auto",
        dtype=torch.float16,
    )
    print(f"Model loaded in {time.time()-t0:.2f}s! Base VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

    # 2. Attach LoRA Adapters
    print("[2/4] Initializing LoRA adapters...", flush=True)
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()

    # 3. Load dataset samples
    print("[3/4] Loading dataset into memory...", flush=True)
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    total_samples = len(samples)
    total_steps = (total_samples * epochs) // grad_accum_steps
    print(f"Loaded {total_samples:,} verified samples. Total training steps: {total_steps:,}", flush=True)

    # Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    def get_lr(current_step):
        warmup_steps = 10
        step = current_step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # 4. Training Loop
    print("\n[4/4] Starting QLoRA fine-tuning training loop...", flush=True)
    start_train_time = time.time()
    accumulated_loss = 0.0
    global_step = 0

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---", flush=True)
        for i, sample in enumerate(samples):
            text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).to("cuda")

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

            del inputs, outputs, loss

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == total_samples:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_train_time
                pct = (global_step / total_steps) * 100
                vram_str = get_gpu_memory_info()

                print(
                    f"[Step {global_step:3d}/{total_steps:3d} ({pct:5.1f}%)] "
                    f"Loss: {accumulated_loss:6.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"GPU: {vram_str} | "
                    f"Elapsed: {elapsed:5.1f}s",
                    flush=True,
                )
                accumulated_loss = 0.0

                # Checkpoint saving
                if global_step % save_every == 0:
                    ckpt_dir = out_path / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True)
                    model.save_pretrained(str(ckpt_dir))
                    print(f"  -> Checkpoint saved at {ckpt_dir}", flush=True)

    # Final Save
    total_time = time.time() - start_train_time
    print(f"\n===========================================================")
    print(f"Training Complete in {total_time:.2f}s ({total_time/60:.2f} mins)!")
    print(f"Saving final LoRA adapter to {out_path.resolve()}...")
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"LoRA adapter successfully saved to: {out_path.resolve()}")
    print(f"===========================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apex Coder 4GB QLoRA Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="E:/apex-coder/data/clean/train_2k.jsonl")
    parser.add_argument("--output", type=str, default="E:/apex-coder/outputs/apex_coder_3b_lora")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=50)

    args = parser.parse_args()

    train_apex_model(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_seq_length=args.seq_len,
        lora_r=args.rank,
        lora_alpha=args.alpha,
        epochs=args.epochs,
        learning_rate=args.lr,
        grad_accum_steps=args.grad_accum,
        save_every=args.save_every,
    )
