import gc
import os
import re
import sys
import time
import json
import math
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

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


def compute_sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels_mask: torch.Tensor) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = labels_mask[..., 1:].contiguous()

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    per_token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

    seq_logps = (per_token_logps * shift_mask).sum(dim=-1)
    return seq_logps


def train_dpo_apex_model(
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
    dpo_dataset_path: str = "E:/apex-coder/data/experience_pool/live_dpo_pairs.jsonl",
    output_dir: str = "E:/apex-coder/outputs/apex_coder_3b_lora_dpo",
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    epochs: int = 2,
    beta: float = 0.1,
    learning_rate: float = 5e-5,
    grad_accum_steps: int = 2,
    adapt_from: str = None,
):
    resolved_model = resolve_model_path(model_name)
    data_file = Path(dpo_dataset_path)

    print("===========================================================")
    print("    APEX CODER - 4GB VRAM GUARDED DPO PREFERENCE TRAINING")
    print("===========================================================")
    print(f"Base Model        : {resolved_model}")
    print(f"DPO Dataset       : {data_file}")
    print(f"Max Seq Length    : {max_seq_length} tokens")
    print(f"LoRA Rank / Alpha : {lora_r} / {lora_alpha}")
    print(f"DPO Beta (Scale)  : {beta}")
    print(f"Device            : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Initial GPU State : {get_gpu_memory_info()}")
    print("-----------------------------------------------------------\n", flush=True)

    if not data_file.exists():
        print(f"[!] DPO dataset not found: {data_file}")
        return False, "DPO dataset file not found"

    dpo_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    dpo_samples.append(json.loads(line))
                except Exception:
                    pass

    if not dpo_samples:
        print("[!] No valid DPO samples found in dataset.")
        return False, "No DPO samples found"

    # 1. Load Model & Tokenizer
    print("[1/4] Loading 4-bit base model into GPU memory...", flush=True)
    t0 = time.time()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

    tokenizer = AutoTokenizer.from_pretrained(resolved_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model,
        device_map=device_map,
        dtype=torch.float16,
        attn_implementation="sdpa",
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

    if adapt_from and os.path.exists(adapt_from):
        print(f"[*] Initializing DPO from active LoRA adapter: {adapt_from}", flush=True)
        model = PeftModel.from_pretrained(model, adapt_from, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 3. Setup Optimizer & Cosine Scheduler
    total_samples = len(dpo_samples)
    total_steps = max(1, (total_samples * epochs) // grad_accum_steps)
    print(f"[3/4] Loaded {total_samples} DPO pairs. Total training steps: {total_steps}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    def get_lr(current_step):
        warmup_steps = max(1, min(5, total_steps // 4))
        step = current_step + 1
        if step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr)

    # 4. DPO Training Loop
    print("\n[4/4] Starting VRAM-Guarded DPO Training Loop...", flush=True)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    start_train_time = time.time()
    accumulated_loss = 0.0
    accumulated_implicit_acc = 0.0
    global_step = 0

    def encode_chat(messages, max_len=max_seq_length):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

        full_tok = tokenizer(full_text, return_tensors="pt", max_length=max_len, truncation=True)
        prompt_tok = tokenizer(prompt_text, return_tensors="pt", max_length=max_len, truncation=True)

        input_ids = full_tok["input_ids"].to("cuda")
        attention_mask = full_tok["attention_mask"].to("cuda")

        prompt_len = min(prompt_tok["input_ids"].shape[1], input_ids.shape[1])
        labels_mask = torch.ones_like(input_ids)
        labels_mask[:, :prompt_len] = 0

        return input_ids, attention_mask, labels_mask

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---", flush=True)
        for i, sample in enumerate(dpo_samples):
            chosen_msgs = sample["messages_chosen"]
            rejected_msgs = sample["messages_rejected"]

            c_ids, c_mask, c_lmask = encode_chat(chosen_msgs)
            r_ids, r_mask, r_lmask = encode_chat(rejected_msgs)

            # Step A: Compute Policy Logprobs (LoRA Active)
            model.train()
            policy_chosen_logps = compute_sequence_logprobs(model, c_ids, c_mask, c_lmask)
            policy_rejected_logps = compute_sequence_logprobs(model, r_ids, r_mask, r_lmask)

            # Clear cache between passes to ensure <= 3.6GB VRAM ceiling
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step B: Compute Reference Logprobs (LoRA Disabled via context manager)
            with torch.no_grad():
                with model.disable_adapter():
                    ref_chosen_logps = compute_sequence_logprobs(model, c_ids, c_mask, c_lmask)
                    ref_rejected_logps = compute_sequence_logprobs(model, r_ids, r_mask, r_lmask)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step C: Compute DPO Loss & Implicit Accuracy
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = beta * (pi_logratios - ref_logratios)

            loss = -torch.nn.functional.logsigmoid(logits).mean() / grad_accum_steps
            loss.backward()

            implicit_acc = (logits > 0).float().mean().item()
            accumulated_loss += loss.item() * grad_accum_steps
            accumulated_implicit_acc += implicit_acc

            del c_ids, c_mask, c_lmask, r_ids, r_mask, r_lmask
            del policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, logits, loss

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == total_samples:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_train_time
                pct = (global_step / total_steps) * 100
                vram_str = get_gpu_memory_info()

                print(
                    f"[DPO Step {global_step:2d}/{total_steps:2d} ({pct:5.1f}%)] "
                    f"Loss: {accumulated_loss:6.4f} | "
                    f"ImpAcc: {accumulated_implicit_acc / grad_accum_steps:4.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"GPU: {vram_str} | "
                    f"Elapsed: {elapsed:5.1f}s",
                    flush=True,
                )
                accumulated_loss = 0.0
                accumulated_implicit_acc = 0.0

    print("\n[*] Saving final fine-tuned DPO LoRA adapter...", flush=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    print(f"[+] DPO Training Complete! Adapter saved to: {out_path.resolve()}\n", flush=True)

    del model, tokenizer, optimizer, scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True, str(out_path.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4GB VRAM Guarded DPO Fine-Tuning")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="E:/apex-coder/data/experience_pool/live_dpo_pairs.jsonl")
    parser.add_argument("--output_dir", type=str, default="E:/apex-coder/outputs/apex_coder_3b_lora_dpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--adapt_from", type=str, default=None)
    args = parser.parse_args()

    train_dpo_apex_model(
        model_name=args.model,
        dpo_dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        adapt_from=args.adapt_from,
    )
