import os
import sys
import json
import zipfile
import time
from pathlib import Path

PROJECT_ROOT = Path("E:/apex-coder")
EXPORT_DIR = PROJECT_ROOT / "export" / "cloud_bundles"


def create_cloud_training_script() -> str:
    return """# Apex Coder - Cloud QLoRA Full-Reasoning Retraining Script (RunPod / Google Colab / Vast.ai)
# Requires: pip install torch transformers peft datasets bitsandbytes accelerate trl

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DATASET_PATH = "train_bundle.jsonl"
OUTPUT_DIR = "apex_coder_3b_master_adapter"
MAX_SEQ_LENGTH = 2048
EPOCHS = 3
LR = 1e-4

print(f"[*] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Cloud GPU (24GB+ VRAM): Full FP16/BF16 loading with FlashAttention/SDPA
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=compute_dtype,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

# Target All Linear Layers for Deep Multi-Step Reasoning & Code Architecture
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.train()

samples = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"[*] Training on {len(samples)} consolidated samples for {EPOCHS} epochs (seq_len={MAX_SEQ_LENGTH})...")
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

grad_accum_steps = 4
global_step = 0
for epoch in range(EPOCHS):
    for i, s in enumerate(samples):
        text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True).to("cuda")
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(samples):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 25 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step} | Loss: {loss.item() * grad_accum_steps:.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[✓] Cloud deep training complete! Master LoRA adapter saved to {OUTPUT_DIR}")
"""


def create_cloud_dpo_script() -> str:
    return """# Apex Coder - Cloud Direct Preference Optimization (DPO) Script
# Aligns model to eliminate hallucinations using chosen (verified) vs rejected (broken) pairs

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DPO_DATASET_PATH = "dpo_bundle.jsonl"
OUTPUT_DIR = "apex_coder_3b_dpo_adapter"

if not os.path.exists(DPO_DATASET_PATH):
    print("[-] No DPO dataset found. Skipping DPO training.")
    exit(0)

pairs = []
with open(DPO_DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            pairs.append({
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"]
            })

if len(pairs) < 2:
    print("[-] Insufficient DPO pairs (< 2). Collect more self-correction runs first.")
    exit(0)

print(f"[*] Loaded {len(pairs)} DPO preference pairs.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=compute_dtype,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=True,
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

dpo_dataset = Dataset.from_list(pairs)
training_args = DPOConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    max_length=2048,
    max_prompt_length=1024,
)

print("[*] Starting Cloud DPO Alignment...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"[✓] DPO Alignment complete! Saved to {OUTPUT_DIR}")
"""


def export_bundle() -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle_id = f"apex_cloud_bundle_{int(time.time())}"
    bundle_dir = EXPORT_DIR / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all SFT dataset sources
    combined_samples = []
    sources = [
        PROJECT_ROOT / "data" / "clean" / "train_2k.jsonl",
        PROJECT_ROOT / "data" / "clean" / "concurrency_300.jsonl",
        PROJECT_ROOT / "data" / "experience_pool" / "live_gold.jsonl",
        PROJECT_ROOT / "data" / "experience_pool" / "consumed_history.jsonl",
    ]

    for p in sources:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            combined_samples.append(json.loads(line))
                        except Exception:
                            pass

    train_data_file = bundle_dir / "train_bundle.jsonl"
    with open(train_data_file, "w", encoding="utf-8") as f:
        for s in combined_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 2. Collect DPO Preference Pairs
    dpo_samples = []
    dpo_source = PROJECT_ROOT / "data" / "experience_pool" / "live_dpo_pairs.jsonl"
    if dpo_source.exists():
        with open(dpo_source, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        dpo_samples.append(json.loads(line))
                    except Exception:
                        pass

    dpo_data_file = bundle_dir / "dpo_bundle.jsonl"
    with open(dpo_data_file, "w", encoding="utf-8") as f:
        for d in dpo_samples:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 3. Write training scripts
    script_sft_file = bundle_dir / "train_cloud.py"
    with open(script_sft_file, "w", encoding="utf-8") as f:
        f.write(create_cloud_training_script())

    script_dpo_file = bundle_dir / "train_dpo_cloud.py"
    with open(script_dpo_file, "w", encoding="utf-8") as f:
        f.write(create_cloud_dpo_script())

    # 4. Create zip archive
    zip_path = EXPORT_DIR / f"{bundle_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(train_data_file, arcname="train_bundle.jsonl")
        z.write(dpo_data_file, arcname="dpo_bundle.jsonl")
        z.write(script_sft_file, arcname="train_cloud.py")
        z.write(script_dpo_file, arcname="train_dpo_cloud.py")

    print(f"===========================================================")
    print(f"      APEX CODER - CLOUD BULK RETRAINING BUNDLE")
    print(f"===========================================================")
    print(f"Total SFT Samples     : {len(combined_samples):,} samples")
    print(f"Total DPO Pairs       : {len(dpo_samples):,} pairs")
    print(f"Bundle Directory      : {bundle_dir.resolve()}")
    print(f"ZIP Archive           : {zip_path.resolve()} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Usage:")
    print(f"  1. Upload {zip_path.name} to Google Colab / RunPod / Vast.ai")
    print(f"  2. Run SFT: python train_cloud.py")
    print(f"  3. Run DPO: python train_dpo_cloud.py")
    print(f"===========================================================\n")
    return zip_path
    print(f"      APEX CODER - CLOUD BULK RETRAINING BUNDLE")
    print(f"===========================================================")
    print(f"Total Dataset Samples : {len(combined_samples):,} samples")
    print(f"Bundle Directory      : {bundle_dir.resolve()}")
    print(f"ZIP Archive           : {zip_path.resolve()} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
    print(f"Usage:")
    print(f"  1. Upload {zip_path.name} to Google Colab / RunPod / Vast.ai")
    print(f"  2. Unzip and run: python train_cloud.py")
    print(f"===========================================================\n")
    return zip_path


if __name__ == "__main__":
    export_bundle()
