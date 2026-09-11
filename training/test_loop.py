import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_path = "E:/hf_cache/hub/models--unsloth--Qwen2.5-Coder-3B-Instruct-bnb-4bit/snapshots/84f4805873850233fe5b2b156d3e3435425b4098"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.float16)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.train()

samples = []
with open("E:/apex-coder/data/clean/train_2k.jsonl", "r", encoding="utf-8") as f:
    for _ in range(30):
        samples.append(json.loads(f.readline()))

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

print("Starting 30-sample test loop with explicit memory cleanup...")
t_start = time.time()

for i, s in enumerate(samples):
    t0 = time.time()
    text = tokenizer.apply_chat_template(s["messages"], tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    dt = time.time() - t0
    vram = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Sample {i+1:2d}/30 ({seq_len:3d} tok) | Loss: {loss.item():.4f} | Time: {dt:.2f}s | VRAM: {vram:.2f}GB | Peak: {peak:.2f}GB", flush=True)
    
    del inputs, outputs, loss

print(f"30 samples completed in {time.time()-t_start:.2f}s (avg: {(time.time()-t_start)/30:.2f}s/sample)!")
