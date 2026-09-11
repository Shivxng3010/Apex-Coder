import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_path = r"E:\hf_cache\hub\models--unsloth--Qwen2.5-Coder-3B-Instruct-bnb-4bit\snapshots\84f4805873850233fe5b2b156d3e3435425b4098"
lora_path = r"E:\apex-coder\outputs\apex_coder_3b_lora\checkpoint-50"

print("\n[*] Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

print("[*] Loading Base Model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="cuda:0"
)

print("[*] Attaching Checkpoint-50 LoRA weights...")
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Write a Python function `two_sum(nums, target)` that returns the indices of the two numbers such that they add up to target. Use an optimal O(n) hash map approach.

### Response:
"""

print("[*] Generating response...\n")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

print("=" * 60)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("=" * 60)
