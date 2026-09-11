import os
import re

path = r"E:\apex-coder\training\train_unsloth.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add PeftModel import
code = code.replace(
    "from peft import LoraConfig, get_peft_model",
    "from peft import LoraConfig, get_peft_model, PeftModel"
)

# 2. Add resume CLI argument
code = code.replace(
    'parser.add_argument("--save-every", type=int, default=50)',
    'parser.add_argument("--save-every", type=int, default=50)\n    parser.add_argument("--resume", type=str, default=None)'
)

# 3. Update train function signature and call
code = code.replace(
    "save_every: int = 50,\n):",
    "save_every: int = 50,\n    resume_checkpoint: str = None,\n):"
)
code = code.replace(
    "save_every=args.save_every,\n    )",
    "save_every=args.save_every,\n        resume_checkpoint=args.resume,\n    )"
)

# 4. Handle checkpoint loading logic
old_lora_block = """    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()"""

new_lora_block = """    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[*] Resuming LoRA adapters from: {resume_checkpoint}", flush=True)
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()"""

code = code.replace(old_lora_block, new_lora_block)

# 5. Skip processed samples and adjust global_step
old_loop_init = """    # 4. Training Loop
    print("\\n[4/4] Starting QLoRA fine-tuning training loop...", flush=True)
    start_train_time = time.time()
    accumulated_loss = 0.0
    global_step = 0"""

new_loop_init = """    # 4. Training Loop
    print("\\n[4/4] Starting QLoRA fine-tuning training loop...", flush=True)
    start_train_time = time.time()
    accumulated_loss = 0.0
    global_step = 0
    if resume_checkpoint:
        match = re.search(r"checkpoint-(\\d+)", str(resume_checkpoint))
        if match:
            start_step = int(match.group(1))
            skip_samples = start_step * grad_accum_steps
            samples = samples[skip_samples:]
            global_step = start_step
            print(f"[*] Resuming from step {global_step} (skipped {skip_samples} samples)", flush=True)"""

code = code.replace(old_loop_init, new_loop_init)

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("SUCCESS: Patch applied cleanly!")
