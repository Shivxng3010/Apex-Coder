file_path = r"E:\apex-coder\training\train_unsloth.py"
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Enable gradient checkpointing on the model
target = "model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)"
replacement = "model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)\n    model.gradient_checkpointing_enable()"

if target in text and "gradient_checkpointing_enable" not in text:
    text = text.replace(target, replacement)
elif "gradient_checkpointing_enable" not in text:
    # If resuming block wasn't exactly matched, enable it right after LoRA init
    text = text.replace("model.train()", "model.gradient_checkpointing_enable()\n    model.train()")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("SUCCESS: Gradient Checkpointing enabled!")
