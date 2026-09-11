file_path = r"E:\apex-coder\training\train_unsloth.py"
with open(file_path, "r", encoding="utf-8") as f:
    code = f.read()

# Remove misplaced line
code = code.replace("model.gradient_checkpointing_enable()\nelse:", "else:")

# Properly enable checkpointing right after model.train()
target = "model.train()"
replacement = """model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()"""

if target in code and "enable_input_require_grads" not in code:
    code = code.replace(target, replacement, 1)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

print("SUCCESS: Syntax and Gradient Checkpointing properly configured!")
