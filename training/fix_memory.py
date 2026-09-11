path = r"E:\apex-coder\training\train_unsloth.py"
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

old_step = "optimizer.zero_grad(set_to_none=True)"
new_step = "optimizer.zero_grad(set_to_none=True)\n            torch.cuda.empty_cache()"

if old_step in code and "torch.cuda.empty_cache()" not in code:
    code = code.replace(old_step, new_step)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("SUCCESS: Memory guard patched!")
else:
    print("Already patched or pattern not found.")
