path = r"E:\apex-coder\training\train_unsloth.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # strip out any malformed empty_cache line injected earlier
    if "torch.cuda.empty_cache()" in line:
        continue
    new_lines.append(line)
    if "optimizer.zero_grad(set_to_none=True)" in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(" " * indent + "torch.cuda.empty_cache()\n")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("SUCCESS: Indentation fixed!")
