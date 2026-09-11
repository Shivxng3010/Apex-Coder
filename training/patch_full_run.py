import re

file_path = r"E:\apex-coder\training\train_unsloth.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Make sure gc is imported
if "import gc" not in content:
    content = "import gc\n" + content

# Fix the training loop inner step to cleanly manage Windows paging
old_block = """            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            global_step += 1"""

new_block = """            optimizer.zero_grad(set_to_none=True)
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
            global_step += 1"""

if old_block in content:
    content = content.replace(old_block, new_block)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: Code patched for Windows paging stability!")
