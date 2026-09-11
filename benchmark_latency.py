import time
from pathlib import Path

MODEL_PATH = r"E:\apex-coder\models\apex-coder-3b-q4_k_m.gguf"

try:
    from llama_cpp import Llama
    print("[*] Loading Q4_K_M GGUF model into memory...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # All layers to GPU
        n_ctx=1024,           # Active context window
        verbose=False
    )

    prompt = """<|im_start|>system
You are Apex Coder. Analyze race conditions first in <thinking> blocks.<|im_end|>
<|im_start|>user
Write a thread-safe AsyncMutex class in TypeScript with acquire() and release() methods.<|im_end|>
<|im_start|>assistant
<thinking>"""

    print("\n--- Running Latency & Generation Test ---")
    t0 = time.perf_counter()

    first_token_time = None
    token_count = 0

    stream = llm(
        prompt,
        max_tokens=200,
        temperature=0.2,
        stop=["<|im_end|>"],
        stream=True
    )

    generated_text = ""
    for chunk in stream:
        token = chunk["choices"][0]["text"]
        if token_count == 0:
            first_token_time = time.perf_counter() - t0
        generated_text += token
        token_count += 1

    total_time = time.perf_counter() - t0
    eval_generation_time = total_time - first_token_time if first_token_time else total_time
    tok_per_sec = (token_count - 1) / eval_generation_time if eval_generation_time > 0 else 0

    print(f"Time to First Token (TTFT) : {first_token_time * 1000:.2f} ms" if first_token_time else "TTFT: N/A")
    print(f"Tokens Generated           : {token_count}")
    print(f"Total Duration             : {total_time:.2f} s")
    print(f"Generation Speed           : {tok_per_sec:.2f} tokens/sec")
    print("\n--- Output Sample ---")
    print("<thinking>" + generated_text)

except ImportError:
    print("[!] 'llama_cpp' Python module not installed in current virtualenv.")
    print("    Use the native C++ 'llama-bench' results below or install via: uv pip install llama-cpp-python")
