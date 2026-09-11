import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path("E:/apex-coder")
sys.path.insert(0, str(PROJECT_ROOT))

from run_chat import load_chat_model, generate_response, SYSTEM_PROMPT

def test_single_prompt(prompt_idx: int):
    prompts = [
        # Prompt 1
        "Write a production-ready TokenBucket rate limiter class in TypeScript. It must support maxCapacity, refillRatePerSecond, and an async consume(tokens: number): Promise<boolean> method. Ensure it handles clock skew and timestamps safely using Date.now().",
        # Prompt 2
        "Implement an AsyncMutex class in TypeScript with acquire() and release() methods. Ensure it prevents race conditions when multiple concurrent async tasks attempt to enter the critical section, and guarantees FIFO ordering for pending lock requests.",
        # Prompt 3
        "Implement a function in Python: `min_days_to_bloom(bloom_day: list[int], m: int, k: int) -> int` to find the minimum number of days needed to make m bouquets of k adjacent flowers. Write complete type hints and an optimal binary search approach. Include pytest test cases covering edge cases (e.g., impossible scenario returning -1).",
        # Prompt 4
        "Create a sliding window request aggregator class in TypeScript designed to run directly in Node.js v24 under standard strip-only ESM mode without a build step. Use node:test for the assertion suite."
    ]

    model, tokenizer = load_chat_model()
    idx = prompt_idx - 1
    prompt = prompts[idx]

    print(f"\n===========================================================")
    print(f"   EVALUATING TEST PROMPT {prompt_idx}/4")
    print(f"===========================================================")
    print(f"Prompt: {prompt}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    t0 = time.time()
    response = generate_response(model, tokenizer, messages, stream=True, max_new_tokens=300)
    dur = time.time() - t0

    print(f"\n\nGeneration Time: {dur:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    p_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    test_single_prompt(p_idx)
