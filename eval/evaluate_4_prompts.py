import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path("E:/apex-coder")
sys.path.insert(0, str(PROJECT_ROOT))

from run_chat import load_chat_model, generate_response, SYSTEM_PROMPT

def evaluate_all():
    prompts = [
        ("Test 1: Concurrency & Clock Drift (TokenBucket TS)",
         "Write a production-ready TokenBucket rate limiter class in TypeScript. It must support maxCapacity, refillRatePerSecond, and an async consume(tokens: number): Promise<boolean> method. Ensure it handles clock skew and timestamps safely using Date.now()."),
        
        ("Test 2: Thread-Safe Mutex & Queue (AsyncMutex TS)",
         "Implement an AsyncMutex class in TypeScript with acquire() and release() methods. Ensure it prevents race conditions when multiple concurrent async tasks attempt to enter the critical section, and guarantees FIFO ordering for pending lock requests."),
        
        ("Test 3: Complex Binary Search DSA (min_days_to_bloom Python)",
         "Implement a function in Python: `min_days_to_bloom(bloom_day: list[int], m: int, k: int) -> int` to find the minimum number of days needed to make m bouquets of k adjacent flowers. Write complete type hints and an optimal binary search approach. Include pytest test cases covering edge cases (e.g., impossible scenario returning -1)."),
        
        ("Test 4: Node.js v24 ESM Compatibility (Request Aggregator TS)",
         "Create a sliding window request aggregator class in TypeScript designed to run directly in Node.js v24 under standard strip-only ESM mode without a build step. Use node:test for the assertion suite.")
    ]

    model, tokenizer = load_chat_model()
    results = []

    for i, (title, prompt) in enumerate(prompts, 1):
        print(f"\n===========================================================")
        print(f"[{i}/4] {title}")
        print(f"===========================================================")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        t0 = time.time()
        resp = generate_response(model, tokenizer, messages, stream=True, max_new_tokens=220)
        dur = time.time() - t0
        print(f"\n[Generated in {dur:.2f}s | Length: {len(resp)} chars]\n")
        results.append({"title": title, "prompt": prompt, "response": resp, "duration_sec": round(dur, 2)})

    out_file = PROJECT_ROOT / "eval" / "four_guardrails_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Results successfully saved to {out_file}")

if __name__ == "__main__":
    evaluate_all()
