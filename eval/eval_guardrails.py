import os
import sys
import json
import time
import re
from pathlib import Path

PROJECT_ROOT = Path("E:/apex-coder")
sys.path.insert(0, str(PROJECT_ROOT))

from run_chat import load_chat_model, generate_response, SYSTEM_PROMPT
from harness.self_correct import SelfCorrectionEngine, AutoVerifyResult

TEST_CASES = [
    {
        "id": "test_1_token_bucket",
        "title": "Test 1: Concurrency & Clock Drift (TypeScript)",
        "language": "typescript",
        "prompt": (
            "Write a production-ready TokenBucket rate limiter class in TypeScript. "
            "It must support maxCapacity, refillRatePerSecond, and an async consume(tokens: number): Promise<boolean> method. "
            "Ensure it handles clock skew and timestamps safely using Date.now()."
        ),
        "checkpoints": [
            ("Thinking CoT Block Generated", lambda t: "<thinking>" in t and "</thinking>" in t),
            ("Stateful Class Declared", lambda t: "class TokenBucket" in t or "class " in t),
            ("Date.now() Clamping / Usage", lambda t: "Date.now()" in t),
            ("Class Properties Maintained", lambda t: "tokens" in t and "refill" in t.lower()),
        ]
    },
    {
        "id": "test_2_async_mutex",
        "title": "Test 2: Thread-Safe Mutex & Queue Handling (TypeScript)",
        "language": "typescript",
        "prompt": (
            "Implement an AsyncMutex class in TypeScript with acquire() and release() methods. "
            "Ensure it prevents race conditions when multiple concurrent async tasks attempt to enter the critical section, "
            "and guarantees FIFO ordering for pending lock requests."
        ),
        "checkpoints": [
            ("Thinking CoT Block Generated", lambda t: "<thinking>" in t and "</thinking>" in t),
            ("AsyncMutex Class Declared", lambda t: "class AsyncMutex" in t or "class " in t),
            ("FIFO Queue / Resolvers Handled", lambda t: "queue" in t.lower() or "resolve" in t.lower()),
            ("acquire and release methods present", lambda t: "acquire" in t and "release" in t),
        ]
    },
    {
        "id": "test_3_bloom_bouquets",
        "title": "Test 3: Complex DSA & Edge Cases (Python)",
        "language": "python",
        "prompt": (
            "Implement a function in Python: `min_days_to_bloom(bloom_day: list[int], m: int, k: int) -> int` "
            "to find the minimum number of days needed to make m bouquets of k adjacent flowers. "
            "Write complete type hints and an optimal binary search approach. "
            "Include pytest test cases covering edge cases (e.g., impossible scenario returning -1)."
        ),
        "checkpoints": [
            ("Thinking CoT Block Generated", lambda t: "<thinking>" in t and "</thinking>" in t),
            ("Function min_days_to_bloom defined", lambda t: "def min_days_to_bloom" in t),
            ("Binary Search Logic Used", lambda t: "while" in t and ("mid" in t or "left" in t)),
            ("Impossible Case Handled (m * k)", lambda t: "m * k" in t or "-1" in t),
        ]
    },
    {
        "id": "test_4_node_esm",
        "title": "Test 4: Node.js v24 ESM Compatibility Check (TypeScript)",
        "language": "typescript",
        "prompt": (
            "Create a sliding window request aggregator class in TypeScript designed to run directly in Node.js v24 "
            "under standard strip-only ESM mode without a build step. Use node:test for the assertion suite."
        ),
        "checkpoints": [
            ("Thinking CoT Block Generated", lambda t: "<thinking>" in t and "</thinking>" in t),
            ("Native node:test Imported", lambda t: "node:test" in t),
            ("Explicit Class Body Fields", lambda t: "class " in t and "{" in t),
            ("Sliding Window Pruning Logic", lambda t: "filter" in t or "shift" in t or "window" in t.lower()),
        ]
    }
]


def run_guardrails_eval(output_file: str = "E:/apex-coder/eval/guardrails_report.json"):
    print("===========================================================")
    print("      APEX CODER - 4 GUARDRAIL PROMPTS EVALUATION          ")
    print("===========================================================\n", flush=True)

    model, tokenizer = load_chat_model()
    verifier = SelfCorrectionEngine(timeout=5)

    report_results = []
    total_checkpoints_passed = 0
    total_checkpoints_count = 0
    sandbox_passed_count = 0

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/4] Executing {test['title']}...")
        print(f"Prompt: {test['prompt'][:85]}...\n", flush=True)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": test["prompt"]}
        ]

        t0 = time.time()
        response = generate_response(model, tokenizer, messages, stream=False, max_new_tokens=280)
        gen_duration = time.time() - t0
        print(response, flush=True)

        # Run Sandbox Verification
        verify_res: AutoVerifyResult = verifier.verify_output(response)
        sandbox_passed = verify_res.is_valid

        print(f"\n\n--- Checkpoint Inspections for {test['id']} ---")
        checkpoint_results = []
        for name, check_fn in test["checkpoints"]:
            total_checkpoints_count += 1
            passed = bool(check_fn(response))
            if passed:
                total_checkpoints_passed += 1
            mark = "PASS" if passed else "FAIL"
            print(f"  [{mark}] {name}")
            checkpoint_results.append({"name": name, "passed": passed})

        sandbox_status = "PASSED" if sandbox_passed else ("NO_TESTS" if not verify_res.has_tests else "FAILED")
        if sandbox_passed:
            sandbox_passed_count += 1
        print(f"  [SANDBOX HARNESS] Result: {sandbox_status} ({verify_res.duration_seconds:.2f}s)")
        if not sandbox_passed and verify_res.error_message:
            print(f"  [ERROR] {verify_res.error_message[:120]}")

        report_results.append({
            "id": test["id"],
            "title": test["title"],
            "prompt": test["prompt"],
            "generation_time_sec": round(gen_duration, 2),
            "response": response,
            "checkpoints": checkpoint_results,
            "sandbox_status": sandbox_status,
            "sandbox_duration_sec": round(verify_res.duration_seconds, 2),
            "sandbox_error": verify_res.error_message,
        })

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.time(),
        "total_tests": len(TEST_CASES),
        "total_checkpoints": total_checkpoints_count,
        "passed_checkpoints": total_checkpoints_passed,
        "checkpoint_pass_rate_pct": round((total_checkpoints_passed / max(1, total_checkpoints_count)) * 100, 1),
        "sandbox_passed_count": sandbox_passed_count,
        "sandbox_pass_rate_pct": round((sandbox_passed_count / max(1, len(TEST_CASES))) * 100, 1),
        "results": report_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n===========================================================")
    print(f"EVALUATION COMPLETE:")
    print(f"Checkpoints Verified : {total_checkpoints_passed}/{total_checkpoints_count} ({summary['checkpoint_pass_rate_pct']}%)")
    print(f"Sandbox Verified     : {sandbox_passed_count}/{len(TEST_CASES)} ({summary['sandbox_pass_rate_pct']}%)")
    print(f"Report saved to      : {out_path.resolve()}")
    print("===========================================================\n", flush=True)


if __name__ == "__main__":
    run_guardrails_eval()
