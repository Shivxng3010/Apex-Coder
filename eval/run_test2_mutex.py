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

PROMPT = (
    "Implement an AsyncMutex class in TypeScript with acquire() and release() methods. "
    "Ensure it prevents race conditions when multiple concurrent async tasks attempt to enter the critical section, "
    "and guarantees FIFO ordering for pending lock requests."
)

def run_test2():
    print("===========================================================")
    print("   TEST 2: ASYNCMUTEX & FIFO QUEUE HANDLING (TYPESCRIPT)   ")
    print("===========================================================")
    print(f"Prompt:\n{PROMPT}\n")
    print("-----------------------------------------------------------", flush=True)

    model, tokenizer = load_chat_model()
    verifier = SelfCorrectionEngine(timeout=10)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT}
    ]

    print("\n--- MODEL GENERATION START ---", flush=True)
    t0 = time.time()
    response = generate_response(model, tokenizer, messages, stream=True, max_new_tokens=600)
    gen_duration = time.time() - t0
    print("\n--- MODEL GENERATION END ---", flush=True)
    print(f"Generated in {gen_duration:.2f}s\n", flush=True)

    # Inspect Checkpoint 1: Promise Resolvers in Queue
    # Typical patterns: queue: (() => void)[], new Promise(res => queue.push(res)), Promise<void>
    has_queue = bool(re.search(r"queue\s*:\s*(?:\(?\s*\(\s*\)\s*=>\s*void\)?\[\]|Array<[^>]+>)", response, re.I)) or ("queue.push" in response and "resolve" in response)
    has_promise_resolver = ("Promise" in response) and ("resolve" in response or "push" in response)
    cp1_passed = has_queue or has_promise_resolver

    # Inspect Checkpoint 2: isLocked / locked flag strictly managed
    has_lock_flag = bool(re.search(r"(?:isLocked|locked|_locked)\s*[:=]", response))
    has_lock_check = bool(re.search(r"if\s*\(\s*!?this\.(?:isLocked|locked|_locked)", response))
    has_release_reset = ("this.locked = false" in response) or ("this._locked = false" in response) or ("this.isLocked = false" in response) or ("queue.shift()" in response)
    cp2_passed = has_lock_flag and (has_lock_check or has_release_reset)

    # Sandbox Verification
    print("-----------------------------------------------------------")
    print("SANDBOX HARNESS VERIFICATION:")
    verify_res: AutoVerifyResult = verifier.verify_output(response)
    print(f"Status      : {verify_res.status.upper()}")
    print(f"Duration    : {verify_res.duration_seconds:.2f}s")
    if verify_res.stdout:
        print(f"Stdout      :\n{verify_res.stdout}")
    if verify_res.stderr:
        print(f"Stderr      :\n{verify_res.stderr}")
    if verify_res.error_message:
        print(f"Error       : {verify_res.error_message}")

    print("-----------------------------------------------------------")
    print("CHECKPOINTS INSPECTION:")
    print(f"  Checkpoint 1 (Promise Resolvers Queue) : {'PASS' if cp1_passed else 'FAIL'}")
    print(f"  Checkpoint 2 (Lock Flag / State Reset) : {'PASS' if cp2_passed else 'FAIL'}")
    print("===========================================================\n", flush=True)

    out_file = PROJECT_ROOT / "eval" / "test2_mutex_report.json"
    report = {
        "prompt": PROMPT,
        "generation_time_sec": round(gen_duration, 2),
        "response": response,
        "checkpoint_1_resolvers_queue": cp1_passed,
        "checkpoint_2_lock_flag_managed": cp2_passed,
        "sandbox_valid": verify_res.is_valid,
        "sandbox_status": verify_res.status,
        "sandbox_stdout": verify_res.stdout,
        "sandbox_stderr": verify_res.stderr,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run_test2()
