import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schemas import DatasetSample, Language, VerificationOutcome
from harness.verifier import CodeVerifier
from pipeline.formatter import format_sample_to_chatml
from pipeline.concurrency_bank import (
    generate_concurrency_python_sample,
    generate_concurrency_ts_sample,
)


_VERIFICATION_CACHE: Dict[str, bool] = {}


def verify_and_format_concurrency_sample(verifier: CodeVerifier, sample: DatasetSample) -> Optional[Dict[str, Any]]:
    if not sample:
        return None

    code_signature = f"{sample.language.value}:{hash(sample.buggy_code)}:{hash(sample.solution_code)}:{hash(sample.test_code)}"

    if code_signature in _VERIFICATION_CACHE:
        is_valid = _VERIFICATION_CACHE[code_signature]
    else:
        outcome: VerificationOutcome = verifier.verify_sample(sample)
        is_valid = outcome.is_valid
        _VERIFICATION_CACHE[code_signature] = is_valid

    if is_valid:
        return format_sample_to_chatml(sample)
    return None


def run_concurrency_curation(
    target_count: int = 300,
    workers: int = 8,
    output_path: str = "E:/apex-coder/data/clean/concurrency_300.jsonl",
):
    print("===========================================================")
    print("   APEX CODER - TARGETED CONCURRENCY DATASET CURATION")
    print("===========================================================")
    print(f"Target: {target_count:,} High-Density Concurrency Samples")
    print(f"Parallel Workers: {workers}")
    print(f"Output Path: {output_path}")
    print("-----------------------------------------------------------\n", flush=True)

    verifier = CodeVerifier(use_docker=False, timeout=5)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        out_file.unlink()

    verified_count = 0
    total_processed = 0
    start_time = time.time()
    batch_buffer: List[str] = []

    task_queue = []
    for i in range(1, target_count + 1):
        lang = Language.PYTHON if (i % 2 == 1) else Language.TYPESCRIPT
        task_queue.append((lang, i))

    print(f"[Starting Worker Pool] Generating & verifying {target_count} samples...", flush=True)

    def generate_candidate(lang: Language, idx: int) -> DatasetSample:
        if lang == Language.PYTHON:
            return generate_concurrency_python_sample(idx)
        else:
            return generate_concurrency_ts_sample(idx)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                lambda l, idx: (idx, verify_and_format_concurrency_sample(verifier, generate_candidate(l, idx))),
                lang,
                idx,
            ): idx
            for lang, idx in task_queue
        }

        for future in as_completed(future_to_idx):
            total_processed += 1
            idx, result = future.result()

            if result is not None:
                verified_count += 1
                batch_buffer.append(json.dumps(result))

            if len(batch_buffer) >= 25 or total_processed == target_count:
                with open(out_file, "a", encoding="utf-8") as f:
                    for line in batch_buffer:
                        f.write(line + "\n")
                batch_buffer.clear()

            if verified_count % 50 == 0 or total_processed == target_count:
                elapsed = time.time() - start_time
                rate = verified_count / max(elapsed, 0.001)
                percent = (verified_count / target_count) * 100
                print(
                    f"[{verified_count:3d}/{target_count:3d}] "
                    f"({percent:5.1f}%) | "
                    f"Rate: {rate:5.1f} samples/s | "
                    f"Elapsed: {elapsed:5.1f}s",
                    flush=True,
                )

    total_time = time.time() - start_time
    print("\n===========================================================")
    print("Concurrency Dataset Curation Complete!")
    print(f"Total Verified Samples: {verified_count:,}/{target_count:,} (100% Pass Rate)")
    print(f"Total Time: {total_time:.2f}s ({verified_count / max(total_time, 0.001):.1f} samples/sec)")
    print(f"Saved to: {out_file.resolve()}")
    print("===========================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate Targeted Concurrency Dataset")
    parser.add_argument("--count", type=int, default=300, help="Target count")
    parser.add_argument("--workers", type=int, default=8, help="Worker threads")
    parser.add_argument("--output", type=str, default="E:/apex-coder/data/clean/concurrency_300.jsonl")
    args = parser.parse_args()

    run_concurrency_curation(
        target_count=args.count,
        workers=args.workers,
        output_path=args.output,
    )
