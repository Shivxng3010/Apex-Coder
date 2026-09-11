import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schemas import DatasetSample, Language, VerificationOutcome
from harness.verifier import CodeVerifier
from pipeline.prompts import SYSTEM_PROMPT, get_curation_prompt, sample_random_topic
from pipeline.formatter import format_sample_to_chatml
from pipeline.problem_bank import generate_python_sample, generate_ts_sample


class CurationEngine:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "qwen/qwen-2.5-coder-32b-instruct",
        mock: bool = False,
    ):
        self.api_key = api_key or os.getenv("GENERATOR_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mock = mock or (not self.api_key)

    def generate_candidate(self, language: Language, index: int) -> Optional[DatasetSample]:
        if self.mock:
            if language == Language.PYTHON:
                return generate_python_sample(index)
            else:
                return generate_ts_sample(index)

        sample_id = f"sample_{language.value[:2]}_{index:04d}"
        category, detail, difficulty = sample_random_topic(language.value)
        prompt = get_curation_prompt(language.value, category, detail, difficulty)
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                return DatasetSample(
                    sample_id=sample_id,
                    language=language,
                    category=parsed.get("category", category),
                    difficulty=parsed.get("difficulty", difficulty),
                    instruction=parsed.get("instruction", ""),
                    buggy_code=parsed.get("buggy_code", ""),
                    solution_code=parsed.get("solution_code", ""),
                    test_code=parsed.get("test_code", ""),
                    explanation=parsed.get("explanation", ""),
                )
        except Exception as e:
            print(f"Generation API error for {sample_id}: {e}", file=sys.stderr)
            return None


_VERIFICATION_CACHE: Dict[str, bool] = {}


def verify_and_format_sample(verifier: CodeVerifier, sample: DatasetSample) -> Optional[Dict[str, Any]]:
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


def run_curation_pipeline(
    target_count: int = 2000,
    workers: int = 8,
    mock: bool = False,
    api_key: Optional[str] = None,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "qwen/qwen-2.5-coder-32b-instruct",
    output_path: str = "apex-coder/data/clean/train_2k.jsonl",
    use_docker: bool = False,
):
    print(f"===========================================================")
    print(f"  APEX CODER DATA CURATION & VERIFICATION PIPELINE")
    print(f"===========================================================")
    print(f"Target: {target_count:,} samples | Workers: {workers} | Docker: {use_docker}")
    print(f"Mode: {'OFFLINE GENERATOR' if (mock or not (api_key or os.getenv('GENERATOR_API_KEY'))) else 'API'}")
    print(f"Output File: {output_path}")
    print(f"-----------------------------------------------------------\n")

    verifier = CodeVerifier(use_docker=use_docker, timeout=5)
    engine = CurationEngine(api_key=api_key, base_url=base_url, model=model, mock=mock)

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

    print(f"[Starting Worker Pool] Generating and verifying {target_count:,} samples...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_idx = {
            executor.submit(
                lambda l, idx: (idx, verify_and_format_sample(verifier, engine.generate_candidate(l, idx))),
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

            if len(batch_buffer) >= 50 or total_processed == target_count:
                with open(out_file, "a", encoding="utf-8") as f:
                    for line in batch_buffer:
                        f.write(line + "\n")
                batch_buffer.clear()

            if verified_count % 100 == 0 or total_processed == target_count:
                elapsed = time.time() - start_time
                rate = verified_count / max(elapsed, 0.001)
                percent = (verified_count / target_count) * 100
                print(
                    f"[{verified_count:4d}/{target_count:4d}] "
                    f"({percent:5.1f}%) | "
                    f"Rate: {rate:5.1f} samples/s | "
                    f"Elapsed: {elapsed:5.1f}s"
                )

    total_time = time.time() - start_time
    print(f"\n===========================================================")
    print(f"Data Curation Pipeline Complete!")
    print(f"Total Verified Samples: {verified_count:,}/{target_count:,}")
    print(f"Total Time: {total_time:.2f}s ({verified_count / max(total_time, 0.001):.1f} samples/sec)")
    print(f"Saved to: {out_file.resolve()}")
    print(f"===========================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Code Curation Pipeline")
    parser.add_argument("--count", type=int, default=2000, help="Number of verified samples (default: 2000)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    parser.add_argument("--mock", action="store_true", help="Run offline high-throughput curation")
    parser.add_argument("--api-key", type=str, default=None, help="Generator API key")
    parser.add_argument("--base-url", type=str, default="https://openrouter.ai/api/v1", help="API URL")
    parser.add_argument("--model", type=str, default="qwen/qwen-2.5-coder-32b-instruct", help="Model name")
    parser.add_argument("--output", type=str, default="apex-coder/data/clean/train_2k.jsonl", help="Output path")
    parser.add_argument("--docker", action="store_true", help="Use Docker for runner")

    args = parser.parse_args()

    run_curation_pipeline(
        target_count=args.count,
        workers=args.workers,
        mock=args.mock,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        output_path=args.output,
        use_docker=args.docker,
    )
