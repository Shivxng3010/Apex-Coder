import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.replay_buffer import ReplayBuffer
from harness.experience_harvester import ExperienceHarvester
ACTIVE_ADAPTER_FILE = PROJECT_ROOT / "outputs" / "ACTIVE_ADAPTER.txt"
DEFAULT_ADAPTER = PROJECT_ROOT / "outputs" / "apex_coder_3b_lora_v5"


def get_active_adapter_path() -> Path:
    """
    Returns the currently active LoRA adapter path.
    """
    if ACTIVE_ADAPTER_FILE.exists():
        try:
            with open(ACTIVE_ADAPTER_FILE, "r", encoding="utf-8") as f:
                path_str = f.read().strip()
                if path_str:
                    p = Path(path_str)
                    if p.is_absolute() and p.exists():
                        return p
                    rel_p = PROJECT_ROOT / path_str
                    if rel_p.exists():
                        return rel_p
                    dir_p = PROJECT_ROOT / "outputs" / p.name
                    if dir_p.exists():
                        return dir_p
        except Exception:
            pass

    if DEFAULT_ADAPTER.exists():
        return DEFAULT_ADAPTER
    for candidate in ["apex_coder_3b_lora_v5", "apex_coder_3b_lora_v4", "apex_coder_3b_lora_v3", "apex_coder_3b_lora_v2", "apex_coder_3b_lora"]:
        p = PROJECT_ROOT / "outputs" / candidate
        if p.exists():
            return p
    return PROJECT_ROOT / "outputs" / "apex_coder_3b_lora"


def set_active_adapter_path(adapter_path: Path):
    """
    Persists the newly promoted active LoRA adapter path.
    """
    ACTIVE_ADAPTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_ADAPTER_FILE, "w", encoding="utf-8") as f:
        f.write(str(adapter_path.resolve()))


def get_next_adapter_path() -> Path:
    """
    Determines the next incremental versioned adapter directory (e.g. lora_v3, lora_v4...).
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    existing_dirs = list(outputs_dir.glob("apex_coder_3b_lora_v*"))
    versions = []
    for d in existing_dirs:
        try:
            v = int(d.name.split("_v")[-1])
            versions.append(v)
        except Exception:
            pass

    next_v = max(versions, default=2) + 1
    return outputs_dir / f"apex_coder_3b_lora_v{next_v}"


def run_benchmark_gate() -> Tuple[bool, float, Dict[str, Any]]:
    """
    Executes the functional benchmark suite on CPU subprocess to verify model capabilities.
    Returns (passed: bool, pass_rate: float, report: dict).
    """
    eval_script = PROJECT_ROOT / "eval" / "benchmark_eval.py"
    report_file = PROJECT_ROOT / "eval" / "benchmark_report.json"

    env = os.environ.copy()
    env["HF_HOME"] = "E:/hf_cache"
    env["UV_CACHE_DIR"] = "E:/uv_cache"
    env["TORCH_HOME"] = "E:/hf_cache/torch"

    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
    proc = subprocess.run(
        [sys.executable, str(eval_script), "--output", str(report_file)],
        env=env,
        capture_output=True,
        text=True,
        creationflags=flags,
        cwd=str(PROJECT_ROOT),
    )

    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                report = json.load(f)
                pass_rate = report.get("pass_rate_pct", 0.0)
                is_passed = (pass_rate == 100.0)
                return is_passed, pass_rate, report
        except Exception:
            pass

    return (proc.returncode == 0), 100.0 if proc.returncode == 0 else 0.0, {}


def execute_continual_learning(
    min_samples: int = 1,
    learning_rate: float = 5e-5,
    epochs: Optional[int] = None,
) -> Tuple[bool, Path, str]:
    """
    Orchestrates the complete continual self-learning pipeline:
    1. Replay Buffer 80/20 data staging (for SFT if gold samples exist)
    2. Subprocess-isolated QLoRA micro-training (SFT) & DPO preference optimization
    3. Benchmark Quality Gate verification (56+ Functional Suites)
    4. Auto-promotion or rollback
    """
    harvester = ExperienceHarvester()
    replay = ReplayBuffer()
    stats = harvester.get_stats()

    has_gold = stats["pending_gold_samples"] >= min_samples
    has_dpo = stats.get("pending_dpo_pairs", 0) >= 1

    if not has_gold and not has_dpo:
        current_adapter = get_active_adapter_path()
        return False, current_adapter, f"Insufficient samples: Gold={stats['pending_gold_samples']}, DPO={stats.get('pending_dpo_pairs', 0)}"

    current_adapter = get_active_adapter_path()
    next_adapter = get_next_adapter_path()
    env = os.environ.copy()
    env["HF_HOME"] = "E:/hf_cache"
    env["UV_CACHE_DIR"] = "E:/uv_cache"
    env["TORCH_HOME"] = "E:/hf_cache/torch"
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    intermediate_adapter = current_adapter

    # 1. SFT Training if gold samples present
    if has_gold:
        print("[1/3] Preparing Anti-Forgetting Replay Buffer (80% Live + 20% Anchors)...", flush=True)
        staged_path = replay.prepare_staged_batch(min_samples=min_samples)
        if staged_path and staged_path.exists():
            staged_count = sum(1 for line in open(staged_path, "r", encoding="utf-8") if line.strip())
            actual_epochs = 3 if staged_count <= 10 else (2 if staged_count <= 30 else 1)
            grad_accum = 2 if staged_count <= 10 else 4

            print(f"[2/3] Launching Subprocess SFT Micro-Training ({current_adapter.name} -> {next_adapter.name})...", flush=True)
            train_script = PROJECT_ROOT / "training" / "train_unsloth.py"
            train_cmd = [
                sys.executable,
                str(train_script),
                "--dataset", str(staged_path),
                "--output", str(next_adapter),
                "--adapt-from", str(current_adapter),
                "--lr", str(learning_rate),
                "--epochs", str(actual_epochs),
                "--grad-accum", str(grad_accum),
                "--seq-len", "1024",
            ]
            t0 = time.time()
            proc = subprocess.run(train_cmd, env=env, creationflags=flags, cwd=str(PROJECT_ROOT))
            if proc.returncode != 0:
                return False, current_adapter, f"SFT training subprocess failed with exit code {proc.returncode}"
            print(f"      SFT Training complete in {time.time()-t0:.2f}s!", flush=True)
            intermediate_adapter = next_adapter

    # 2. DPO Training if DPO pairs present
    if has_dpo:
        dpo_file = PROJECT_ROOT / "data" / "experience_pool" / "live_dpo_pairs.jsonl"
        dpo_script = PROJECT_ROOT / "training" / "train_dpo.py"
        dpo_count = stats.get("pending_dpo_pairs", 0)
        dpo_epochs = 1 if dpo_count <= 10 else 2
        print(f"[2b/3] Launching Subprocess DPO Training on {dpo_count} pairs (Epochs: {dpo_epochs})...", flush=True)
        dpo_cmd = [
            sys.executable,
            str(dpo_script),
            "--dataset", str(dpo_file),
            "--output_dir", str(next_adapter),
            "--adapt_from", str(intermediate_adapter),
            "--epochs", str(dpo_epochs),
        ]
        t0 = time.time()
        proc = subprocess.run(dpo_cmd, env=env, creationflags=flags, cwd=str(PROJECT_ROOT))
        if proc.returncode != 0:
            return False, current_adapter, f"DPO training subprocess failed with exit code {proc.returncode}"
        print(f"      DPO Training complete in {time.time()-t0:.2f}s!", flush=True)

    # 3. Run Benchmark Quality Gate
    print("[3/3] Running Automated Benchmark Quality Gate (56+ Functional Suites)...", flush=True)
    gate_passed, pass_rate, _ = run_benchmark_gate()

    if gate_passed:
        set_active_adapter_path(next_adapter)
        if has_gold:
            replay.archive_consumed_batch()
        msg = f"SUCCESS: Quality Gate 100% Passed. Promoted to {next_adapter.name}!"
        print(f"      {msg}", flush=True)
        return True, next_adapter, msg
    else:
        msg = f"REGRESSION DETECTED: Pass rate {pass_rate:.1f}% < 100%. Rolled back to {current_adapter.name}."
        print(f"      {msg}", flush=True)
        return False, current_adapter, msg


if __name__ == "__main__":
    success, active, message = execute_continual_learning(min_samples=1)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'} | Active Adapter: {active} | Message: {message}")
