import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

PROJECT_ROOT = Path("E:/apex-coder")
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure standard cache environments
os.environ.setdefault("HF_HOME", "E:/hf_cache")
os.environ.setdefault("UV_CACHE_DIR", "E:/uv_cache")
os.environ.setdefault("TORCH_HOME", "E:/hf_cache/torch")

from harness.experience_harvester import ExperienceHarvester
from pipeline.synthetic_dpo_generator import SyntheticDPOGenerator
from training.auto_learn import (
    get_active_adapter_path,
    get_next_adapter_path,
    set_active_adapter_path,
    run_benchmark_gate,
    execute_continual_learning,
)


def validate_experience_pools() -> Dict[str, Any]:
    """
    Inspects and validates the integrity of both DPO and SFT gold experience pools.
    """
    harvester = ExperienceHarvester()
    stats = harvester.get_stats()
    dpo_file = PROJECT_ROOT / "data" / "experience_pool" / "live_dpo_pairs.jsonl"
    staged_file = PROJECT_ROOT / "data" / "experience_pool" / "staged_replay_batch.jsonl"

    dpo_valid_lines = 0
    if dpo_file.exists():
        with open(dpo_file, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        json.loads(line_s)
                        dpo_valid_lines += 1
                    except Exception:
                        pass

    staged_valid_lines = 0
    if staged_file.exists():
        with open(staged_file, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    try:
                        json.loads(line_s)
                        staged_valid_lines += 1
                    except Exception:
                        pass

    return {
        "pending_gold_samples": stats.get("pending_gold_samples", 0),
        "pending_dpo_pairs": stats.get("pending_dpo_pairs", 0),
        "dpo_file_valid_records": dpo_valid_lines,
        "staged_file_valid_records": staged_valid_lines,
        "consumed_history_samples": stats.get("consumed_history_samples", 0),
        "pool_dir": str(stats.get("pool_dir", "")),
    }


def run_flywheel(
    dpo_count: int = 20,
    min_samples: int = 1,
    dry_run: bool = False,
    skip_dpo_gen: bool = False,
) -> Tuple[bool, str]:
    start_time = time.time()
    print("=" * 65)
    print("      APEX CODER - AUTONOMOUS SCHEDULED FLYWHEEL ENGINE")
    print("=" * 65)

    # 1. Dynamic Active Adapter Resolution
    active_adapter = get_active_adapter_path()
    target_next_adapter = get_next_adapter_path()
    print(f"[*] Timestamp           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Current Active LoRA : {active_adapter.name} ({active_adapter})")
    print(f"[*] Target Promotion    : {target_next_adapter.name} ({target_next_adapter})")
    print("-" * 65)

    # Step 1: Synthetic DPO Pair Generation
    print(f"\n[Step 1/5] Executing Synthetic DPO Preference Generation (Target: {dpo_count} pairs)...")
    if not skip_dpo_gen:
        generator = SyntheticDPOGenerator()
        generated_count = generator.generate_batch(count=dpo_count)
        print(f"          Successfully harvested {generated_count} verified DPO preference pairs.")
    else:
        print("          Skipped DPO generation per request flag (--skip-dpo-gen).")

    # Step 2: Pool Validation
    print("\n[Step 2/5] Validating Experience Pool & Preference Datasets...")
    pool_stats = validate_experience_pools()
    print(f"          Pending Gold Samples   : {pool_stats['pending_gold_samples']}")
    print(f"          Pending DPO Pairs      : {pool_stats['pending_dpo_pairs']}")
    print(f"          Validated DPO Records  : {pool_stats['dpo_file_valid_records']}")
    print(f"          Validated Replay Batch : {pool_stats['staged_file_valid_records']}")

    if pool_stats["pending_gold_samples"] == 0 and pool_stats["pending_dpo_pairs"] == 0:
        msg = "No pending experience pool samples available for training. Flywheel completed safely."
        print(f"[*] {msg}\n")
        return True, msg

    if dry_run:
        print("\n[Step 3-5/5] [DRY RUN] Simulating Quality Gate Benchmark Verification...")
        gate_passed, pass_rate, report = run_benchmark_gate()
        print(f"          Benchmark Gate Pass Rate: {pass_rate:.1f}% (Passed: {gate_passed})")
        msg = f"Dry Run completed successfully. Pass rate: {pass_rate:.1f}%. Active adapter kept at {active_adapter.name}."
        print(f"\n[*] {msg}\n" + "=" * 65)
        return True, msg

    # Step 3 & 4: Continual Micro-Training & Benchmark Gate Verification
    print(f"\n[Step 3/5] Triggering Continual Micro-Learning Pipeline ({active_adapter.name} -> {target_next_adapter.name})...")
    success, final_adapter, message = execute_continual_learning(min_samples=min_samples)

    # Step 5: Promotion / Rollback Result Verification
    print("\n[Step 5/5] Finalizing Adapter Promotion & Persistence...")
    current_active_after = get_active_adapter_path()
    total_elapsed = time.time() - start_time

    print("=" * 65)
    print("                    FLYWHEEL EXECUTION SUMMARY")
    print("=" * 65)
    print(f"Total Duration     : {total_elapsed:.2f} seconds")
    print(f"Execution Status   : {'PROMOTED' if success else 'ROLLED BACK / UNCHANGED'}")
    print(f"Result Message     : {message}")
    print(f"Final Active LoRA  : {current_active_after.name} ({current_active_after})")
    print("=" * 65 + "\n")

    return success, message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apex Coder Scheduled Autonomous Flywheel")
    parser.add_argument("--count", type=int, default=20, help="Number of DPO pairs to generate")
    parser.add_argument("--min-samples", type=int, default=1, help="Minimum gold samples for SFT")
    parser.add_argument("--dry-run", action="store_true", help="Perform DPO generation and benchmark check without training")
    parser.add_argument("--skip-dpo-gen", action="store_true", help="Skip DPO generation and process existing pool")
    args = parser.parse_args()

    success, msg = run_flywheel(
        dpo_count=args.count,
        min_samples=args.min_samples,
        dry_run=args.dry_run,
        skip_dpo_gen=args.skip_dpo_gen,
    )
    sys.exit(0 if success else 1)
