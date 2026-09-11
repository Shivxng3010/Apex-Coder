import sys
import os
import time
import json
import shutil
import threading
from pathlib import Path

# Add project root
PROJECT_ROOT = Path("E:/apex-coder")
sys.path.insert(0, str(PROJECT_ROOT))

from harness.experience_harvester import ExperienceHarvester, FileLock
from pipeline.replay_buffer import ReplayBuffer
from training.auto_learn import get_active_adapter_path, set_active_adapter_path, get_next_adapter_path
from harness.self_correct import AutoVerifyResult, Language

TEST_TEMP_DIR = PROJECT_ROOT / "data" / "test_experience_pool"
if TEST_TEMP_DIR.exists():
    shutil.rmtree(TEST_TEMP_DIR, ignore_errors=True)

harvester = ExperienceHarvester(pool_dir=str(TEST_TEMP_DIR))
replay = ReplayBuffer(pool_dir=str(TEST_TEMP_DIR))

print("===========================================================")
print("      APEX CODER - CONTINUAL LEARNING TEST SUITE")
print("===========================================================\n")

# Test 1: Successful Experience Harvesting
res_valid = AutoVerifyResult(
    has_tests=True,
    is_valid=True,
    language=Language.TYPESCRIPT,
    status="passed",
    duration_seconds=0.42,
    stdout="",
    stderr="",
    extracted_code="export class Queue {\n  private items: number[] = [];\n  push(x: number) { this.items.push(x); }\n  pop() { return this.items.shift(); }\n}",
    extracted_test="import { describe, it } from 'node:test';\nimport assert from 'node:assert';\nimport { Queue } from './solution.ts';\ndescribe('Queue', () => {\n  it('works', () => {\n    const q = new Queue();\n    q.push(1);\n    assert.strictEqual(q.pop(), 1);\n  });\n});"
)
prompt_1 = "Implement a FIFO Queue in TypeScript with push and pop."
response_1 = "<thinking>FIFO queue</thinking>\n\n### Production-Ready Code\n```typescript\n" + res_valid.extracted_code + "\n```\n\n### Unit Test Suite\n```typescript\n" + res_valid.extracted_test + "\n```"

ok1, sample_id1 = harvester.harvest(prompt_1, response_1, res_valid)
print(f"[Test 1: Harvesting Valid Sample] Success: {ok1} | ID: {sample_id1}")
assert ok1 is True
assert sample_id1.startswith("exp_")

# Test 2: Deduplication Gate
ok2, reason2 = harvester.harvest(prompt_1, response_1, res_valid)
print(f"[Test 2: Deduplication Detection] Rejected: {not ok2} | Reason: {reason2}")
assert ok2 is False
assert "Duplicate" in reason2

# Test 3: Trivial / Failing Quality Gate Rejection
res_invalid = AutoVerifyResult(
    has_tests=True,
    is_valid=False,
    language=Language.TYPESCRIPT,
    status="failed",
    duration_seconds=0.1,
    stdout="",
    stderr="AssertionError",
    extracted_code="export const f = () => 1;",
    extracted_test="assert(false);"
)
ok3, reason3 = harvester.harvest("bad prompt", "bad code", res_invalid)
print(f"[Test 3: Quality Gate Rejection] Rejected: {not ok3} | Reason: {reason3}")
assert ok3 is False

# Test 4: FileLock Concurrency Simulation
concurrent_success = []
def worker(idx):
    res_w = AutoVerifyResult(
        has_tests=True,
        is_valid=True,
        language=Language.PYTHON,
        status="passed",
        duration_seconds=0.2,
        stdout="",
        stderr="",
        extracted_code=f"def fn_{idx}(x):\n    # implementation\n    return x * {idx}\n",
        extracted_test=f"def test_{idx}():\n    assert fn_{idx}(2) == {2 * idx}\n    assert True\n"
    )
    p = f"Implement fn_{idx}"
    r = f"### Production-Ready Code\n```python\n{res_w.extracted_code}\n```\n### Unit Test Suite\n```python\n{res_w.extracted_test}\n```"
    ok_w, _ = harvester.harvest(p, r, res_w, source="concurrency_test")
    if ok_w:
        concurrent_success.append(idx)

threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(f"[Test 4: Concurrent File Lock Harvesting] Successfully written: {len(concurrent_success)}/10 samples")
assert len(concurrent_success) == 10

stats = harvester.get_stats()
print(f"[Test 5: Harvester Stats] Pending Gold Samples: {stats['pending_gold_samples']}")
assert stats["pending_gold_samples"] == 11  # 1 from Test 1 + 10 from Test 4

# Test 6: Anti-Forgetting Replay Buffer 80/20 Mixing
staged_batch = replay.prepare_staged_batch(min_samples=5, live_ratio=0.8)
print(f"[Test 6: Replay Buffer Staging] Staged File: {staged_batch}")
assert staged_batch is not None
assert staged_batch.exists()

staged_records = []
with open(staged_batch, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            staged_records.append(json.loads(line))

print(f"         Total Staged Batch Size: {len(staged_records)} (Live: 11 + Anchors: {len(staged_records)-11})")
assert len(staged_records) >= 13  # 11 live + at least 2 anchors

# Test 7: Consumed Batch Archiving
replay.archive_consumed_batch()
post_stats = harvester.get_stats()
print(f"[Test 7: Consumed Batch Archiving] Post-Archive Pending: {post_stats['pending_gold_samples']} | Consumed History: {post_stats['consumed_history_samples']}")
assert post_stats["pending_gold_samples"] == 0
assert post_stats["consumed_history_samples"] == 11

# Test 8: Version Management & Active Adapter Tracking
active_path = get_active_adapter_path()
next_path = get_next_adapter_path()
print(f"[Test 8: Adapter Version Tracking] Active: {active_path.name} | Next: {next_path.name}")
assert "apex_coder_3b_lora" in active_path.name
assert "apex_coder_3b_lora_v" in next_path.name

# Cleanup test directory
shutil.rmtree(TEST_TEMP_DIR, ignore_errors=True)

print("\nAll 8 Continual Learning & Experience Harvester tests PASSED 100%!")
