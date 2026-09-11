import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schemas import DatasetSample, Language
from harness.verifier import CodeVerifier
from pipeline.problem_bank import (
    _py_binary_search,
    _py_lru_cache,
    _py_merge_intervals,
    _py_dict_counter,
    _py_sliding_window_max,
    _py_flatten_nested_list,
    _py_debounce_accumulator,
    _py_valid_parentheses,
    _py_trie_prefix,
    _py_async_rate_limiter,
    _py_matrix_rotation,
    _py_linked_list_cycle,
    _py_top_k_frequent,
    _py_group_anagrams,
    _py_find_peak_element,
    _py_coin_change,
    _py_bounded_blocking_queue,
    _py_ttl_cache_ms,
    _py_trie_wildcard,
    _py_sliding_window_rate_limiter,
    _py_pubsub_broker,
    _py_graph_bfs_shortest_path,
    _py_monotonic_increasing_stack,
    _py_min_heap_priority_queue,
    _py_token_bucket_limiter,
    _ts_stack_generic,
    _ts_deep_clone,
    _ts_retry_backoff,
    _ts_event_emitter,
    _ts_promise_pool,
    _ts_typed_debounce,
    _ts_lru_cache_generic,
    _ts_flatten_object,
    _ts_circular_queue,
    _ts_binary_search_generic,
    _ts_bi_map,
    _ts_pipe_compose,
    _ts_custom_promise_all,
    _ts_rate_limiter_token_bucket,
    _ts_merge_sorted_arrays,
    _ts_clamp_and_paginate,
    _ts_reactive_signal,
    _ts_transactional_store,
    _ts_stream_chunk_transformer,
    _ts_deep_object_diff,
    _ts_cancellable_task_pool,
    _ts_lfu_cache,
    _ts_event_bus_typed,
    _ts_sorted_interval_insert,
    _ts_async_retry_exponential,
)
from pipeline.concurrency_bank import (
    _ts_fair_async_rwlock,
    _ts_weakref_event_bus,
    _ts_transactional_kv_store_savepoints,
    _ts_lfu_cache_o1,
    _ts_async_countdown_latch,
    _ts_microsecond_leaky_bucket,
)


BENCHMARK_SUITES = [
    ("Python: Binary Search Exact Bounds", _py_binary_search(9001)),
    ("Python: LRU Cache Recency & Eviction", _py_lru_cache(9002)),
    ("Python: Unsorted Interval Merging", _py_merge_intervals(9003)),
    ("Python: Fast Dict Frequency Counter", _py_dict_counter(9004)),
    ("Python: Sliding Window Max (Monotonic Deque)", _py_sliding_window_max(9005)),
    ("Python: Recursive List Flattening", _py_flatten_nested_list(9006)),
    ("Python: Timed Debounce Accumulator", _py_debounce_accumulator(9007)),
    ("Python: Valid Matching Parentheses", _py_valid_parentheses(9008)),
    ("Python: Trie Prefix Auto-Complete", _py_trie_prefix(9009)),
    ("Python: Async Token Rate Limiter", _py_async_rate_limiter(9010)),
    ("Python: In-Place 90-Deg Matrix Rotation", _py_matrix_rotation(9011)),
    ("Python: Linked List Cycle Detection", _py_linked_list_cycle(9012)),
    ("Python: Top-K Frequent Elements", _py_top_k_frequent(9013)),
    ("Python: Sorted Key Group Anagrams", _py_group_anagrams(9014)),
    ("Python: Binary Search Peak Element", _py_find_peak_element(9015)),
    ("Python: Dynamic Programming Coin Change", _py_coin_change(9016)),
    ("Python: Bounded Blocking Queue (Condition)", _py_bounded_blocking_queue(9017)),
    ("Python: Millisecond TTL Cache Eviction", _py_ttl_cache_ms(9018)),
    ("Python: Trie Wildcard Pattern Search", _py_trie_wildcard(9019)),
    ("Python: Sliding Window Log Rate Limiter", _py_sliding_window_rate_limiter(9020)),
    ("Python: Pub-Sub Topic Event Broker", _py_pubsub_broker(9021)),
    ("Python: BFS Unweighted Shortest Path", _py_graph_bfs_shortest_path(9022)),
    ("Python: Monotonic Stack Daily Temps", _py_monotonic_increasing_stack(9023)),
    ("Python: FIFO Min-Heap Priority Queue", _py_min_heap_priority_queue(9024)),
    ("Python: Smooth Token Bucket Limiter", _py_token_bucket_limiter(9025)),
    ("TypeScript: Generic Typed Stack", _ts_stack_generic(9026)),
    ("TypeScript: Deep Clone Strict (Array/Object)", _ts_deep_clone(9027)),
    ("TypeScript: Async Retry Backoff", _ts_retry_backoff(9028)),
    ("TypeScript: Typed Event Emitter", _ts_event_emitter(9029)),
    ("TypeScript: Async Promise Pool Concurrency", _ts_promise_pool(9030)),
    ("TypeScript: Typed Micro Debounce", _ts_typed_debounce(9031)),
    ("TypeScript: Generic LRU Cache", _ts_lru_cache_generic(9032)),
    ("TypeScript: Deep Object Flattening", _ts_flatten_object(9033)),
    ("TypeScript: Circular Ring Queue Buffer", _ts_circular_queue(9034)),
    ("TypeScript: Comparator Binary Search", _ts_binary_search_generic(9035)),
    ("TypeScript: Bidirectional Map Invalidation", _ts_bi_map(9036)),
    ("TypeScript: Functional Pipe Composition", _ts_pipe_compose(9037)),
    ("TypeScript: Custom Promise.all Polyfill", _ts_custom_promise_all(9038)),
    ("TypeScript: Token Bucket Rate Limiter", _ts_rate_limiter_token_bucket(9039)),
    ("TypeScript: Merge Two Sorted Arrays", _ts_merge_sorted_arrays(9040)),
    ("TypeScript: Clamped Pagination Slicer", _ts_clamp_and_paginate(9041)),
    ("TypeScript: Reactive Signal Dependency Tracker", _ts_reactive_signal(9042)),
    ("TypeScript: Transactional Key-Value Store", _ts_transactional_store(9043)),
    ("TypeScript: Stream Delimiter Chunk Transformer", _ts_stream_chunk_transformer(9044)),
    ("TypeScript: Deep Object Diff & Path Engine", _ts_deep_object_diff(9045)),
    ("TypeScript: Cancellable Concurrent Task Pool", _ts_cancellable_task_pool(9046)),
    ("TypeScript: Least Frequently Used (LFU) Cache", _ts_lfu_cache(9047)),
    ("TypeScript: Typed Subscription Event Bus", _ts_event_bus_typed(9048)),
    ("TypeScript: Insert & Merge Sorted Interval", _ts_sorted_interval_insert(9049)),
    ("TypeScript: Exponential Async Retry Loop", _ts_async_retry_exponential(9050)),
    ("TypeScript: Fair Async Read-Write Lock", _ts_fair_async_rwlock(9051)),
    ("TypeScript: WeakRef Memory-Safe EventBus", _ts_weakref_event_bus(9052)),
    ("TypeScript: Transactional KV-Store Savepoints", _ts_transactional_kv_store_savepoints(9053)),
    ("TypeScript: O(1) LFU Cache Doubly Linked", _ts_lfu_cache_o1(9054)),
    ("TypeScript: Async CountDownLatch Barrier", _ts_async_countdown_latch(9055)),
    ("TypeScript: Microsecond Leaky Bucket Scheduler", _ts_microsecond_leaky_bucket(9056)),
]


def run_benchmark_eval(output_report: str = "E:/apex-coder/eval/benchmark_report.json"):
    print("===========================================================")
    print(f"      APEX CODER - {len(BENCHMARK_SUITES)} CODING BENCHMARK EVALUATION")
    print("===========================================================")
    print(f"Total Suites: {len(BENCHMARK_SUITES)} (25 Python + {len(BENCHMARK_SUITES)-25} TypeScript)")
    print("Sandbox Engine: Isolated Subprocess Execution Harness")
    print("-----------------------------------------------------------\n", flush=True)

    verifier = CodeVerifier(use_docker=False, timeout=5)
    results = []
    passed = 0
    start_time = time.time()

    for i, (name, sample) in enumerate(BENCHMARK_SUITES, 1):
        t0 = time.time()
        outcome = verifier.verify_sample(sample)
        duration = time.time() - t0

        status_str = "PASSED" if outcome.is_valid else "FAILED"
        if outcome.is_valid:
            passed += 1

        print(f"[{i:2d}/{len(BENCHMARK_SUITES)}] {name:48s} | Status: {status_str:6s} | Time: {duration:5.2f}s", flush=True)

        results.append({
            "index": i,
            "suite_name": name,
            "sample_id": sample.sample_id,
            "language": sample.language.value,
            "category": sample.category,
            "difficulty": sample.difficulty,
            "is_valid": outcome.is_valid,
            "duration_sec": round(duration, 3),
            "rejection_reason": outcome.rejection_reason,
        })

    total_time = time.time() - start_time
    pass_rate = (passed / len(BENCHMARK_SUITES)) * 100

    report = {
        "benchmark_title": f"Apex Coder {len(BENCHMARK_SUITES)} Functional Evaluation Suites",
        "total_suites": len(BENCHMARK_SUITES),
        "passed_suites": passed,
        "pass_rate_pct": pass_rate,
        "total_time_sec": round(total_time, 2),
        "results": results,
    }

    out_file = Path(output_report)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n===========================================================")
    print(f"Evaluation Complete! Score: {passed}/{len(BENCHMARK_SUITES)} ({pass_rate:.1f}%)")
    print(f"Total Duration: {total_time:.2f}s")
    print(f"Detailed Report Saved to: {out_file.resolve()}")
    print("===========================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 50 coding benchmark suites")
    parser.add_argument("--output", type=str, default="E:/apex-coder/eval/benchmark_report.json")
    args = parser.parse_args()

    run_benchmark_eval(output_report=args.output)
