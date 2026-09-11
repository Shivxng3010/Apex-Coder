import random
from typing import List, Dict, Any
from harness.schemas import DatasetSample, Language


# Dynamic generators that generate parameterized, guaranteed-correct bug/fix/test pairs
def generate_python_sample(index: int) -> DatasetSample:
    archetypes = [
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
    ]
    gen = archetypes[index % len(archetypes)]
    return gen(index)


def generate_ts_sample(index: int) -> DatasetSample:
    archetypes = [
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
    ]
    gen = archetypes[index % len(archetypes)]
    return gen(index)


# ==================== PYTHON ARCHETYPES ====================

def _py_binary_search(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="easy",
        instruction="Fix the binary search function so it returns the correct 0-indexed position or -1 if the target is absent.",
        buggy_code="""def binary_search(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid
        else:
            right = mid
    return -1
""",
        solution_code="""def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        test_code="""from solution import binary_search

def test_binary_search():
    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 11) == 5
    assert binary_search(arr, 6) == -1
    assert binary_search(arr, 0) == -1
    assert binary_search([], 42) == -1
""",
        explanation="The pointers must be adjusted by +/- 1 relative to mid to avoid infinite loops and examine all elements."
    )


def _py_lru_cache(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="medium",
        instruction="Fix the LRUCache so get operations properly update key recency, avoiding premature eviction.",
        buggy_code="""class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            first = next(iter(self.cache))
            del self.cache[first]
        self.cache[key] = value
""",
        solution_code="""class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        val = self.cache.pop(key)
        self.cache[key] = val
        return val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            first = next(iter(self.cache))
            del self.cache[first]
        self.cache[key] = value
""",
        test_code="""from solution import LRUCache

def test_lru_recency():
    c = LRUCache(2)
    c.put(1, 10)
    c.put(2, 20)
    assert c.get(1) == 10
    c.put(3, 30)
    assert c.get(2) == -1
    assert c.get(1) == 10
    assert c.get(3) == 30
""",
        explanation="On get(key), pop and re-insert the key to maintain recency order."
    )


def _py_merge_intervals(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix merge_intervals to handle unsorted interval inputs properly.",
        buggy_code="""def merge_intervals(intervals):
    if not intervals:
        return []
    res = [intervals[0]]
    for cur in intervals[1:]:
        last = res[-1]
        if cur[0] <= last[1]:
            last[1] = max(last[1], cur[1])
        else:
            res.append(cur)
    return res
""",
        solution_code="""def merge_intervals(intervals):
    if not intervals:
        return []
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    res = [sorted_intervals[0]]
    for cur in sorted_intervals[1:]:
        last = res[-1]
        if cur[0] <= last[1]:
            last[1] = max(last[1], cur[1])
        else:
            res.append(cur)
    return res
""",
        test_code="""from solution import merge_intervals

def test_merge_intervals():
    assert merge_intervals([[1, 3], [2, 6], [8, 10]]) == [[1, 6], [8, 10]]
    assert merge_intervals([[2, 3], [1, 4]]) == [[1, 4]]
    assert merge_intervals([[5, 7], [1, 2]]) == [[1, 2], [5, 7]]
    assert merge_intervals([]) == []
""",
        explanation="Intervals must be sorted by start boundary before merging adjacent intervals."
    )


def _py_dict_counter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="collections",
        difficulty="easy",
        instruction="Fix the word frequency counter to safely handle new words without raising KeyError.",
        buggy_code="""def count_words(words):
    result = {}
    for w in words:
        result[w] += 1
    return result
""",
        solution_code="""def count_words(words):
    result = {}
    for w in words:
        result[w] = result.get(w, 0) + 1
    return result
""",
        test_code="""from solution import count_words

def test_count_words():
    assert count_words(["apple", "banana", "apple"]) == {"apple": 2, "banana": 1}
    assert count_words([]) == {}
    assert count_words(["x"]) == {"x": 1}
""",
        explanation="Use dict.get(key, 0) to initialize missing counts."
    )


def _py_sliding_window_max(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="hard",
        instruction="Fix max_sliding_window to return maximums for each sliding window of size k in O(n) time.",
        buggy_code="""def max_sliding_window(nums, k):
    if not nums or k <= 0:
        return []
    res = []
    for i in range(len(nums) - k):
        res.append(max(nums[i:i + k]))
    return res
""",
        solution_code="""from collections import deque

def max_sliding_window(nums, k):
    if not nums or k <= 0:
        return []
    if k >= len(nums):
        return [max(nums)]
    dq = deque()
    res = []
    for i, n in enumerate(nums):
        while dq and nums[dq[-1]] < n:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res
""",
        test_code="""from solution import max_sliding_window

def test_sliding_window():
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert max_sliding_window([1], 1) == [1]
    assert max_sliding_window([1, -1], 1) == [1, -1]
    assert max_sliding_window([9, 11], 2) == [11]
""",
        explanation="Use monotonic deque to track maximum indices efficiently within sliding window boundaries."
    )


def _py_flatten_nested_list(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="recursion",
        difficulty="easy",
        instruction="Fix the flatten function so it recursively unwraps nested lists of arbitrary depth into a 1D list.",
        buggy_code="""def flatten(nested):
    res = []
    for item in nested:
        if isinstance(item, list):
            res.extend(item)
        else:
            res.append(item)
    return res
""",
        solution_code="""def flatten(nested):
    res = []
    for item in nested:
        if isinstance(item, list):
            res.extend(flatten(item))
        else:
            res.append(item)
    return res
""",
        test_code="""from solution import flatten

def test_flatten():
    assert flatten([1, [2, [3, [4]], 5], 6]) == [1, 2, 3, 4, 5, 6]
    assert flatten([]) == []
    assert flatten([[1], [2], [3]]) == [1, 2, 3]
    assert flatten([1, 2, 3]) == [1, 2, 3]
""",
        explanation="Recursively invoke flatten(item) when encountering nested list items."
    )


def _py_debounce_accumulator(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="concurrency",
        difficulty="medium",
        instruction="Fix the BatchAccumulator class so flush empties accumulated items and returns a snapshot.",
        buggy_code="""class BatchAccumulator:
    def __init__(self, max_batch_size: int = 5):
        self.max_batch_size = max_batch_size
        self.items = []

    def add(self, item):
        self.items.append(item)
        if len(self.items) >= self.max_batch_size:
            return self.flush()
        return None

    def flush(self):
        self.items.clear()
        return self.items
""",
        solution_code="""class BatchAccumulator:
    def __init__(self, max_batch_size: int = 5):
        self.max_batch_size = max_batch_size
        self.items = []

    def add(self, item):
        self.items.append(item)
        if len(self.items) >= self.max_batch_size:
            return self.flush()
        return None

    def flush(self):
        batch = list(self.items)
        self.items.clear()
        return batch
""",
        test_code="""from solution import BatchAccumulator

def test_accumulator():
    acc = BatchAccumulator(max_batch_size=3)
    assert acc.add(1) is None
    assert acc.add(2) is None
    batch = acc.add(3)
    assert batch == [1, 2, 3]
    assert acc.flush() == []
""",
        explanation="Create a shallow copy of items before invoking self.items.clear()."
    )


def _py_valid_parentheses(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="easy",
        instruction="Fix is_valid_brackets to check balanced matching brackets properly including mismatched closing brackets.",
        buggy_code="""def is_valid_brackets(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top = stack.pop()
            if mapping[char] != top:
                return False
        else:
            stack.append(char)
    return len(stack) == 0
""",
        solution_code="""def is_valid_brackets(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return len(stack) == 0
""",
        test_code="""from solution import is_valid_brackets

def test_brackets():
    assert is_valid_brackets("()[]{}") is True
    assert is_valid_brackets("([{}])") is True
    assert is_valid_brackets("(]") is False
    assert is_valid_brackets("]") is False
    assert is_valid_brackets("") is True
""",
        explanation="Check if stack is non-empty before popping when encountering a closing delimiter."
    )


def _py_trie_prefix(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="medium",
        instruction="Fix Trie.starts_with so it returns True when a prefix exists without requiring an end-of-word marker.",
        buggy_code="""class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end
""",
        solution_code="""class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def starts_with(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
""",
        test_code="""from solution import Trie

def test_trie():
    t = Trie()
    t.insert("apple")
    assert t.starts_with("app") is True
    assert t.starts_with("apple") is True
    assert t.starts_with("applause") is False
""",
        explanation="starts_with only needs to verify the path of characters exists, returning True upon reaching the end of prefix."
    )


def _py_async_rate_limiter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix the sliding window count in RequestThrottler to evict timestamps outside the window.",
        buggy_code="""class RequestThrottler:
    def __init__(self, max_requests: int, window_sec: float):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.timestamps = []

    def allow_request(self, current_time: float) -> bool:
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
""",
        solution_code="""class RequestThrottler:
    def __init__(self, max_requests: int, window_sec: float):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.timestamps = []

    def allow_request(self, current_time: float) -> bool:
        cutoff = current_time - self.window_sec
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
""",
        test_code="""from solution import RequestThrottler

def test_throttler():
    t = RequestThrottler(max_requests=2, window_sec=10.0)
    assert t.allow_request(1.0) is True
    assert t.allow_request(2.0) is True
    assert t.allow_request(3.0) is False
    assert t.allow_request(12.0) is True
""",
        explanation="Filter out timestamps older than current_time - window_sec before evaluating capacity."
    )


def _py_matrix_rotation(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix rotate_matrix_90_clockwise to perform in-place 90-degree clockwise rotation.",
        buggy_code="""def rotate_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
""",
        solution_code="""def rotate_matrix(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    for row in matrix:
        row.reverse()
""",
        test_code="""from solution import rotate_matrix

def test_rotate():
    mat = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    rotate_matrix(mat)
    assert mat == [
        [7, 4, 1],
        [8, 5, 2],
        [9, 6, 3]
    ]
""",
        explanation="Clockwise 90-degree rotation requires transposing the matrix and then reversing each individual row."
    )


def _py_linked_list_cycle(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="easy",
        instruction="Fix has_cycle using Floyd's Tortoise and Hare algorithm.",
        buggy_code="""class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode) -> bool:
    slow = head
    fast = head
    while fast:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""",
        solution_code="""class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head: ListNode) -> bool:
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
""",
        test_code="""from solution import ListNode, has_cycle

def test_cycle():
    n1, n2, n3 = ListNode(1), ListNode(2), ListNode(3)
    n1.next = n2
    n2.next = n3
    assert has_cycle(n1) is False

    n3.next = n1
    assert has_cycle(n1) is True
    assert has_cycle(None) is False
""",
        explanation="The loop guard must verify both fast and fast.next are not None before advancing fast by two steps."
    )


def _py_top_k_frequent(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix top_k_frequent to return the k most common elements from an array.",
        buggy_code="""import collections

def top_k_frequent(nums, k):
    counts = collections.Counter(nums)
    sorted_items = sorted(counts.items(), key=lambda x: x[1])
    return [item[0] for item in sorted_items[:k]]
""",
        solution_code="""import collections

def top_k_frequent(nums, k):
    counts = collections.Counter(nums)
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:k]]
""",
        test_code="""from solution import top_k_frequent

def test_top_k():
    res = top_k_frequent([1, 1, 1, 2, 2, 3], 2)
    assert sorted(res) == [1, 2]
    assert top_k_frequent([4], 1) == [4]
""",
        explanation="Sort frequency items in descending order (reverse=True) to obtain highest frequency keys first."
    )


def _py_group_anagrams(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix group_anagrams to group words that are anagrams of each other.",
        buggy_code="""from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = sorted(s)
        groups[key].append(s)
    return list(groups.values())
""",
        solution_code="""from collections import defaultdict

def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
""",
        test_code="""from solution import group_anagrams

def test_group_anagrams():
    res = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    sorted_res = [sorted(g) for g in sorted(res, key=len)]
    assert sorted_res == [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]]
""",
        explanation="Convert sorted(s) list into an immutable tuple or string so it can serve as a hashable dictionary key."
    )


def _py_find_peak_element(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix find_peak_element to locate any local peak index in O(log n) time.",
        buggy_code="""def find_peak(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid - 1
        else:
            left = mid + 1
    return left
""",
        solution_code="""def find_peak(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid
        else:
            left = mid + 1
    return left
""",
        test_code="""from solution import find_peak

def test_find_peak():
    assert find_peak([1, 2, 3, 1]) == 2
    assert find_peak([1, 2, 1, 3, 5, 6, 4]) in (1, 5)
    assert find_peak([1]) == 0
""",
        explanation="When nums[mid] > nums[mid + 1], a peak exists at mid or to the left, so right must be updated to mid."
    )


def _py_coin_change(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="dynamic-programming",
        difficulty="medium",
        instruction="Fix coin_change to return fewest coins needed to make amount or -1 if impossible.",
        buggy_code="""def coin_change(coins, amount):
    dp = [0] * (amount + 1)
    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount]
""",
        solution_code="""def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
""",
        test_code="""from solution import coin_change

def test_coin_change():
    assert coin_change([1, 2, 5], 11) == 3
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0
""",
        explanation="Initialize dp array with infinity and set dp[0] = 0 before computing transitions."
    )


# ==================== TYPESCRIPT ARCHETYPES ====================

def _ts_stack_generic(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="easy",
        instruction="Fix Stack so peek inspects the top element without removing it.",
        buggy_code="""export class Stack<T> {
  private items: T[] = [];
  push(item: T): void { this.items.push(item); }
  pop(): T | undefined { return this.items.pop(); }
  peek(): T | undefined {
    return this.items.pop();
  }
  size(): number { return this.items.length; }
}
""",
        solution_code="""export class Stack<T> {
  private items: T[] = [];
  push(item: T): void { this.items.push(item); }
  pop(): T | undefined { return this.items.pop(); }
  peek(): T | undefined {
    return this.items.length > 0 ? this.items[this.items.length - 1] : undefined;
  }
  size(): number { return this.items.length; }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Stack } from './solution.ts';

describe('Stack', () => {
  it('peeks without popping', () => {
    const s = new Stack<number>();
    s.push(100);
    s.push(200);
    assert.strictEqual(s.peek(), 200);
    assert.strictEqual(s.size(), 2);
    assert.strictEqual(s.pop(), 200);
    assert.strictEqual(s.peek(), 100);
  });
});
""",
        explanation="peek must read the last index without mutating items."
    )


def _ts_deep_clone(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="immutability",
        difficulty="medium",
        instruction="Fix deepClone so arrays and nested primitives are correctly duplicated.",
        buggy_code="""export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  const copy = {} as any;
  for (const k in obj) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      copy[k] = deepClone((obj as any)[k]);
    }
  }
  return copy;
}
""",
        solution_code="""export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;
  if (Array.isArray(obj)) {
    return obj.map(item => deepClone(item)) as unknown as T;
  }
  const copy: Record<string, any> = {};
  for (const k of Object.keys(obj)) {
    copy[k] = deepClone((obj as any)[k]);
  }
  return copy as T;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { deepClone } from './solution.ts';

describe('deepClone', () => {
  it('clones arrays and objects', () => {
    const orig = { a: [1, 2], b: { c: 'hello' } };
    const cloned = deepClone(orig);
    assert.deepStrictEqual(cloned, orig);
    assert.notStrictEqual(cloned.a, orig.a);
    assert.notStrictEqual(cloned.b, orig.b);
  });
});
""",
        explanation="Check Array.isArray(obj) and map elements to preserve Array prototype."
    )


def _ts_retry_backoff(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="medium",
        instruction="Fix retryWithBackoff to re-throw error once maxRetries is exceeded.",
        buggy_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  delayMs: number = 5
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      await new Promise(r => setTimeout(r, delayMs * attempt));
    }
  }
}
""",
        solution_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  delayMs: number = 5
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      if (attempt > maxRetries) throw err;
      await new Promise(r => setTimeout(r, delayMs * attempt));
    }
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { retryWithBackoff } from './solution.ts';

describe('retry', () => {
  it('throws on persistent failure', async () => {
    let count = 0;
    await assert.rejects(
      async () => {
        await retryWithBackoff(async () => {
          count++;
          throw new Error('fail');
        }, 2, 2);
      },
      /fail/
    );
    assert.strictEqual(count, 3);
  });
});
""",
        explanation="Check attempt > maxRetries and throw the caught error."
    )


def _ts_event_emitter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="event-emitter",
        difficulty="medium",
        instruction="Fix EventEmitter so off removes only the target listener function.",
        buggy_code="""export class EventEmitter {
  private listeners: Record<string, Function[]> = {};
  on(event: string, fn: Function): void {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(fn);
  }
  emit(event: string, ...args: any[]): void {
    (this.listeners[event] || []).forEach(fn => fn(...args));
  }
  off(event: string, fn: Function): void {
    this.listeners[event] = [];
  }
}
""",
        solution_code="""export class EventEmitter {
  private listeners: Record<string, Function[]> = {};
  on(event: string, fn: Function): void {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(fn);
  }
  emit(event: string, ...args: any[]): void {
    (this.listeners[event] || []).forEach(fn => fn(...args));
  }
  off(event: string, fn: Function): void {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(cb => cb !== fn);
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { EventEmitter } from './solution.ts';

describe('EventEmitter', () => {
  it('removes specific listener with off', () => {
    const ee = new EventEmitter();
    let c1 = 0, c2 = 0;
    const fn1 = () => c1++;
    const fn2 = () => c2++;
    ee.on('ping', fn1);
    ee.on('ping', fn2);
    ee.off('ping', fn1);
    ee.emit('ping');
    assert.strictEqual(c1, 0);
    assert.strictEqual(c2, 1);
  });
});
""",
        explanation="Filter listener array to retain non-matching callbacks on off."
    )


def _ts_promise_pool(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="hard",
        instruction="Fix runWithConcurrency to execute promises limiting concurrent tasks to maxConcurrent.",
        buggy_code="""export async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrent: number
): Promise<T[]> {
  return Promise.all(tasks.map(t => t()));
}
""",
        solution_code="""export async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrent: number
): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let nextIdx = 0;

  async function worker() {
    while (nextIdx < tasks.length) {
      const cur = nextIdx++;
      results[cur] = await tasks[cur]();
    }
  }

  const workers = Array.from({ length: Math.min(maxConcurrent, tasks.length) }, () => worker());
  await Promise.all(workers);
  return results;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { runWithConcurrency } from './solution.ts';

describe('runWithConcurrency', () => {
  it('respects concurrency limit', async () => {
    let active = 0;
    let maxSeen = 0;
    const makeTask = (val: number) => async () => {
      active++;
      maxSeen = Math.max(maxSeen, active);
      await new Promise(r => setTimeout(r, 10));
      active--;
      return val * 2;
    };
    const tasks = [1, 2, 3, 4, 5].map(makeTask);
    const res = await runWithConcurrency(tasks, 2);
    assert.deepStrictEqual(res, [2, 4, 6, 8, 10]);
    assert.strictEqual(maxSeen <= 2, true);
  });
});
""",
        explanation="Spawn a bounded pool of asynchronous workers that pull task indices atomically from a shared cursor."
    )


def _ts_typed_debounce(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="functional",
        difficulty="medium",
        instruction="Fix debounce to pass arguments correctly to the wrapped function.",
        buggy_code="""export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timer: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { fn(); }, delayMs);
  };
}
""",
        solution_code="""export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): (...args: Parameters<T>) => void {
  let timer: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => { fn(...args); }, delayMs);
  };
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { debounce } from './solution.ts';

describe('debounce', () => {
  it('invokes with latest arguments', async () => {
    let received = '';
    const debounced = debounce((msg: string) => { received = msg; }, 10);
    debounced('first');
    debounced('second');
    await new Promise(r => setTimeout(r, 30));
    assert.strictEqual(received, 'second');
  });
});
""",
        explanation="Pass the closure's captured ...args into the callback execution."
    )


def _ts_lru_cache_generic(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix TS LRUCache to utilize Map key iteration for O(1) eviction.",
        buggy_code="""export class LRUCache<K, V> {
  private map = new Map<K, V>();
  private capacity: number;
  constructor(capacity: number) {
    this.capacity = capacity;
  }
  get(key: K): V | undefined {
    return this.map.get(key);
  }
  set(key: K, val: V): void {
    this.map.set(key, val);
  }
}
""",
        solution_code="""export class LRUCache<K, V> {
  private map = new Map<K, V>();
  private capacity: number;
  constructor(capacity: number) {
    this.capacity = capacity;
  }
  get(key: K): V | undefined {
    if (!this.map.has(key)) return undefined;
    const val = this.map.get(key)!;
    this.map.delete(key);
    this.map.set(key, val);
    return val;
  }
  set(key: K, val: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.capacity) {
      const oldestKey = this.map.keys().next().value;
      if (oldestKey !== undefined) this.map.delete(oldestKey);
    }
    this.map.set(key, val);
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { LRUCache } from './solution.ts';

describe('LRUCache TS', () => {
  it('maintains recency', () => {
    const c = new LRUCache<string, number>(2);
    c.set('a', 1);
    c.set('b', 2);
    assert.strictEqual(c.get('a'), 1);
    c.set('c', 3);
    assert.strictEqual(c.get('b'), undefined);
    assert.strictEqual(c.get('a'), 1);
    assert.strictEqual(c.get('c'), 3);
  });
});
""",
        explanation="Re-insert key on read and evict the first Map key when capacity limit is reached."
    )


def _ts_flatten_object(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix flattenObject to convert nested objects into dot-notation flat keys.",
        buggy_code="""export function flattenObject(obj: Record<string, any>, prefix = ''): Record<string, any> {
  const result: Record<string, any> = {};
  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}.${key}` : key;
    result[newKey] = value;
  }
  return result;
}
""",
        solution_code="""export function flattenObject(obj: Record<string, any>, prefix = ''): Record<string, any> {
  let result: Record<string, any> = {};
  for (const [key, value] of Object.entries(obj)) {
    const newKey = prefix ? `${prefix}.${key}` : key;
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(result, flattenObject(value, newKey));
    } else {
      result[newKey] = value;
    }
  }
  return result;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { flattenObject } from './solution.ts';

describe('flattenObject', () => {
  it('flattens nested structures', () => {
    const nested = { a: 1, b: { c: 2, d: { e: 3 } } };
    assert.deepStrictEqual(flattenObject(nested), {
      'a': 1,
      'b.c': 2,
      'b.d.e': 3
    });
  });
});
""",
        explanation="Check if value is an object and recursively merge nested keys with newKey prefix."
    )


def _ts_circular_queue(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix CircularQueue so enqueue correctly wraps around buffer when tail reaches capacity.",
        buggy_code="""export class CircularQueue<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private tail = 0;
  private count = 0;
  private capacity: number;
  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }
  enqueue(item: T): boolean {
    if (this.isFull()) return false;
    this.buffer[this.tail] = item;
    this.tail++;
    this.count++;
    return true;
  }
  dequeue(): T | undefined {
    if (this.isEmpty()) return undefined;
    const item = this.buffer[this.head];
    this.buffer[this.head] = undefined;
    this.head = (this.head + 1) % this.capacity;
    this.count--;
    return item;
  }
  isFull(): boolean { return this.count === this.capacity; }
  isEmpty(): boolean { return this.count === 0; }
}
""",
        solution_code="""export class CircularQueue<T> {
  private buffer: (T | undefined)[];
  private head = 0;
  private tail = 0;
  private count = 0;
  private capacity: number;
  constructor(capacity: number) {
    this.capacity = capacity;
    this.buffer = new Array(capacity);
  }
  enqueue(item: T): boolean {
    if (this.isFull()) return false;
    this.buffer[this.tail] = item;
    this.tail = (this.tail + 1) % this.capacity;
    this.count++;
    return true;
  }
  dequeue(): T | undefined {
    if (this.isEmpty()) return undefined;
    const item = this.buffer[this.head];
    this.buffer[this.head] = undefined;
    this.head = (this.head + 1) % this.capacity;
    this.count--;
    return item;
  }
  isFull(): boolean { return this.count === this.capacity; }
  isEmpty(): boolean { return this.count === 0; }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { CircularQueue } from './solution.ts';

describe('CircularQueue', () => {
  it('wraps around and dequeues properly', () => {
    const q = new CircularQueue<number>(2);
    assert.strictEqual(q.enqueue(1), true);
    assert.strictEqual(q.enqueue(2), true);
    assert.strictEqual(q.enqueue(3), false);
    assert.strictEqual(q.dequeue(), 1);
    assert.strictEqual(q.enqueue(3), true);
    assert.strictEqual(q.dequeue(), 2);
    assert.strictEqual(q.dequeue(), 3);
  });
});
""",
        explanation="Apply (this.tail + 1) % this.capacity when incrementing tail index."
    )


def _ts_binary_search_generic(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="algorithms",
        difficulty="easy",
        instruction="Fix binarySearch in TypeScript with comparator function.",
        buggy_code="""export function binarySearch<T>(
  arr: T[],
  target: T,
  compare: (a: T, b: T) => number
): number {
  let l = 0, r = arr.length;
  while (l < r) {
    const m = Math.floor((l + r) / 2);
    const cmp = compare(arr[m], target);
    if (cmp === 0) return m;
    if (cmp < 0) l = m;
    else r = m;
  }
  return -1;
}
""",
        solution_code="""export function binarySearch<T>(
  arr: T[],
  target: T,
  compare: (a: T, b: T) => number
): number {
  let l = 0, r = arr.length - 1;
  while (l <= r) {
    const m = Math.floor((l + r) / 2);
    const cmp = compare(arr[m], target);
    if (cmp === 0) return m;
    if (cmp < 0) l = m + 1;
    else r = m - 1;
  }
  return -1;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { binarySearch } from './solution.ts';

describe('binarySearch TS', () => {
  it('finds target index using comparator', () => {
    const nums = [10, 20, 30, 40, 50];
    const cmp = (a: number, b: number) => a - b;
    assert.strictEqual(binarySearch(nums, 30, cmp), 2);
    assert.strictEqual(binarySearch(nums, 10, cmp), 0);
    assert.strictEqual(binarySearch(nums, 50, cmp), 4);
    assert.strictEqual(binarySearch(nums, 99, cmp), -1);
  });
});
""",
        explanation="Adjust search bounds using l = m + 1 and r = m - 1 with l <= r loop condition."
    )


def _ts_bi_map(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix BiMap so setting an existing value rebinds the inverse lookup correctly.",
        buggy_code="""export class BiMap<K, V> {
  private forward = new Map<K, V>();
  private reverse = new Map<V, K>();

  set(key: K, val: V): void {
    this.forward.set(key, val);
    this.reverse.set(val, key);
  }
  getByVal(val: V): K | undefined { return this.reverse.get(val); }
  getByKey(key: K): V | undefined { return this.forward.get(key); }
}
""",
        solution_code="""export class BiMap<K, V> {
  private forward = new Map<K, V>();
  private reverse = new Map<V, K>();

  set(key: K, val: V): void {
    if (this.forward.has(key)) {
      const oldVal = this.forward.get(key)!;
      this.reverse.delete(oldVal);
    }
    if (this.reverse.has(val)) {
      const oldKey = this.reverse.get(val)!;
      this.forward.delete(oldKey);
    }
    this.forward.set(key, val);
    this.reverse.set(val, key);
  }
  getByVal(val: V): K | undefined { return this.reverse.get(val); }
  getByKey(key: K): V | undefined { return this.forward.get(key); }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { BiMap } from './solution.ts';

describe('BiMap', () => {
  it('updates reverse mapping when key value changes', () => {
    const bm = new BiMap<string, number>();
    bm.set('a', 1);
    bm.set('a', 2);
    assert.strictEqual(bm.getByKey('a'), 2);
    assert.strictEqual(bm.getByVal(1), undefined);
    assert.strictEqual(bm.getByVal(2), 'a');
  });
});
""",
        explanation="Clean up stale forward and reverse entries before establishing the new bidirectional pair."
    )


def _ts_pipe_compose(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="functional",
        difficulty="easy",
        instruction="Fix pipe to apply functions from left to right.",
        buggy_code="""export function pipe<T>(...fns: ((arg: any) => any)[]): (initial: T) => any {
  return (initial: T) => fns.reduceRight((acc, fn) => fn(acc), initial);
}
""",
        solution_code="""export function pipe<T>(...fns: ((arg: any) => any)[]): (initial: T) => any {
  return (initial: T) => fns.reduce((acc, fn) => fn(acc), initial);
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { pipe } from './solution.ts';

describe('pipe', () => {
  it('applies functions from left to right', () => {
    const add2 = (x: number) => x + 2;
    const mult3 = (x: number) => x * 3;
    const p = pipe<number>(add2, mult3);
    assert.strictEqual(p(5), 21);
  });
});
""",
        explanation="Use reduce instead of reduceRight so functions evaluate in left-to-right order."
    )


def _ts_custom_promise_all(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="medium",
        instruction="Fix allPromises to resolve immediately with an empty array when given an empty array.",
        buggy_code="""export function allPromises<T>(promises: Promise<T>[]): Promise<T[]> {
  return new Promise((resolve, reject) => {
    const results: T[] = [];
    let completed = 0;
    promises.forEach((p, i) => {
      p.then(val => {
        results[i] = val;
        completed++;
        if (completed === promises.length) resolve(results);
      }).catch(reject);
    });
  });
}
""",
        solution_code="""export function allPromises<T>(promises: Promise<T>[]): Promise<T[]> {
  return new Promise((resolve, reject) => {
    if (promises.length === 0) {
      resolve([]);
      return;
    }
    const results: T[] = new Array(promises.length);
    let completed = 0;
    promises.forEach((p, i) => {
      p.then(val => {
        results[i] = val;
        completed++;
        if (completed === promises.length) resolve(results);
      }).catch(reject);
    });
  });
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { allPromises } from './solution.ts';

describe('allPromises', () => {
  it('handles empty input immediately', async () => {
    const res = await allPromises([]);
    assert.deepStrictEqual(res, []);
  });

  it('resolves multiple values preserving order', async () => {
    const p1 = new Promise<number>(r => setTimeout(() => r(1), 10));
    const p2 = Promise.resolve(2);
    const res = await allPromises([p1, p2]);
    assert.deepStrictEqual(res, [1, 2]);
  });
});
""",
        explanation="Check if promises.length === 0 and resolve([]) immediately."
    )


def _ts_rate_limiter_token_bucket(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="algorithms",
        difficulty="medium",
        instruction="Fix TokenBucket to clamp tokens at maxTokens during refill.",
        buggy_code="""export class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  private maxTokens: number;
  private refillRatePerSec: number;
  constructor(maxTokens: number, refillRatePerSec: number, now: number) {
    this.maxTokens = maxTokens;
    this.refillRatePerSec = refillRatePerSec;
    this.tokens = maxTokens;
    this.lastRefill = now;
  }
  consume(now: number): boolean {
    const elapsedSec = (now - this.lastRefill) / 1000;
    this.tokens += elapsedSec * this.refillRatePerSec;
    this.lastRefill = now;
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }
}
""",
        solution_code="""export class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  private maxTokens: number;
  private refillRatePerSec: number;
  constructor(maxTokens: number, refillRatePerSec: number, now: number) {
    this.maxTokens = maxTokens;
    this.refillRatePerSec = refillRatePerSec;
    this.tokens = maxTokens;
    this.lastRefill = now;
  }
  consume(now: number): boolean {
    const elapsedSec = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.maxTokens, this.tokens + elapsedSec * this.refillRatePerSec);
    this.lastRefill = now;
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { TokenBucket } from './solution.ts';

describe('TokenBucket', () => {
  it('clamps tokens to max capacity', () => {
    const tb = new TokenBucket(2, 10, 0);
    tb.consume(0);
    tb.consume(0);
    assert.strictEqual(tb.consume(0), false);
    assert.strictEqual(tb.consume(100000), true);
    assert.strictEqual(tb.consume(100000), true);
    assert.strictEqual(tb.consume(100000), false);
  });
});
""",
        explanation="Clamp tokens using Math.min(this.maxTokens, ...) during refill calculations."
    )


def _ts_merge_sorted_arrays(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="algorithms",
        difficulty="easy",
        instruction="Fix mergeSorted to merge two sorted number arrays in O(n + m) time.",
        buggy_code="""export function mergeSorted(a: number[], b: number[]): number[] {
  const res: number[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) {
      res.push(a[i]);
      i++;
    } else {
      res.push(b[j]);
      j++;
    }
  }
  while (i < a.length) {
    res.push(a[i]);
    i++;
  }
  return res;
}
""",
        solution_code="""export function mergeSorted(a: number[], b: number[]): number[] {
  const res: number[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] <= b[j]) {
      res.push(a[i]);
      i++;
    } else {
      res.push(b[j]);
      j++;
    }
  }
  while (i < a.length) { res.push(a[i]); i++; }
  while (j < b.length) { res.push(b[j]); j++; }
  return res;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { mergeSorted } from './solution.ts';

describe('mergeSorted', () => {
  it('merges remaining elements from both arrays', () => {
    assert.deepStrictEqual(mergeSorted([1, 4, 7], [2, 5, 8, 9]), [1, 2, 4, 5, 7, 8, 9]);
    assert.deepStrictEqual(mergeSorted([], [1, 2]), [1, 2]);
    assert.deepStrictEqual(mergeSorted([3, 4], []), [3, 4]);
  });
});
""",
        explanation="Drain remaining elements from array b with while (j < b.length) loop."
    )


def _ts_clamp_and_paginate(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="utility",
        difficulty="easy",
        instruction="Fix paginate to return the requested page items without out-of-bound slice errors.",
        buggy_code="""export function paginate<T>(items: T[], page: number, pageSize: number): T[] {
  const start = page * pageSize;
  return items.slice(start, start + pageSize);
}
""",
        solution_code="""export function paginate<T>(items: T[], page: number, pageSize: number): T[] {
  if (page < 1 || pageSize < 1) return [];
  const start = (page - 1) * pageSize;
  return items.slice(start, start + pageSize);
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { paginate } from './solution.ts';

describe('paginate', () => {
  it('extracts correct page slice', () => {
    const list = [1, 2, 3, 4, 5, 6, 7];
    assert.deepStrictEqual(paginate(list, 1, 3), [1, 2, 3]);
    assert.deepStrictEqual(paginate(list, 2, 3), [4, 5, 6]);
    assert.deepStrictEqual(paginate(list, 3, 3), [7]);
    assert.deepStrictEqual(paginate(list, 4, 3), []);
  });
});
""",
        explanation="Calculate start index as (page - 1) * pageSize for 1-based page numbering."
    )


def _py_bounded_blocking_queue(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="concurrency",
        difficulty="hard",
        instruction="Fix the BoundedBlockingQueue class so put blocks when full and get blocks when empty using threading.Condition.",
        buggy_code="""import threading
from collections import deque

class BoundedBlockingQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = deque()
        self.lock = threading.Lock()

    def put(self, item, timeout=1.0) -> bool:
        if len(self.queue) < self.capacity:
            self.queue.append(item)
            return True
        return False

    def get(self, timeout=1.0):
        if self.queue:
            return self.queue.popleft()
        return None
""",
        solution_code="""import threading
import time
from collections import deque

class BoundedBlockingQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = deque()
        self.cond = threading.Condition()

    def put(self, item, timeout: float = 1.0) -> bool:
        with self.cond:
            end_time = time.time() + timeout
            while len(self.queue) >= self.capacity:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return False
                self.cond.wait(remaining)
            self.queue.append(item)
            self.cond.notify_all()
            return True

    def get(self, timeout: float = 1.0):
        with self.cond:
            end_time = time.time() + timeout
            while not self.queue:
                remaining = end_time - time.time()
                if remaining <= 0:
                    return None
                self.cond.wait(remaining)
            item = self.queue.popleft()
            self.cond.notify_all()
            return item
""",
        test_code="""from solution import BoundedBlockingQueue
import threading
import time

def test_bounded_queue_basic():
    q = BoundedBlockingQueue(2)
    assert q.put(10) is True
    assert q.put(20) is True
    assert q.put(30, timeout=0.05) is False
    assert q.get() == 10
    assert q.put(30, timeout=0.05) is True
    assert q.get() == 20
    assert q.get() == 30
    assert q.get(timeout=0.05) is None

def test_bounded_queue_concurrent():
    q = BoundedBlockingQueue(3)
    consumed = []
    def consumer():
        for _ in range(5):
            val = q.get(timeout=2.0)
            if val is not None:
                consumed.append(val)
    t = threading.Thread(target=consumer)
    t.start()
    for i in range(5):
        time.sleep(0.01)
        q.put(i)
    t.join()
    assert consumed == [0, 1, 2, 3, 4]
""",
        explanation="Use threading.Condition with wait(timeout) and notify_all() to synchronize producers and consumers correctly."
    )


def _py_ttl_cache_ms(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="medium",
        instruction="Fix the TTLCache so keys expire after ttl_ms and get/cleanup properly evicts expired entries.",
        buggy_code="""import time

class TTLCache:
    def __init__(self, default_ttl_ms: int = 1000):
        self.default_ttl_ms = default_ttl_ms
        self.store = {}

    def put(self, key, value, ttl_ms=None):
        ttl = ttl_ms if ttl_ms is not None else self.default_ttl_ms
        self.store[key] = (value, time.time() + ttl)

    def get(self, key):
        if key in self.store:
            val, _ = self.store[key]
            return val
        return None
""",
        solution_code="""import time

class TTLCache:
    def __init__(self, default_ttl_ms: int = 1000):
        self.default_ttl_ms = default_ttl_ms
        self.store = {}

    def put(self, key, value, ttl_ms=None):
        ttl = ttl_ms if ttl_ms is not None else self.default_ttl_ms
        expire_at = time.time() + (ttl / 1000.0)
        self.store[key] = (value, expire_at)

    def get(self, key):
        if key not in self.store:
            return None
        val, expire_at = self.store[key]
        if time.time() >= expire_at:
            del self.store[key]
            return None
        return val

    def cleanup(self) -> int:
        now = time.time()
        expired = [k for k, (_, exp) in self.store.items() if now >= exp]
        for k in expired:
            del self.store[k]
        return len(expired)
""",
        test_code="""from solution import TTLCache
import time

def test_ttl_cache():
    cache = TTLCache(default_ttl_ms=50)
    cache.put("a", 100)
    assert cache.get("a") == 100
    cache.put("b", 200, ttl_ms=300)
    time.sleep(0.08)
    assert cache.get("a") is None
    assert cache.get("b") == 200
    assert cache.cleanup() == 0
""",
        explanation="Convert ttl_ms to seconds for time.time() calculation and evict expired keys on get() and cleanup()."
    )


def _py_trie_wildcard(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix the WordDictionary to support adding words and searching with the '.' wildcard character.",
        buggy_code="""class WordDictionary:
    def __init__(self):
        self.root = {}

    def add_word(self, word: str) -> None:
        curr = self.root
        for ch in word:
            if ch not in curr:
                curr[ch] = {}
            curr = curr[ch]
        curr["#"] = True

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return "#" in node
            ch = word[idx]
            if ch == ".":
                return any(dfs(node[child], idx + 1) for child in node)
            if ch in node:
                return dfs(node[ch], idx + 1)
            return False
        return dfs(self.root, 0)
""",
        solution_code="""class WordDictionary:
    def __init__(self):
        self.root = {}

    def add_word(self, word: str) -> None:
        curr = self.root
        for ch in word:
            if ch not in curr:
                curr[ch] = {}
            curr = curr[ch]
        curr["#"] = True

    def search(self, word: str) -> bool:
        def dfs(node, idx):
            if idx == len(word):
                return "#" in node
            ch = word[idx]
            if ch == ".":
                for child, child_node in node.items():
                    if child != "#" and dfs(child_node, idx + 1):
                        return True
                return False
            if ch in node:
                return dfs(node[ch], idx + 1)
            return False
        return dfs(self.root, 0)
""",
        test_code="""from solution import WordDictionary

def test_word_dictionary_wildcard():
    wd = WordDictionary()
    wd.add_word("bad")
    wd.add_word("dad")
    wd.add_word("mad")
    assert wd.search("pad") is False
    assert wd.search("bad") is True
    assert wd.search(".ad") is True
    assert wd.search("b..") is True
    assert wd.search("b...") is False
""",
        explanation="When matching wildcard '.', ignore the end-of-word marker '#' to avoid recursion into boolean values."
    )


def _py_sliding_window_rate_limiter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="system-design",
        difficulty="medium",
        instruction="Fix the SlidingWindowRateLimiter so allow_request evicts old timestamps and strictly enforces max_requests within window_seconds.",
        buggy_code="""from collections import deque
import time

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow_request(self, now: float = None) -> bool:
        if now is None:
            now = time.time()
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(now)
            return True
        return False
""",
        solution_code="""from collections import deque
import time

class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps = deque()

    def allow_request(self, now: float = None) -> bool:
        if now is None:
            now = time.time()
        cutoff = now - self.window_seconds
        while self.timestamps and self.timestamps[0] <= cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(now)
            return True
        return False
""",
        test_code="""from solution import SlidingWindowRateLimiter

def test_rate_limiter_sliding_window():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=10.0)
    t0 = 100.0
    assert limiter.allow_request(t0) is True
    assert limiter.allow_request(t0 + 2.0) is True
    assert limiter.allow_request(t0 + 4.0) is True
    assert limiter.allow_request(t0 + 6.0) is False
    assert limiter.allow_request(t0 + 11.0) is True
    assert limiter.allow_request(t0 + 12.5) is True
""",
        explanation="Evict timestamps older than (now - window_seconds) before checking remaining capacity."
    )


def _py_pubsub_broker(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="system-design",
        difficulty="medium",
        instruction="Fix the PubSubBroker so subscribers receive messages on exact topic match or wildcard topics (e.g. 'events.*').",
        buggy_code="""class PubSubBroker:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic: str, callback) -> None:
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, message) -> int:
        count = 0
        if topic in self.subscribers:
            for cb in self.subscribers[topic]:
                cb(message)
                count += 1
        return count
""",
        solution_code="""import fnmatch

class PubSubBroker:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, pattern: str, callback) -> None:
        if pattern not in self.subscribers:
            self.subscribers[pattern] = []
        self.subscribers[pattern].append(callback)

    def unsubscribe(self, pattern: str, callback) -> bool:
        if pattern in self.subscribers and callback in self.subscribers[pattern]:
            self.subscribers[pattern].remove(callback)
            return True
        return False

    def publish(self, topic: str, message) -> int:
        count = 0
        for pattern, callbacks in self.subscribers.items():
            if pattern == topic or fnmatch.fnmatch(topic, pattern):
                for cb in callbacks:
                    cb(message)
                    count += 1
        return count
""",
        test_code="""from solution import PubSubBroker

def test_pubsub_broker():
    broker = PubSubBroker()
    received_all = []
    received_order = []
    broker.subscribe("order.*", lambda m: received_order.append(m))
    broker.subscribe("*", lambda m: received_all.append(m))

    count = broker.publish("order.created", {"id": 1})
    assert count == 2
    assert len(received_order) == 1
    assert len(received_all) == 1

    broker.publish("user.login", {"user": "alice"})
    assert len(received_order) == 1
    assert len(received_all) == 2
""",
        explanation="Use fnmatch pattern matching to support wildcard subscriptions alongside exact matches."
    )


def _ts_reactive_signal(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="architecture",
        difficulty="hard",
        instruction="Fix createSignal and createEffect to track reactive dependencies dynamically and re-run effects on change.",
        buggy_code="""let currentEffect: (() => void) | null = null;

export function createSignal<T>(initialValue: T): [() => T, (val: T) => void] {
  let val = initialValue;
  const read = () => val;
  const write = (newVal: T) => { val = newVal; };
  return [read, write];
}

export function createEffect(fn: () => void): void {
  currentEffect = fn;
  fn();
  currentEffect = null;
}
""",
        solution_code="""let currentEffect: (() => void) | null = null;

export function createSignal<T>(initialValue: T): [() => T, (val: T) => void] {
  let val = initialValue;
  const subscribers = new Set<() => void>();

  const read = (): T => {
    if (currentEffect) {
      subscribers.add(currentEffect);
    }
    return val;
  };

  const write = (newVal: T): void => {
    if (val !== newVal) {
      val = newVal;
      const subs = Array.from(subscribers);
      for (const sub of subs) {
        sub();
      }
    }
  };

  return [read, write];
}

export function createEffect(fn: () => void): void {
  const run = () => {
    currentEffect = run;
    try {
      fn();
    } finally {
      currentEffect = null;
    }
  };
  run();
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { createSignal, createEffect } from './solution.ts';

describe('Reactive Signals', () => {
  it('updates downstream effects when signals change', () => {
    const [count, setCount] = createSignal(1);
    let log = 0;
    createEffect(() => {
      log = count() * 2;
    });

    assert.strictEqual(log, 2);
    setCount(5);
    assert.strictEqual(log, 10);
    setCount(10);
    assert.strictEqual(log, 20);
  });
});
""",
        explanation="Register active subscriber in createSignal reader and notify all subscribers on value change."
    )


def _ts_transactional_store(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix TransactionalStore to support transactions with commit and rollback capabilities.",
        buggy_code="""export class TransactionalStore<K, V> {
  private store = new Map<K, V>();
  private activeTx: Map<K, V> | null = null;

  get(key: K): V | undefined {
    return this.store.get(key);
  }

  set(key: K, value: V): void {
    this.store.set(key, value);
  }

  begin(): void {
    this.activeTx = new Map();
  }

  commit(): boolean {
    this.activeTx = null;
    return true;
  }

  rollback(): boolean {
    this.activeTx = null;
    return true;
  }
}
""",
        solution_code="""export class TransactionalStore<K, V> {
  private store = new Map<K, V>();
  private activeTx: Map<K, V | null> | null = null;

  get(key: K): V | undefined {
    if (this.activeTx && this.activeTx.has(key)) {
      const val = this.activeTx.get(key);
      return val === null ? undefined : val;
    }
    return this.store.get(key);
  }

  set(key: K, value: V): void {
    if (this.activeTx) {
      this.activeTx.set(key, value);
    } else {
      this.store.set(key, value);
    }
  }

  begin(): void {
    this.activeTx = new Map();
  }

  commit(): boolean {
    if (!this.activeTx) return false;
    for (const [k, v] of this.activeTx.entries()) {
      if (v === null) {
        this.store.delete(k);
      } else {
        this.store.set(k, v);
      }
    }
    this.activeTx = null;
    return true;
  }

  rollback(): boolean {
    if (!this.activeTx) return false;
    this.activeTx = null;
    return true;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { TransactionalStore } from './solution.ts';

describe('TransactionalStore', () => {
  it('supports commit and rollback', () => {
    const store = new TransactionalStore<string, number>();
    store.set('a', 10);
    assert.strictEqual(store.get('a'), 10);

    store.begin();
    store.set('a', 20);
    store.set('b', 30);
    assert.strictEqual(store.get('a'), 20);
    assert.strictEqual(store.get('b'), 30);
    store.rollback();

    assert.strictEqual(store.get('a'), 10);
    assert.strictEqual(store.get('b'), undefined);

    store.begin();
    store.set('c', 100);
    store.commit();
    assert.strictEqual(store.get('c'), 100);
  });
});
""",
        explanation="Buffer changes inside activeTx during a transaction, applying to master store only on commit and discarding on rollback."
    )


def _ts_stream_chunk_transformer(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="utility",
        difficulty="medium",
        instruction="Fix DelimitedStreamTransformer to buffer incoming data chunks and split cleanly across custom delimiters.",
        buggy_code="""export class DelimitedStreamTransformer {
  private delimiter: string;

  constructor(delimiter: string = '\\n') {
    this.delimiter = delimiter;
  }

  transform(chunk: string): string[] {
    return chunk.split(this.delimiter);
  }

  flush(): string[] {
    return [];
  }
}
""",
        solution_code="""export class DelimitedStreamTransformer {
  private delimiter: string;
  private buffer: string = '';

  constructor(delimiter: string = '\\n') {
    this.delimiter = delimiter;
  }

  transform(chunk: string): string[] {
    this.buffer += chunk;
    const parts = this.buffer.split(this.delimiter);
    this.buffer = parts.pop() || '';
    return parts;
  }

  flush(): string[] {
    if (this.buffer.length > 0) {
      const remaining = [this.buffer];
      this.buffer = '';
      return remaining;
    }
    return [];
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { DelimitedStreamTransformer } from './solution.ts';

describe('DelimitedStreamTransformer', () => {
  it('handles chunk splitting across message boundaries', () => {
    const transformer = new DelimitedStreamTransformer('\\n');
    assert.deepStrictEqual(transformer.transform('hello\\nwor'), ['hello']);
    assert.deepStrictEqual(transformer.transform('ld\\nfoo\\nbar'), ['world', 'foo']);
    assert.deepStrictEqual(transformer.flush(), ['bar']);
    assert.deepStrictEqual(transformer.flush(), []);
  });
});
""",
        explanation="Maintain internal buffer across chunks, retaining the incomplete tail and flushing residual on stream completion."
    )


def _ts_deep_object_diff(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="utility",
        difficulty="medium",
        instruction="Fix deepDiff to calculate added, removed, and updated property paths between two objects.",
        buggy_code="""export interface DiffResult {
  added: Record<string, any>;
  removed: Record<string, any>;
  updated: Record<string, { from: any; to: any }>;
}

export function deepDiff(obj1: Record<string, any>, obj2: Record<string, any>): DiffResult {
  return { added: {}, removed: {}, updated: {} };
}
""",
        solution_code="""export interface DiffResult {
  added: Record<string, any>;
  removed: Record<string, any>;
  updated: Record<string, { from: any; to: any }>;
}

export function deepDiff(obj1: Record<string, any>, obj2: Record<string, any>): DiffResult {
  const result: DiffResult = { added: {}, removed: {}, updated: {} };

  const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);
  for (const key of allKeys) {
    if (!(key in obj1)) {
      result.added[key] = obj2[key];
    } else if (!(key in obj2)) {
      result.removed[key] = obj1[key];
    } else if (obj1[key] !== obj2[key]) {
      result.updated[key] = { from: obj1[key], to: obj2[key] };
    }
  }

  return result;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { deepDiff } from './solution.ts';

describe('deepDiff', () => {
  it('accurately identifies added, removed, and modified keys', () => {
    const oldObj = { a: 1, b: 2, c: 3 };
    const newObj = { b: 20, c: 3, d: 4 };

    const diff = deepDiff(oldObj, newObj);
    assert.deepStrictEqual(diff.added, { d: 4 });
    assert.deepStrictEqual(diff.removed, { a: 1 });
    assert.deepStrictEqual(diff.updated, { b: { from: 2, to: 20 } });
  });
});
""",
        explanation="Iterate combined key set and classify presence/changes into added, removed, and updated buckets."
    )


def _ts_cancellable_task_pool(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="concurrency",
        difficulty="hard",
        instruction="Fix runWithConcurrency to execute async tasks respecting concurrency limit and return results in order.",
        buggy_code="""export async function runWithConcurrency<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  return Promise.all(tasks.map(t => t()));
}
""",
        solution_code="""export async function runWithConcurrency<T>(tasks: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let currentIndex = 0;

  const workers = Array.from({ length: Math.min(limit, tasks.length) }, async () => {
    while (currentIndex < tasks.length) {
      const idx = currentIndex++;
      results[idx] = await tasks[idx]();
    }
  });

  await Promise.all(workers);
  return results;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { runWithConcurrency } from './solution.ts';

describe('runWithConcurrency', () => {
  it('limits active concurrent tasks while preserving order', async () => {
    let active = 0;
    let maxActive = 0;

    const makeTask = (val: number, delayMs: number) => async () => {
      active++;
      maxActive = Math.max(maxActive, active);
      await new Promise(r => setTimeout(r, delayMs));
      active--;
      return val * 10;
    };

    const tasks = [
      makeTask(1, 30),
      makeTask(2, 20),
      makeTask(3, 10),
      makeTask(4, 25),
    ];

    const results = await runWithConcurrency(tasks, 2);
    assert.deepStrictEqual(results, [10, 20, 30, 40]);
    assert.ok(maxActive <= 2, `maxActive ${maxActive} exceeded limit 2`);
  });
});
""",
        explanation="Spawn a pool of worker loops sharing an atomic index cursor to enforce concurrent limit and maintain result ordering."
    )


def _py_graph_bfs_shortest_path(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix shortest_path to return the shortest path list of nodes between start and target in an unweighted graph using BFS.",
        buggy_code="""from collections import deque

def shortest_path(graph: dict, start: str, target: str):
    if start == target:
        return [start]
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return [node]
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return None
""",
        solution_code="""from collections import deque

def shortest_path(graph: dict, start: str, target: str):
    if start == target:
        return [start]
    visited = {start}
    queue = deque([(start, [start])])
    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None
""",
        test_code="""from solution import shortest_path

def test_shortest_path():
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"]
    }
    assert shortest_path(graph, "A", "F") == ["A", "C", "F"]
    assert shortest_path(graph, "A", "D") == ["A", "B", "D"]
    assert shortest_path(graph, "A", "A") == ["A"]
    assert shortest_path(graph, "D", "Z") is None
""",
        explanation="Track current path along with current node in BFS queue to reconstruct the shortest route upon reaching target."
    )


def _py_monotonic_increasing_stack(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="medium",
        instruction="Fix daily_temperatures so it returns an array where ans[i] is the number of days until a warmer temperature using a monotonic stack.",
        buggy_code="""def daily_temperatures(temperatures: list) -> list:
    res = [0] * len(temperatures)
    stack = []
    for i, t in enumerate(temperatures):
        while stack and temperatures[stack[-1]] <= t:
            stack.pop()
        stack.append(i)
    return res
""",
        solution_code="""def daily_temperatures(temperatures: list) -> list:
    res = [0] * len(temperatures)
    stack = []
    for i, t in enumerate(temperatures):
        while stack and t > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            res[prev_idx] = i - prev_idx
        stack.append(i)
    return res
""",
        test_code="""from solution import daily_temperatures

def test_daily_temperatures():
    temps = [73, 74, 75, 71, 69, 72, 76, 73]
    assert daily_temperatures(temps) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert daily_temperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert daily_temperatures([30, 30, 30]) == [0, 0, 0]
    assert daily_temperatures([]) == []
""",
        explanation="When a warmer temperature is encountered, pop indices from stack and record the day difference (i - prev_idx)."
    )


def _py_min_heap_priority_queue(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="data-structures",
        difficulty="medium",
        instruction="Fix PriorityTaskQueue to return tasks in ascending priority order (lowest priority integer first), breaking ties by FIFO order.",
        buggy_code="""import heapq

class PriorityTaskQueue:
    def __init__(self):
        self.heap = []

    def push(self, task: str, priority: int):
        heapq.heappush(self.heap, (priority, task))

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]
        return None
""",
        solution_code="""import heapq

class PriorityTaskQueue:
    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, task: str, priority: int) -> None:
        heapq.heappush(self.heap, (priority, self.counter, task))
        self.counter += 1

    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[2]
        return None

    def __len__(self):
        return len(self.heap)
""",
        test_code="""from solution import PriorityTaskQueue

def test_priority_queue():
    pq = PriorityTaskQueue()
    pq.push("task_z", 5)  # pushed first
    pq.push("task_high", 1)
    pq.push("task_a", 5)  # pushed second
    pq.push("task_low", 10)

    assert pq.pop() == "task_high"
    assert pq.pop() == "task_z"  # FIFO tie-break must return task_z before task_a
    assert pq.pop() == "task_a"
    assert pq.pop() == "task_low"
    assert pq.pop() is None
""",
        explanation="Include an insertion counter in the heap tuple (priority, counter, task) to enforce deterministic FIFO order for identical priorities."
    )


def _py_token_bucket_limiter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_py_{idx:04d}",
        language=Language.PYTHON,
        category="system-design",
        difficulty="medium",
        instruction="Fix the TokenBucketRateLimiter so tokens refill smoothly based on elapsed time and consumption respects capacity limits.",
        buggy_code="""import time

class TokenBucketRateLimiter:
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = capacity
        self.rate = refill_rate_per_sec
        self.tokens = capacity
        self.last_refill = time.time()

    def allow(self, tokens: float = 1.0, now: float = None) -> bool:
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
""",
        solution_code="""import time

class TokenBucketRateLimiter:
    def __init__(self, capacity: float, refill_rate_per_sec: float):
        self.capacity = float(capacity)
        self.rate = float(refill_rate_per_sec)
        self.tokens = float(capacity)
        self.last_refill = None

    def allow(self, tokens: float = 1.0, now: float = None) -> bool:
        if now is None:
            now = time.time()
        if self.last_refill is None:
            self.last_refill = now

        elapsed = max(0.0, now - self.last_refill)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
""",
        test_code="""from solution import TokenBucketRateLimiter

def test_token_bucket():
    tb = TokenBucketRateLimiter(capacity=5.0, refill_rate_per_sec=2.0)
    t0 = 1000.0
    assert tb.allow(tokens=3.0, now=t0) is True
    assert tb.allow(tokens=3.0, now=t0) is False
    assert tb.allow(tokens=2.0, now=t0) is True
    assert tb.allow(tokens=1.0, now=t0 + 0.5) is True
    assert tb.allow(tokens=1.0, now=t0 + 0.5) is False
    assert tb.allow(tokens=5.0, now=t0 + 5.0) is True
""",
        explanation="Refill tokens dynamically as elapsed * rate (clamped at capacity) on every allow call before checking balance."
    )


def _ts_lfu_cache(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="hard",
        instruction="Fix LFUCache to evict the least frequently used key when capacity is exceeded, breaking frequency ties by LRU recency.",
        buggy_code="""export class LFUCache<K, V> {
  private capacity: number;
  private store = new Map<K, V>();

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    return this.store.get(key);
  }

  put(key: K, value: V): void {
    if (this.capacity <= 0) return;
    if (this.store.size >= this.capacity && !this.store.has(key)) {
      const firstKey = this.store.keys().next().value;
      if (firstKey !== undefined) this.store.delete(firstKey);
    }
    this.store.set(key, value);
  }
}
""",
        solution_code="""export class LFUCache<K, V> {
  private capacity: number;
  private values = new Map<K, V>();
  private counts = new Map<K, number>();
  private freqMap = new Map<number, Set<K>>();
  private minFreq = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    if (!this.values.has(key)) return undefined;
    this.incrementFreq(key);
    return this.values.get(key);
  }

  put(key: K, value: V): void {
    if (this.capacity <= 0) return;

    if (this.values.has(key)) {
      this.values.set(key, value);
      this.incrementFreq(key);
      return;
    }

    if (this.values.size >= this.capacity) {
      const minSet = this.freqMap.get(this.minFreq);
      if (minSet) {
        const evictKey = minSet.values().next().value;
        if (evictKey !== undefined) {
          minSet.delete(evictKey);
          this.values.delete(evictKey);
          this.counts.delete(evictKey);
        }
      }
    }

    this.values.set(key, value);
    this.counts.set(key, 1);
    this.minFreq = 1;
    if (!this.freqMap.has(1)) this.freqMap.set(1, new Set());
    this.freqMap.get(1)!.add(key);
  }

  private incrementFreq(key: K): void {
    const count = this.counts.get(key)!;
    this.counts.set(key, count + 1);
    const oldSet = this.freqMap.get(count)!;
    oldSet.delete(key);
    if (this.minFreq === count && oldSet.size === 0) {
      this.minFreq++;
    }
    if (!this.freqMap.has(count + 1)) this.freqMap.set(count + 1, new Set());
    this.freqMap.get(count + 1)!.add(key);
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { LFUCache } from './solution.ts';

describe('LFUCache', () => {
  it('evicts least frequently used key and breaks ties by LRU recency', () => {
    const lfu = new LFUCache<number, number>(2);
    lfu.put(1, 10);
    lfu.put(2, 20);
    assert.strictEqual(lfu.get(1), 10);

    lfu.put(3, 30);
    assert.strictEqual(lfu.get(2), undefined);
    assert.strictEqual(lfu.get(3), 30);

    lfu.put(4, 40);
    assert.strictEqual(lfu.get(1), undefined);
    assert.strictEqual(lfu.get(3), 30);
    assert.strictEqual(lfu.get(4), 40);
  });
});
""",
        explanation="Track frequency counts with frequency bucket Sets and maintain minFreq pointer to evict lowest frequency key with LRU tie-breaking."
    )


def _ts_event_bus_typed(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="architecture",
        difficulty="medium",
        instruction="Fix TypedEventBus to support registering, invoking, and unsubscribing typed event listeners.",
        buggy_code="""export class TypedEventBus<Events extends Record<string, any>> {
  private listeners: Record<string, ((payload: any) => void)[]> = {};

  on<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void {
    const ev = String(event);
    if (!this.listeners[ev]) this.listeners[ev] = [];
    this.listeners[ev].push(handler);
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    const ev = String(event);
    if (this.listeners[ev]) {
      this.listeners[ev].forEach(h => h(payload));
    }
  }

  off<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void {
    const ev = String(event);
    this.listeners[ev] = [];
  }
}
""",
        solution_code="""export class TypedEventBus<Events extends Record<string, any>> {
  private listeners: Map<string, Set<(payload: any) => void>> = new Map();

  on<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): () => void {
    const ev = String(event);
    if (!this.listeners.has(ev)) {
      this.listeners.set(ev, new Set());
    }
    this.listeners.get(ev)!.add(handler);
    return () => this.off(event, handler);
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    const ev = String(event);
    const handlers = this.listeners.get(ev);
    if (handlers) {
      Array.from(handlers).forEach(h => h(payload));
    }
  }

  off<K extends keyof Events>(event: K, handler: (payload: Events[K]) => void): void {
    const ev = String(event);
    const handlers = this.listeners.get(ev);
    if (handlers) {
      handlers.delete(handler);
    }
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { TypedEventBus } from './solution.ts';

interface AppEvents {
  userLogin: { username: string; id: number };
  logout: void;
}

describe('TypedEventBus', () => {
  it('subscribes, emits, and safely unsubscribes specific handlers', () => {
    const bus = new TypedEventBus<AppEvents>();
    const logins: string[] = [];

    const unsubscribe = bus.on('userLogin', (data) => {
      logins.push(data.username);
    });

    bus.emit('userLogin', { username: 'alice', id: 1 });
    bus.emit('userLogin', { username: 'bob', id: 2 });
    assert.deepStrictEqual(logins, ['alice', 'bob']);

    unsubscribe();
    bus.emit('userLogin', { username: 'charlie', id: 3 });
    assert.deepStrictEqual(logins, ['alice', 'bob']);
  });
});
""",
        explanation="Use Map<string, Set<Handler>> to delete only the specified handler in off() and return an unsubscribe callback from on()."
    )


def _ts_sorted_interval_insert(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="algorithms",
        difficulty="medium",
        instruction="Fix insertInterval to insert a new interval into non-overlapping sorted intervals and merge overlaps in O(n).",
        buggy_code="""export function insertInterval(intervals: [number, number][], newInterval: [number, number]): [number, number][] {
  intervals.push(newInterval);
  return intervals;
}
""",
        solution_code="""export function insertInterval(intervals: [number, number][], newInterval: [number, number]): [number, number][] {
  const result: [number, number][] = [];
  let i = 0;
  const n = intervals.length;

  while (i < n && intervals[i][1] < newInterval[0]) {
    result.push(intervals[i]);
    i++;
  }

  let start = newInterval[0];
  let end = newInterval[1];
  while (i < n && intervals[i][0] <= end) {
    start = Math.min(start, intervals[i][0]);
    end = Math.max(end, intervals[i][1]);
    i++;
  }
  result.push([start, end]);

  while (i < n) {
    result.push(intervals[i]);
    i++;
  }

  return result;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { insertInterval } from './solution.ts';

describe('insertInterval', () => {
  it('inserts and merges overlapping intervals', () => {
    const list1: [number, number][] = [[1, 3], [6, 9]];
    assert.deepStrictEqual(insertInterval(list1, [2, 5]), [[1, 5], [6, 9]]);

    const list2: [number, number][] = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]];
    assert.deepStrictEqual(insertInterval(list2, [4, 8]), [[1, 2], [3, 10], [12, 16]]);

    assert.deepStrictEqual(insertInterval([], [5, 7]), [[5, 7]]);
  });
});
""",
        explanation="Partition into before intervals, merge overlapping intervals into [minStart, maxEnd], and append after intervals."
    )


def _ts_async_retry_exponential(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"sample_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="concurrency",
        difficulty="medium",
        instruction="Fix retryWithBackoff to retry a failing async function up to maxRetries with exponential backoff delay.",
        buggy_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 10
): Promise<T> {
  return fn();
}
""",
        solution_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 10
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      if (attempt > maxRetries) {
        throw err;
      }
      const delay = baseDelayMs * Math.pow(2, attempt - 1);
      await new Promise(res => setTimeout(res, delay));
    }
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { retryWithBackoff } from './solution.ts';

describe('retryWithBackoff', () => {
  it('retries until success within maxRetries limit', async () => {
    let attempts = 0;
    const flakyTask = async () => {
      attempts++;
      if (attempts < 3) throw new Error('temporary failure');
      return 'success';
    };

    const res = await retryWithBackoff(flakyTask, 3, 5);
    assert.strictEqual(res, 'success');
    assert.strictEqual(attempts, 3);
  });

  it('throws final error if maxRetries is exceeded', async () => {
    let attempts = 0;
    const failingTask = async () => {
      attempts++;
      throw new Error('permanent error');
    };

    await assert.rejects(
      async () => retryWithBackoff(failingTask, 2, 5),
      /permanent error/
    );
    assert.strictEqual(attempts, 3);
  });
});
""",
        explanation="Catch errors in a retry loop, calculate exponential delay as baseDelay * 2^(attempt-1), and rethrow on exceeding maxRetries."
    )
