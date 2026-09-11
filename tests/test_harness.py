import sys
from pathlib import Path

# Add apex-coder root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schemas import DatasetSample, Language
from harness.verifier import CodeVerifier


def test_python_valid_sample():
    verifier = CodeVerifier(use_docker=False)

    sample = DatasetSample(
        sample_id="sample_py_001",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="easy",
        instruction="Fix the binary search function to correctly return the index of the target or -1 if not found.",
        buggy_code="""
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid
        else:
            right = mid
    return -1
""",
        solution_code="""
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        test_code="""
from solution import binary_search

def test_binary_search_found():
    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 7) == 3
    assert binary_search(arr, 11) == 5

def test_binary_search_not_found():
    arr = [1, 3, 5, 7, 9, 11]
    assert binary_search(arr, 0) == -1
    assert binary_search(arr, 6) == -1
    assert binary_search(arr, 15) == -1

def test_empty():
    assert binary_search([], 42) == -1
""",
    )

    outcome = verifier.verify_sample(sample)
    print(f"[Python Valid Test] Valid: {outcome.is_valid}, Reason: {outcome.rejection_reason}")
    if not outcome.is_valid:
        print("POS STDOUT:", outcome.positive_check.stdout)
        print("POS STDERR:", outcome.positive_check.stderr)
        print("NEG STDOUT:", outcome.negative_check.stdout)
        print("NEG STDERR:", outcome.negative_check.stderr)
    assert outcome.is_valid is True
    assert outcome.negative_check.status.value in ["failed", "timeout"]
    assert outcome.positive_check.status.value == "passed"


def test_python_broken_fix():
    verifier = CodeVerifier(use_docker=False)

    sample = DatasetSample(
        sample_id="sample_py_002",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="easy",
        instruction="Fix the function",
        buggy_code="""
def add(a, b):
    return a - b
""",
        solution_code="""
def add(a, b):
    return a * b  # Still wrong!
""",
        test_code="""
from solution import add

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
""",
    )

    outcome = verifier.verify_sample(sample)
    print(f"[Python Broken Fix Test] Valid: {outcome.is_valid}, Reason: {outcome.rejection_reason}")
    assert outcome.is_valid is False
    assert "Broken Fix" in (outcome.rejection_reason or "")


def test_python_trivial_test():
    verifier = CodeVerifier(use_docker=False)

    sample = DatasetSample(
        sample_id="sample_py_003",
        language=Language.PYTHON,
        category="algorithms",
        difficulty="easy",
        instruction="Fix the function",
        buggy_code="""
def add(a, b):
    return a - b
""",
        solution_code="""
def add(a, b):
    return a + b
""",
        test_code="""
from solution import add

def test_vacuous():
    assert True
""",
    )

    outcome = verifier.verify_sample(sample)
    print(f"[Python Vacuous Test] Valid: {outcome.is_valid}, Reason: {outcome.rejection_reason}")
    assert outcome.is_valid is False
    assert "Trivial/Invalid Test" in (outcome.rejection_reason or "")


def test_typescript_valid_sample():
    verifier = CodeVerifier(use_docker=False)

    sample = DatasetSample(
        sample_id="sample_ts_001",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="easy",
        instruction="Fix the TypeScript Stack class so that pop returns undefined when empty instead of throwing an unhandled exception, and peek returns the top element without removing it.",
        buggy_code="""
export class Stack<T> {
  private items: T[] = [];

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items.pop();
  }

  size(): number {
    return this.items.length;
  }
}
""",
        solution_code="""
export class Stack<T> {
  private items: T[] = [];

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items.length > 0 ? this.items[this.items.length - 1] : undefined;
  }

  size(): number {
    return this.items.length;
  }
}
""",
        test_code="""
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Stack } from './solution.ts';

describe('Stack tests', () => {
  it('handles push, peek, and pop correctly', () => {
    const stack = new Stack<number>();
    stack.push(10);
    stack.push(20);
    
    assert.strictEqual(stack.peek(), 20);
    assert.strictEqual(stack.size(), 2);
    assert.strictEqual(stack.pop(), 20);
    assert.strictEqual(stack.size(), 1);
    assert.strictEqual(stack.peek(), 10);
    assert.strictEqual(stack.pop(), 10);
    assert.strictEqual(stack.size(), 0);
    assert.strictEqual(stack.peek(), undefined);
    assert.strictEqual(stack.pop(), undefined);
  });
});
""",
    )

    outcome = verifier.verify_sample(sample)
    print(f"[TypeScript Valid Test] Valid: {outcome.is_valid}, Reason: {outcome.rejection_reason}")
    if not outcome.is_valid:
        print("TS POS STDOUT:", outcome.positive_check.stdout)
        print("TS POS STDERR:", outcome.positive_check.stderr)
        print("TS NEG STDOUT:", outcome.negative_check.stdout)
        print("TS NEG STDERR:", outcome.negative_check.stderr)
    assert outcome.is_valid is True
    assert outcome.negative_check.status.value in ["failed", "timeout"]
    assert outcome.positive_check.status.value == "passed"


if __name__ == "__main__":
    test_python_valid_sample()
    test_python_broken_fix()
    test_python_trivial_test()
    test_typescript_valid_sample()
    print("\nAll verification harness checks passed successfully!")
