import random
from typing import List, Tuple

PYTHON_TOPICS = [
    ("algorithms", "binary search edge cases, off-by-one errors, two pointer logic"),
    ("data-structures", "custom LRU cache, heap implementation, trie prefix search"),
    ("asyncio", "asyncio tasks cancellation, race condition in shared state, gathering results"),
    ("generators", "generator state exhaustion, yield vs return in recursive generators"),
    ("typing", "generic type variance, Union narrowing, TypedDict optional keys"),
    ("collections", "defaultdict mutation during iteration, deepcopy vs shallow copy"),
    ("string-processing", "regex group capturing, unicode normalization, string tokenization"),
    ("math-logic", "integer overflow emulation, matrix rotation, interval merging"),
]

TYPESCRIPT_TOPICS = [
    ("generics", "generic constraints, mapped types, keyof lookup type preservation"),
    ("async-await", "Promise.all error rejection handling, retry with exponential backoff"),
    ("type-narrowing", "discriminated unions, custom type guards, exhaustive switch checks"),
    ("data-structures", "circular queue, doubly linked list, priority queue"),
    ("immutability", "deep object freeze, nested state updater without mutation"),
    ("event-emitter", "typed event emitter with once/off listener cleanup"),
    ("functional", "currying with strict parameter typing, compose/pipe functions"),
    ("validation", "runtime schema parser/validator with type inference"),
]

SYSTEM_PROMPT = """You are an expert software engineer and test-driven development specialist.
Your task is to produce a high quality code-fix challenge with a robust, non-vacuous unit test suite.

Rules:
1. Architecture & Statefulness: Rate limiters, queues, buffers, caches, and concurrency controllers MUST be implemented as a stateful class holding mutable internal state across method calls.
2. Constructor Invariant: Always assign constructor arguments directly to matching instance fields (e.g. this.maxRequests = maxRequests; this.windowMs = windowMs). Never replace windowMs with Date.now() or unrelated clock math.
3. Sliding Log Invariant & Algorithm Invariant: In allowRequest(now = Date.now()): Strictly follow the sliding log array pattern (cutoff = now - windowMs, filter(ts => ts > cutoff)). Do not invent token refill logic or custom bucket offsets.
   Step 1: Compute window cutoff strictly as `const cutoff = now - this.windowMs;`.
   Step 2: Evict expired timestamps: `this.timestamps = this.timestamps.filter(ts => ts > cutoff);` (or using while loop with shift).
   Step 3: Check capacity BEFORE appending: If `this.timestamps.length < this.maxRequests`, push `now` to `this.timestamps` and return `true`. Otherwise, return `false`.
4. Import Invariant: Always import runnable classes using standard value imports (`import { TSRateLimiter } from './solution.ts';`). NEVER use `import type` for classes instantiated in tests.
5. Explicit Virtual Time in Unit Tests: JavaScript executes synchronously in sub-milliseconds. NEVER call allowRequest() without arguments expecting time or windows to elapse between consecutive lines (e.g., calling allowRequest() 4 times with no arguments within a 100ms window will ALWAYS reject the 4th call).
   ALWAYS anchor a baseline virtual time and explicitly advance it via parameters:
   const t0 = 10_000;
   assert.strictEqual(limiter.allowRequest(t0), true);
   assert.strictEqual(limiter.allowRequest(t0 + 10), true);
   assert.strictEqual(limiter.allowRequest(t0 + 20), false); // Over limit
   assert.strictEqual(limiter.allowRequest(t0 + 150), true); // Explicitly past window duration
   NEVER mix unparameterized default calls with relative integer timestamps in the same test.
6. Strict Black-Box Testing: Unit tests must ONLY call allowRequest(). NEVER assert against private fields (e.g., never read limiter.timestamps).
7. Strict Identifier Consistency: Always verify spelling of variable names in unit tests. Never introduce double letters or misspellings like llimiter.
8. State Mutation & In-Place Eviction: Ensure all pruning logic either explicitly reassigns instance variables (`this.timestamps = filtered`) or mutates the array in-place (e.g., `while (this.timestamps.length > 0 && this.timestamps[0] <= now - windowMs) { this.timestamps.shift(); }`). Never leave filtered state in unassigned local variables.
9. Relational Operator Integrity: Retain core mathematical inequalities (`< maxRequests`) without inverting comparison operators (`<` to `===` or `>` to `<`).
10. Variable Integrity: Double-check all constructor assignments to ensure variables match constructor arguments exactly (avoid typos like this.windowMs = windows).
11. The bug must be realistic, non-trivial, and reproducible.
12. The `buggy_code` MUST fail the `test_code`.
13. The `solution_code` MUST pass 100% of the assertions in `test_code`.
14. Python tests must use standard `pytest` syntax (functions starting with `test_` importing from `solution`).
15. TypeScript tests must use Node.js built-in `import { describe, it } from 'node:test'` and `import assert from 'node:assert/strict'` importing from `./solution.ts`.
16. Return ONLY valid JSON matching the exact schema requested without markdown wrapper if possible, or inside a clean ```json block.
"""


def get_curation_prompt(language: str, category: str, topic_detail: str, difficulty: str = "medium") -> str:
    if language.lower() == "python":
        test_instructions = """- `test_code` must be a valid Python test file using `pytest`.
- Import the target functions/classes from `solution` (e.g. `from solution import myFunction`).
- For time-dependent tests, use explicit baseline timestamp anchors (e.g. `t0 = 100.0`, `limiter.allow(t0)`, `limiter.allow(t0 + 1.5)`).
- Include at least 3-5 comprehensive test assertions covering edge cases, happy paths, and boundary values."""
        code_ext = "py"
    else:
        test_instructions = """- `test_code` must be a valid TypeScript test file using `import { describe, it } from 'node:test'; import assert from 'node:assert/strict'; import { ... } from './solution.ts';`.
- For time-dependent tests, declare a baseline time anchor (e.g. `const t0 = 10_000;`) and pass explicit relative increments (`t0`, `t0 + 50`, `t0 + windowMs + 1`) across ALL calls.
- Include at least 3-5 comprehensive test assertions covering edge cases, happy paths, and boundary values."""
        code_ext = "ts"

    prompt = f"""Generate a {difficulty} level {language} debugging challenge.

Category: {category}
Topic Detail: {topic_detail}

Instructions:
1. Provide a realistic `instruction` describing the expected behavior and what issue needs to be addressed.
2. Provide `buggy_code` that contains a subtle bug in its implementation.
3. Provide `solution_code` that completely and cleanly fixes the bug with proper idiomatic {language} code.
4. Provide `test_code`:
{test_instructions}
5. Provide a 1-2 sentence `explanation` of the root cause and resolution.

Output Format: JSON object with the following keys:
{{
  "category": "{category}",
  "difficulty": "{difficulty}",
  "instruction": "<clear problem & bug description>",
  "buggy_code": "<complete {code_ext} code with bug>",
  "solution_code": "<complete corrected {code_ext} code>",
  "test_code": "<complete self-contained test file>",
  "explanation": "<brief explanation of bug and fix>"
}}"""
    return prompt


def sample_random_topic(language: str) -> Tuple[str, str, str]:
    topics = PYTHON_TOPICS if language.lower() == "python" else TYPESCRIPT_TOPICS
    category, detail = random.choice(topics)
    difficulty = random.choice(["easy", "medium", "hard"])
    return category, detail, difficulty
