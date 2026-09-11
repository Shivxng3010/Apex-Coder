import sys
sys.path.insert(0, 'E:/apex-coder')
from harness.self_correct import SelfCorrectionEngine

engine = SelfCorrectionEngine()

# Test 1: Passing Python sample
sample_pass = """
<thinking>
We need a binary search function.
</thinking>
```python
def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        m = (l + r) // 2
        if arr[m] == target:
            return m
        elif arr[m] < target:
            l = m + 1
        else:
            r = m - 1
    return -1
```

```python
from solution import binary_search

def test_bs():
    assert binary_search([1, 3, 5], 3) == 1
    assert binary_search([1, 3, 5], 9) == -1
```
"""

res1 = engine.verify_output(sample_pass)
print('[Pass Test] Valid:', res1.is_valid, '| Has Tests:', res1.has_tests, '| Duration:', f'{res1.duration_seconds:.2f}s')
assert res1.is_valid is True
assert res1.has_tests is True

# Test 2: Failing State Persistence bug (missing lastTime = now update)
sample_fail = """
```typescript
export class RateLimiter {
  private lastTime = 0;
  private limit = 2;
  private count = 0;

  allow(now: number): boolean {
    if (now - this.lastTime > 1000) {
      this.count = 0;
      // Bug: Missing this.lastTime = now;
    }
    if (this.count < this.limit) {
      this.count++;
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { RateLimiter } from './solution.ts';

describe('RateLimiter', () => {
  it('resets count correctly and tracks time', () => {
    const rl = new RateLimiter();
    assert.strictEqual(rl.allow(0), true);
    assert.strictEqual(rl.allow(0), true);
    assert.strictEqual(rl.allow(0), false);
    assert.strictEqual(rl.allow(2000), true);
    assert.strictEqual(rl.allow(2000), true);
    assert.strictEqual(rl.allow(2000), false);
    assert.strictEqual(rl.allow(2500), false); // Within the 2000 window, should still be full!
  });
});
```
"""

res2 = engine.verify_output(sample_fail)
print('[Fail Test] Valid:', res2.is_valid, '| Status:', res2.status, '| Error detected:', res2.error_message is not None)
assert res2.is_valid is False
assert res2.has_tests is True

# Test 3: TypeScript with exported interface and type-only import normalization
sample_ts_interface = """
```typescript
export interface TaskConfig {
  id: string;
  timeoutMs: number;
}

export class TaskRunner {
  private active = false;

  async execute(config: TaskConfig): Promise<string> {
    if (this.active) {
      throw new Error('Task already in progress');
    }
    this.active = true;
    try {
      await new Promise(r => setTimeout(r, 5));
      return `Done: ${config.id}`;
    } finally {
      this.active = false;
    }
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { TaskConfig, TaskRunner } from './solution.ts';

describe('TaskRunner with Interface', () => {
  it('executes task conforming to TaskConfig interface', async () => {
    const runner = new TaskRunner();
    const config: TaskConfig = { id: 'task-1', timeoutMs: 100 };
    const result = await runner.execute(config);
    assert.strictEqual(result, 'Done: task-1');
  });
});
```
"""

res3 = engine.verify_output(sample_ts_interface)
print('[TS Interface Test] Valid:', res3.is_valid, '| Has Tests:', res3.has_tests, '| Duration:', f'{res3.duration_seconds:.2f}s')
if not res3.is_valid:
    print('STDOUT:\n', res3.stdout)
    print('STDERR:\n', res3.stderr)
assert res3.is_valid is True
assert res3.has_tests is True

# Test 4: TypeScript class using constructor parameter properties (sanitized for strip-only mode)
sample_ts_param_props = """
```typescript
export class TokenBucket {
  private tokens: number;
  private lastRefill: number;

  constructor(private maxCapacity: number, public refillRateMs: number = 500) {
    this.tokens = maxCapacity;
    this.lastRefill = 0;
  }

  tryConsume(now: number): boolean {
    const elapsed = Math.max(0, now - this.lastRefill);
    const addedTokens = Math.floor(elapsed / this.refillRateMs);
    if (addedTokens > 0) {
      this.tokens = Math.min(this.maxCapacity, this.tokens + addedTokens);
      this.lastRefill = now;
    }
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }

  getCapacity(): number {
    return this.maxCapacity;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { TokenBucket } from './solution.ts';

describe('TokenBucket with Parameter Properties', () => {
  it('correctly initializes fields from parameter properties and consumes tokens', () => {
    const tb = new TokenBucket(2, 100);
    assert.strictEqual(tb.getCapacity(), 2);
    assert.strictEqual(tb.tryConsume(0), true);
    assert.strictEqual(tb.tryConsume(0), true);
    assert.strictEqual(tb.tryConsume(0), false);
    assert.strictEqual(tb.tryConsume(200), true);
  });
});
```
"""

res4 = engine.verify_output(sample_ts_param_props)
print('[TS Param Props Test] Valid:', res4.is_valid, '| Has Tests:', res4.has_tests, '| Duration:', f'{res4.duration_seconds:.2f}s')
assert res4.is_valid is True
assert res4.has_tests is True

# Test 5: Ground Truth SlidingWindowRateLimiter with Jest `expect` syntax (auto-shimmed to node:assert/strict)
sample_ground_truth = """
### Production-Ready Code
```typescript
export interface RateLimiterConfig {
  maxRequests: number;
  windowDurationMs: number;
}

export class SlidingWindowRateLimiter {
  private config: RateLimiterConfig;
  private timestamps: number[];

  constructor(config: RateLimiterConfig) {
    this.config = config;
    this.timestamps = [];
  }

  public allowRequest(now: number = Date.now()): boolean {
    const windowStart = now - this.config.windowDurationMs;
    while (this.timestamps.length > 0 && this.timestamps[0] <= windowStart) {
      this.timestamps.shift();
    }

    if (this.timestamps.length < this.config.maxRequests) {
      this.timestamps.push(now);
      return true;
    }
    return false;
  }
}
```

### Unit Test Suite
```typescript
describe('SlidingWindowRateLimiter Ground Truth', () => {
  it('allows requests within window and evicts expired timestamps', () => {
    const limiter = new SlidingWindowRateLimiter({ maxRequests: 2, windowDurationMs: 1000 });
    
    // Testing Jest-style expect() assertions with auto-shim
    expect(limiter.allowRequest(100)).toBe(true);
    expect(limiter.allowRequest(200)).toBe(true);
    expect(limiter.allowRequest(300)).toBe(false); // Exceeded 2 requests
    
    // At t=1101, t=100 expires (window: [101, 1101]) -> active: [200, 1101]
    expect(limiter.allowRequest(1101)).toBe(true);
    expect(limiter.allowRequest(1102)).toBe(false); // 200 and 1101 still active
    
    // At t=1201, t=200 expires (window: [201, 1201]) -> active: [1101, 1201]
    expect(limiter.allowRequest(1201)).toBe(true);
    expect(limiter.allowRequest(1202)).toBe(false);
  });
});
```
"""

res5 = engine.verify_output(sample_ground_truth)
print('[Ground Truth + Expect Shim Test] Valid:', res5.is_valid, '| Has Tests:', res5.has_tests, '| Duration:', f'{res5.duration_seconds:.2f}s')
if not res5.is_valid:
    print('STDOUT:\n', res5.stdout)
    print('STDERR:\n', res5.stderr)
assert res5.is_valid is True
assert res5.has_tests is True

# Test 6: In-block decoupling test (single code block without headings)
sample_inblock = """
```typescript
export class Counter {
  private count = 0;
  inc() { return ++this.count; }
  get() { return this.count; }
}

describe('Counter in-block', () => {
  it('increments correctly', () => {
    const c = new Counter();
    expect(c.inc()).toBe(1);
    expect(c.get()).toBe(1);
  });
});
```
"""

res6 = engine.verify_output(sample_inblock)
print('[In-Block Decoupling Test] Valid:', res6.is_valid, '| Has Tests:', res6.has_tests, '| Duration:', f'{res6.duration_seconds:.2f}s')
assert res6.is_valid is True
assert res6.has_tests is True

# Test 7: ReferenceError Auto-Diagnostics Detection
sample_reference_error = """
```typescript
export class WindowCounter {
  private windowMs: number;
  constructor(windowSize: number) {
    // Typo bug: 'windows' instead of 'windowSize'
    this.windowMs = windows;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { WindowCounter } from './solution.ts';

describe('WindowCounter', () => {
  it('initializes correctly', () => {
    const wc = new WindowCounter(1000);
    assert.ok(wc);
  });
});
```
"""

res7 = engine.verify_output(sample_reference_error)
print('[ReferenceError Test] Valid:', res7.is_valid, '| Status:', res7.status, '| Error detected:', res7.error_message is not None)
assert res7.is_valid is False
reflection7 = engine.build_reflection_prompt('query', sample_reference_error, res7)
assert "A variable name is undefined or misspelled. Fix the variable name to match the argument list exactly." in reflection7

# Test 8: AssertionError Auto-Diagnostics Detection & Non-Inversion Guard
reflection2 = engine.build_reflection_prompt('query', sample_fail, res2)
print('\n[Generated Reflection Feedback Prompt (AssertionError)]:\n', reflection2)
assert "DO NOT invert comparison operators (< to === or > to <)" in reflection2
assert "Verify whether state mutation (this.timestamps = ...) or mock timestamp progression in the test suite is the actual root cause" in reflection2
assert "Check order of operations: Did you prune expired items before checking capacity?" in reflection2
assert "Did you check `length < maxRequests` before pushing `now`?" in reflection2

# Test 9: Sliding Window with Baseline Time Anchor (t0 = 10_000) and In-Place Eviction
sample_baseline_anchor = """
```typescript
export class SlidingWindowRateLimiter {
  private maxRequests: number;
  private windowMs: number;
  private timestamps: number[];

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.timestamps = [];
  }

  allowRequest(now: number = Date.now()): boolean {
    while (this.timestamps.length > 0 && (now - this.timestamps[0] >= this.windowMs)) {
      this.timestamps.shift();
    }
    if (this.timestamps.length < this.maxRequests) {
      this.timestamps.push(now);
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SlidingWindowRateLimiter } from './solution.ts';

describe('SlidingWindow with Baseline Time Anchor', () => {
  it('correctly tracks and evicts timestamps across monotonic time progression', () => {
    const limiter = new SlidingWindowRateLimiter(2, 1000);
    const t0 = 10_000;

    assert.strictEqual(limiter.allowRequest(t0), true);
    assert.strictEqual(limiter.allowRequest(t0 + 100), true);
    assert.strictEqual(limiter.allowRequest(t0 + 200), false); // Capacity 2 exceeded

    // At t0 + 1001, t0 (10_000) expires -> 10_100 remains -> 1 slot available
    assert.strictEqual(limiter.allowRequest(t0 + 1001), true);
    assert.strictEqual(limiter.allowRequest(t0 + 1002), false);

    // At t0 + 1101, 10_100 expires -> 10_1001 remains -> 1 slot available
    assert.strictEqual(limiter.allowRequest(t0 + 1101), true);
  });
});
```
"""

res9 = engine.verify_output(sample_baseline_anchor)
print('[Baseline Time Anchor Test] Valid:', res9.is_valid, '| Has Tests:', res9.has_tests, '| Duration:', f'{res9.duration_seconds:.2f}s')
assert res9.is_valid is True
assert res9.has_tests is True

# Test 10: State Reassignment Omission Bug (local filter without instance update)
sample_unassigned_filter = """
```typescript
export class LeakyLimiter {
  private timestamps: number[];
  private limit: number;
  private windowMs: number;

  constructor(limit: number, windowMs: number) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.timestamps = [];
  }

  allow(now: number): boolean {
    // Bug: Filter is created locally but never assigned back to this.timestamps!
    const filtered = this.timestamps.filter(ts => (now - ts) < this.windowMs);
    if (this.timestamps.length < this.limit) {
      this.timestamps.push(now);
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { LeakyLimiter } from './solution.ts';

describe('LeakyLimiter Reassignment Check', () => {
  it('fails when timestamps are not evicted due to missing reassignment', () => {
    const limiter = new LeakyLimiter(2, 500);
    const t0 = 10_000;
    assert.strictEqual(limiter.allow(t0), true);
    assert.strictEqual(limiter.allow(t0 + 10), true);
    assert.strictEqual(limiter.allow(t0 + 20), false);

    // After 600ms, both previous requests should have expired
    assert.strictEqual(limiter.allow(t0 + 600), true);
  });
});
```
"""

res10 = engine.verify_output(sample_unassigned_filter)
print('[Reassignment Omission Test] Valid:', res10.is_valid, '| Status:', res10.status, '| Error detected:', res10.error_message is not None)
assert res10.is_valid is False
reflection10 = engine.build_reflection_prompt('query', sample_unassigned_filter, res10)
assert "State Mutation & In-Place Eviction" in reflection10

# Test 11: Sliding Log Invariant with filter(ts => ts > cutoff) and check before push
sample_sliding_log_filter = """
```typescript
export class SlidingWindowLogLimiter {
  private maxRequests: number;
  private windowMs: number;
  private timestamps: number[];

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.timestamps = [];
  }

  allowRequest(now: number = Date.now()): boolean {
    const cutoff = now - this.windowMs;
    this.timestamps = this.timestamps.filter(ts => ts > cutoff);
    if (this.timestamps.length < this.maxRequests) {
      this.timestamps.push(now);
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SlidingWindowLogLimiter } from './solution.ts';

describe('SlidingWindowLogLimiter with Filter Eviction', () => {
  it('strictly adheres to algorithmic steps 1, 2, and 3', () => {
    const limiter = new SlidingWindowLogLimiter(3, 1000);
    const t0 = 20_000;

    assert.strictEqual(limiter.allowRequest(t0), true);
    assert.strictEqual(limiter.allowRequest(t0 + 100), true);
    assert.strictEqual(limiter.allowRequest(t0 + 200), true);
    assert.strictEqual(limiter.allowRequest(t0 + 300), false); // Max 3 reached

    // At t0 + 1001, t0 (20_000) is pruned because 20_000 <= 20_001 - 1000
    assert.strictEqual(limiter.allowRequest(t0 + 1001), true);
    assert.strictEqual(limiter.allowRequest(t0 + 1002), false);
  });
});
```
"""

res11 = engine.verify_output(sample_sliding_log_filter)
print('[Sliding Log Invariant Filter Test] Valid:', res11.is_valid, '| Has Tests:', res11.has_tests, '| Duration:', f'{res11.duration_seconds:.2f}s')
assert res11.is_valid is True
assert res11.has_tests is True

# Test 12: Lifecycle Hooks Normalization (beforeEach, afterEach, before, after)
sample_lifecycle_hooks = """
```typescript
export class SessionManager {
  private activeSessions: Set<string> = new Set();

  startSession(id: string) {
    this.activeSessions.add(id);
  }

  endSession(id: string) {
    this.activeSessions.delete(id);
  }

  count(): number {
    return this.activeSessions.size;
  }
}
```

```typescript
// Test code initially only imports describe and it, but uses beforeEach and afterEach
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SessionManager } from './solution.ts';

describe('SessionManager with Lifecycle Hooks', () => {
  let manager: SessionManager;

  beforeEach(() => {
    manager = new SessionManager();
  });

  afterEach(() => {
    manager.endSession('test-1');
  });

  it('adds and clears sessions with beforeEach/afterEach hooks', () => {
    assert.strictEqual(manager.count(), 0);
    manager.startSession('test-1');
    assert.strictEqual(manager.count(), 1);
  });
});
```
"""

res12 = engine.verify_output(sample_lifecycle_hooks)
print('[Lifecycle Hooks Normalization Test] Valid:', res12.is_valid, '| Has Tests:', res12.has_tests, '| Duration:', f'{res12.duration_seconds:.2f}s')
assert res12.is_valid is True
assert res12.has_tests is True

# Test 13: AST / Regex Identifier Typo Normalization (llimiter, limeder, llims -> limiter)
sample_typo_test = """
```typescript
export class SlidingWindowRateLimiter {
  private maxRequests: number;
  private windowMs: number;
  private timestamps: number[];

  constructor(maxRequests: number, windowMs: number) {
    this.maxRequests = maxRequests;
    this.windowMs = windowMs;
    this.timestamps = [];
  }

  allowRequest(now: number = Date.now()): boolean {
    const cutoff = now - this.windowMs;
    this.timestamps = this.timestamps.filter(ts => ts > cutoff);
    if (this.timestamps.length < this.maxRequests) {
      this.timestamps.push(now);
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SlidingWindowRateLimiter } from './solution.ts';

describe('SlidingWindowRateLimiter Typo Normalization', () => {
  it('automatically normalizes misspelled identifiers (llimiter, limeder, llims)', () => {
    const limiter = new SlidingWindowRateLimiter(2, 1000);
    const t0 = 10_000;

    // Notice intentional model typos: llimiter, limeder, llims
    assert.strictEqual(llimiter.allowRequest(t0), true);
    assert.strictEqual(limeder.allowRequest(t0 + 100), true);
    assert.strictEqual(llims.allowRequest(t0 + 200), false);
    assert.strictEqual(limiter.allowRequest(t0 + 1001), true);
  });
});
```
"""

res13 = engine.verify_output(sample_typo_test)
print('[Typo Normalizer Test] Valid:', res13.is_valid, '| Has Tests:', res13.has_tests, '| Duration:', f'{res13.duration_seconds:.2f}s')
assert res13.is_valid is True
assert res13.has_tests is True

# Test 14: Zero Elapsed Time Reflection Diagnostic on Unparameterized Calls
sample_unparameterized_fail = """
```typescript
export class RateLimiter {
  private count = 0;
  allowRequest(now: number = Date.now()): boolean {
    if (this.count < 2) {
      this.count++;
      return true;
    }
    return false;
  }
}
```

```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { RateLimiter } from './solution.ts';

describe('RateLimiter Synchronous Calls', () => {
  it('expects time to elapse between unparameterized synchronous calls', () => {
    const limiter = new RateLimiter();
    assert.strictEqual(limiter.allowRequest(), true);
    assert.strictEqual(limiter.allowRequest(), true);
    // 3rd call in same millisecond fails count < 2
    assert.strictEqual(limiter.allowRequest(), true);
  });
});
```
"""

res14 = engine.verify_output(sample_unparameterized_fail)
print('[Unparameterized Calls Test] Valid:', res14.is_valid, '| Status:', res14.status)
assert res14.is_valid is False
reflection14 = engine.build_reflection_prompt('query', sample_unparameterized_fail, res14)
print('\n[Generated Reflection Feedback (Zero Elapsed Time)]:\n', reflection14)
assert "Zero Elapsed Time" in reflection14
assert "Assertion failed due to zero elapsed time" in reflection14
assert "Fix the unit test by passing explicit advancing timestamps" in reflection14

# Test 15: Degenerate Repetition Loop Breaker
from harness.self_correct import break_degenerate_repetition
sample_repetitive_text = """
```typescript
export class TestClass {
  hello() { return 'world'; }
}
```
TS TS TS TS TS TS TS TS TS TS
TS TS TS TS TS TS TS TS TS TS
```typescript
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TestClass } from './solution.ts';

describe('TestClass', () => {
  it('runs', () => {
    const t = new TestClass();
    assert.strictEqual(t.hello(), 'world');
  });
});
```
"""
res15 = engine.verify_output(sample_repetitive_text)
print('[Repetition Loop Breaker Test] Valid:', res15.is_valid, '| Has Tests:', res15.has_tests, '| Duration:', f'{res15.duration_seconds:.2f}s')
assert res15.is_valid is True
assert res15.has_tests is True

# Test 16: TypeScript import type / mixed inline type specifier sanitization
sample_mixed_import_type = """
```typescript
export interface LimiterConfig {
  max: number;
}

export class TSRateLimiter {
  private count = 0;
  private max: number;
  constructor(config: LimiterConfig) {
    this.max = config.max;
  }
  allow(): boolean {
    if (this.count < this.max) {
      this.count++;
      return true;
    }
    return false;
  }
}
```

```typescript
// Test uses both `import type` on runnable class and mixed inline `type LimiterConfig`
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { type LimiterConfig, TSRateLimiter } from './solution.ts';

describe('TSRateLimiter with Mixed Import Specifiers', () => {
  it('instantiates TSRateLimiter without ReferenceError', () => {
    const config: LimiterConfig = { max: 2 };
    const limiter = new TSRateLimiter(config);
    assert.strictEqual(limiter.allow(), true);
    assert.strictEqual(limiter.allow(), true);
    assert.strictEqual(limiter.allow(), false);
  });
});
```
"""

res16 = engine.verify_output(sample_mixed_import_type)
print('[Mixed Import Type Sanitization Test] Valid:', res16.is_valid, '| Has Tests:', res16.has_tests, '| Duration:', f'{res16.duration_seconds:.2f}s')
if not res16.is_valid:
    print('STDOUT:\n', res16.stdout)
    print('STDERR:\n', res16.stderr)
assert res16.is_valid is True
assert res16.has_tests is True

print('\nAll 16 self-correction, auto-diagnostics, time advancement, import sanitization, and loop-breaker test suites PASSED 100%!')
