import os
import sys
from typing import List
from harness.schemas import DatasetSample, Language


def generate_concurrency_python_sample(index: int) -> DatasetSample:
    archetypes = [
        _py_async_semaphore_pool,
        _py_async_sliding_window_limiter,
        _py_async_queue_producer_consumer,
        _py_async_retry_exponential,
        _py_async_debounce_cancel,
        _py_async_mutex_lock,
        _py_async_batch_buffer,
        _py_async_pubsub_broadcast,
    ]
    gen = archetypes[index % len(archetypes)]
    return gen(index)


def generate_concurrency_ts_sample(index: int) -> DatasetSample:
    archetypes = [
        _ts_async_mutex_lock,
        _ts_async_promise_pool,
        _ts_async_token_bucket_clamped,
        _ts_async_retry_backoff,
        _ts_async_event_emitter,
        _ts_async_debounce_cancel,
        _ts_async_batch_queue,
        _ts_async_rw_lock,
        _ts_fair_async_rwlock,
        _ts_weakref_event_bus,
        _ts_transactional_kv_store_savepoints,
        _ts_lfu_cache_o1,
        _ts_async_countdown_latch,
        _ts_microsecond_leaky_bucket,
    ]
    gen = archetypes[index % len(archetypes)]
    return gen(index)


# ==================== PYTHON ASYNC ARCHETYPES ====================

def _py_async_semaphore_pool(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix run_concurrent_tasks to limit simultaneous async coroutine execution using asyncio.Semaphore.",
        buggy_code="""import asyncio
from typing import List, Callable, Any

async def run_concurrent_tasks(tasks: List[Callable[[], Any]], max_concurrency: int) -> List[Any]:
    # Bug: Directly creates all tasks with asyncio.gather, ignoring max_concurrency limit
    results = await asyncio.gather(*(task() for task in tasks))
    return list(results)
""",
        solution_code="""import asyncio
from typing import List, Callable, Any

async def run_concurrent_tasks(tasks: List[Callable[[], Any]], max_concurrency: int) -> List[Any]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    sem = asyncio.Semaphore(max_concurrency)

    async def sem_task(task_fn):
        async with sem:
            return await task_fn()

    results = await asyncio.gather(*(sem_task(t) for t in tasks))
    return list(results)
""",
        test_code="""import asyncio
import pytest
from solution import run_concurrent_tasks

@pytest.mark.asyncio
async def test_semaphore_pool():
    active = 0
    max_seen = 0

    async def worker(val):
        nonlocal active, max_seen
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.02)
        active -= 1
        return val * 2

    tasks = [lambda v=i: worker(v) for i in range(6)]
    results = await run_concurrent_tasks(tasks, 2)
    assert results == [0, 2, 4, 6, 8, 10]
    assert max_seen <= 2
""",
        explanation="Wrap each task inside an asyncio.Semaphore context manager to strictly constrain maximum concurrent execution."
    )


def _py_async_sliding_window_limiter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix AsyncRateLimiter so allow_request prunes expired timestamps based on time.monotonic().",
        buggy_code="""import time
from typing import List

class AsyncRateLimiter:
    def __init__(self, max_requests: int, window_sec: float):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.timestamps: List[float] = []

    def allow_request(self, current_time: float) -> bool:
        # Bug: Appends timestamps without filtering expired ones
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
""",
        solution_code="""from typing import List

class AsyncRateLimiter:
    def __init__(self, max_requests: int, window_sec: float):
        if max_requests <= 0 or window_sec <= 0:
            raise ValueError("max_requests and window_sec must be positive")
        self.max_requests = max_requests
        self.window_sec = window_sec
        self.timestamps: List[float] = []

    def allow_request(self, current_time: float) -> bool:
        cutoff = current_time - self.window_sec
        self.timestamps = [t for t in self.timestamps if t > cutoff]
        if len(self.timestamps) < self.max_requests:
            self.timestamps.append(current_time)
            return True
        return False
""",
        test_code="""from solution import AsyncRateLimiter

def test_rate_limiter():
    limiter = AsyncRateLimiter(max_requests=2, window_sec=5.0)
    assert limiter.allow_request(1.0) is True
    assert limiter.allow_request(2.0) is True
    assert limiter.allow_request(3.0) is False  # Limit reached
    assert limiter.allow_request(6.5) is True   # 1.0 has expired
    assert limiter.allow_request(7.5) is True   # 2.0 has expired
    assert limiter.allow_request(8.0) is False  # 6.5 and 7.5 active
""",
        explanation="Filter timestamps to retain only those strictly greater than current_time - window_sec."
    )


def _py_async_queue_producer_consumer(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix process_stream so consumer tasks process all queued items and exit cleanly upon receiving a None sentinel token.",
        buggy_code="""import asyncio
from typing import List

async def process_stream(items: List[int]) -> List[int]:
    queue = asyncio.Queue()
    results = []

    async def worker():
        while True:
            val = await queue.get()
            if val is None:
                break
            results.append(val * 2)

    task = asyncio.create_task(worker())
    for item in items:
        await queue.put(item)
    # Bug: Returns immediately without sentinel or waiting for worker to finish processing
    return results
""",
        solution_code="""import asyncio
from typing import List

async def process_stream(items: List[int]) -> List[int]:
    queue = asyncio.Queue()
    results = []

    async def worker():
        while True:
            val = await queue.get()
            if val is None:
                queue.task_done()
                break
            results.append(val * 2)
            queue.task_done()

    task = asyncio.create_task(worker())

    for item in items:
        await queue.put(item)
    await queue.put(None)

    await task
    return sorted(results)
""",
        test_code="""import asyncio
import pytest
from solution import process_stream

@pytest.mark.asyncio
async def test_queue_stream():
    res = await process_stream([3, 1, 4, 2])
    assert res == [2, 4, 6, 8]
    empty_res = await process_stream([])
    assert empty_res == []
""",
        explanation="Emit a None sentinel token into the queue and await the worker task so all elements are processed before returning."
    )


def _py_async_retry_exponential(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix async_retry so it propagates the exception once max_retries is reached.",
        buggy_code="""import asyncio
from typing import Callable, Any

async def async_retry(fn: Callable[[], Any], max_retries: int, base_delay: float = 0.01) -> Any:
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            attempt += 1
            # Bug: Infinite loop, misses attempt > max_retries check
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
""",
        solution_code="""import asyncio
from typing import Callable, Any

async def async_retry(fn: Callable[[], Any], max_retries: int, base_delay: float = 0.01) -> Any:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise e
            await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
""",
        test_code="""import asyncio
import pytest
from solution import async_retry

@pytest.mark.asyncio
async def test_retry_success():
    calls = 0
    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RuntimeError("Temporary error")
        return "success"

    res = await async_retry(flaky, max_retries=3, base_delay=0.005)
    assert res == "success"
    assert calls == 3

@pytest.mark.asyncio
async def test_retry_failure():
    calls = 0
    async def fail_always():
        nonlocal calls
        calls += 1
        raise ValueError("Fatal error")

    with pytest.raises(ValueError, match="Fatal error"):
        await async_retry(fail_always, max_retries=2, base_delay=0.005)
    assert calls == 3
""",
        explanation="Check if attempt > max_retries after each caught exception and re-raise the error."
    )


def _py_async_debounce_cancel(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix AsyncDebouncer so subsequent calls cancel any pending invocation timer.",
        buggy_code="""import asyncio
from typing import Callable, Any, Optional

class AsyncDebouncer:
    def __init__(self, delay: float):
        self.delay = delay
        self._task: Optional[asyncio.Task] = None

    async def call(self, fn: Callable[[], Any]):
        # Bug: Does not cancel previous running task before spawning new one
        await asyncio.sleep(self.delay)
        return await fn()
""",
        solution_code="""import asyncio
from typing import Callable, Any, Optional

class AsyncDebouncer:
    def __init__(self, delay: float):
        self.delay = delay
        self._task: Optional[asyncio.Task] = None

    async def call(self, fn: Callable[[], Any]) -> asyncio.Task:
        if self._task and not self._task.done():
            self._task.cancel()

        async def runner():
            await asyncio.sleep(self.delay)
            return await fn()

        self._task = asyncio.create_task(runner())
        return self._task
""",
        test_code="""import asyncio
import pytest
from solution import AsyncDebouncer

@pytest.mark.asyncio
async def test_debouncer():
    debouncer = AsyncDebouncer(delay=0.03)
    executed = []

    async def action(val):
        executed.append(val)
        return val

    t1 = await debouncer.call(lambda: action(1))
    t2 = await debouncer.call(lambda: action(2))
    t3 = await debouncer.call(lambda: action(3))

    try:
        await t3
    except asyncio.CancelledError:
        pass

    assert executed == [3]
""",
        explanation="Cancel prior uncompleted tasks using self._task.cancel() to ensure only the latest call executes."
    )


def _py_async_mutex_lock(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix AsyncCounter to prevent race conditions during concurrent increment operations using asyncio.Lock.",
        buggy_code="""import asyncio

class AsyncCounter:
    def __init__(self):
        self.value = 0

    async def increment(self):
        # Bug: Simulates async I/O read-modify-write without lock, leading to race condition
        val = self.value
        await asyncio.sleep(0.001)
        self.value = val + 1
""",
        solution_code="""import asyncio

class AsyncCounter:
    def __init__(self):
        self.value = 0
        self.lock = asyncio.Lock()

    async def increment(self):
        async with self.lock:
            val = self.value
            await asyncio.sleep(0.001)
            self.value = val + 1
""",
        test_code="""import asyncio
import pytest
from solution import AsyncCounter

@pytest.mark.asyncio
async def test_async_counter_lock():
    counter = AsyncCounter()
    await asyncio.gather(*(counter.increment() for _ in range(10)))
    assert counter.value == 10
""",
        explanation="Protect shared mutable state using async with self.lock context manager."
    )


def _py_async_batch_buffer(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix AsyncBatchBuffer so flush returns a shallow copy of items before clearing buffer state.",
        buggy_code="""from typing import List, Any

class AsyncBatchBuffer:
    def __init__(self, max_size: int = 4):
        self.max_size = max_size
        self.buffer: List[Any] = []

    def push(self, item: Any) -> bool:
        self.buffer.append(item)
        return len(self.buffer) >= self.max_size

    def flush(self) -> List[Any]:
        # Bug: Clears and returns empty list reference
        self.buffer.clear()
        return self.buffer
""",
        solution_code="""from typing import List, Any

class AsyncBatchBuffer:
    def __init__(self, max_size: int = 4):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.buffer: List[Any] = []

    def push(self, item: Any) -> bool:
        self.buffer.append(item)
        return len(self.buffer) >= self.max_size

    def flush(self) -> List[Any]:
        snapshot = list(self.buffer)
        self.buffer.clear()
        return snapshot
""",
        test_code="""from solution import AsyncBatchBuffer

def test_batch_buffer():
    buf = AsyncBatchBuffer(max_size=3)
    assert buf.push(10) is False
    assert buf.push(20) is False
    assert buf.push(30) is True
    batch = buf.flush()
    assert batch == [10, 20, 30]
    assert buf.flush() == []
""",
        explanation="Copy the list before calling self.buffer.clear() to avoid returning an empty list."
    )


def _py_async_pubsub_broadcast(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_py_{idx:04d}",
        language=Language.PYTHON,
        category="asyncio",
        difficulty="medium",
        instruction="Fix AsyncBroadcastChannel so unsubscribe removes only the specific subscriber queue.",
        buggy_code="""import asyncio
from typing import Set

class AsyncBroadcastChannel:
    def __init__(self):
        self.subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        # Bug: Clears all subscribers
        self.subscribers.clear()

    async def publish(self, message: str):
        for q in self.subscribers:
            await q.put(message)
""",
        solution_code="""import asyncio
from typing import Set

class AsyncBroadcastChannel:
    def __init__(self):
        self.subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self.subscribers.discard(q)

    async def publish(self, message: str):
        for q in list(self.subscribers):
            await q.put(message)
""",
        test_code="""import asyncio
import pytest
from solution import AsyncBroadcastChannel

@pytest.mark.asyncio
async def test_broadcast_channel():
    chan = AsyncBroadcastChannel()
    s1 = chan.subscribe()
    s2 = chan.subscribe()

    await chan.publish("hello")
    assert await s1.get() == "hello"
    assert await s2.get() == "hello"

    chan.unsubscribe(s1)
    await chan.publish("world")
    assert await s2.get() == "world"
    assert s1.empty() is True
""",
        explanation="Use self.subscribers.discard(q) to remove only the targeted queue without disrupting other subscribers."
    )


# ==================== TYPESCRIPT ASYNC ARCHETYPES ====================

def _ts_async_mutex_lock(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="medium",
        instruction="Fix AsyncMutex so acquired locks are released sequentially to waiting callers.",
        buggy_code="""export class AsyncMutex {
  private locked = false;
  private queue: (() => void)[] = [];

  async acquire(): Promise<() => void> {
    if (this.locked) {
      await new Promise<void>(resolve => this.queue.push(resolve));
    }
    this.locked = true;
    // Bug: Release function does not pass lock to the next queued caller
    return () => {
      this.locked = false;
    };
  }
}
""",
        solution_code="""export class AsyncMutex {
  private locked = false;
  private queue: (() => void)[] = [];

  async acquire(): Promise<() => void> {
    if (this.locked) {
      await new Promise<void>(resolve => this.queue.push(resolve));
    }
    this.locked = true;

    return () => {
      if (this.queue.length > 0) {
        const next = this.queue.shift()!;
        next();
      } else {
        this.locked = false;
      }
    };
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { AsyncMutex } from './solution.ts';

describe('AsyncMutex', () => {
  it('enforces serial execution of critical sections', async () => {
    const mutex = new AsyncMutex();
    const order: number[] = [];

    async function task(id: number, delayMs: number) {
      const release = await mutex.acquire();
      order.push(id);
      await new Promise(r => setTimeout(r, delayMs));
      release();
    }

    await Promise.all([
      task(1, 20),
      task(2, 10),
      task(3, 5),
    ]);

    assert.deepStrictEqual(order, [1, 2, 3]);
  });
});
""",
        explanation="When releasing the mutex, pop the next resolve function from queue and invoke it, preserving lock ownership."
    )


def _ts_async_promise_pool(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="hard",
        instruction="Fix promisePool so tasks run concurrently up to maxConcurrency limit, returning all results.",
        buggy_code="""export async function promisePool<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrency: number
): Promise<T[]> {
  // Bug: Ignores maxConcurrency and runs all promises concurrently
  return Promise.all(tasks.map(t => t()));
}
""",
        solution_code="""export async function promisePool<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrency: number
): Promise<T[]> {
  if (maxConcurrency < 1) throw new Error("maxConcurrency must be at least 1");
  const results: T[] = new Array(tasks.length);
  let cursor = 0;

  async function worker() {
    while (cursor < tasks.length) {
      const idx = cursor++;
      results[idx] = await tasks[idx]();
    }
  }

  const pool = Array.from(
    { length: Math.min(maxConcurrency, tasks.length) },
    () => worker()
  );
  await Promise.all(pool);
  return results;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { promisePool } from './solution.ts';

describe('promisePool', () => {
  it('limits active concurrent tasks to maxConcurrency', async () => {
    let active = 0;
    let maxSeen = 0;

    const makeTask = (val: number) => async () => {
      active++;
      maxSeen = Math.max(maxSeen, active);
      await new Promise(r => setTimeout(r, 15));
      active--;
      return val * 10;
    };

    const tasks = [1, 2, 3, 4, 5].map(makeTask);
    const res = await promisePool(tasks, 2);
    assert.deepStrictEqual(res, [10, 20, 30, 40, 50]);
    assert.strictEqual(maxSeen <= 2, true);
  });
});
""",
        explanation="Dispatch a fixed pool of workers that atomically take task indices from a shared cursor."
    )


def _ts_async_token_bucket_clamped(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="algorithms",
        difficulty="medium",
        instruction="Fix TokenBucket so token accumulation is clamped at maxCapacity during refill.",
        buggy_code="""export class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  private maxCapacity: number;
  private refillRatePerSec: number;

  constructor(maxCapacity: number, refillRatePerSec: number, startTime: number) {
    this.maxCapacity = maxCapacity;
    this.refillRatePerSec = refillRatePerSec;
    this.tokens = maxCapacity;
    this.lastRefill = startTime;
  }

  tryConsume(now: number): boolean {
    const elapsedSec = (now - this.lastRefill) / 1000;
    // Bug: Uncapped token refill allows tokens to grow infinitely
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
  private maxCapacity: number;
  private refillRatePerSec: number;

  constructor(maxCapacity: number, refillRatePerSec: number, startTime: number) {
    if (maxCapacity <= 0 || refillRatePerSec <= 0) {
      throw new Error("Parameters must be positive");
    }
    this.maxCapacity = maxCapacity;
    this.refillRatePerSec = refillRatePerSec;
    this.tokens = maxCapacity;
    this.lastRefill = startTime;
  }

  tryConsume(now: number): boolean {
    const elapsedSec = Math.max(0, (now - this.lastRefill) / 1000);
    this.tokens = Math.min(this.maxCapacity, this.tokens + elapsedSec * this.refillRatePerSec);
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
  it('clamps tokens to max capacity even after long idle time', () => {
    const bucket = new TokenBucket(3, 5, 0);
    assert.strictEqual(bucket.tryConsume(0), true);
    assert.strictEqual(bucket.tryConsume(0), true);
    assert.strictEqual(bucket.tryConsume(0), true);
    assert.strictEqual(bucket.tryConsume(0), false);

    // Advance time by 100 seconds
    assert.strictEqual(bucket.tryConsume(100000), true);
    assert.strictEqual(bucket.tryConsume(100000), true);
    assert.strictEqual(bucket.tryConsume(100000), true);
    assert.strictEqual(bucket.tryConsume(100000), false); // Max capacity is 3
  });
});
""",
        explanation="Clamp tokens using Math.min(this.maxCapacity, ...) during the refill computation."
    )


def _ts_async_retry_backoff(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="medium",
        instruction="Fix retryWithBackoff to re-throw error once maxRetries is exceeded.",
        buggy_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  baseDelayMs: number = 5
): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      // Bug: Missing check if attempt > maxRetries before continuing
      await new Promise(r => setTimeout(r, baseDelayMs * Math.pow(2, attempt - 1)));
    }
  }
}
""",
        solution_code="""export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number,
  baseDelayMs: number = 5
): Promise<T> {
  if (maxRetries < 0) throw new Error("maxRetries must be non-negative");
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (err) {
      attempt++;
      if (attempt > maxRetries) {
        throw err;
      }
      await new Promise(r => setTimeout(r, baseDelayMs * Math.pow(2, attempt - 1)));
    }
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { retryWithBackoff } from './solution.ts';

describe('retryWithBackoff', () => {
  it('succeeds after temporary failures', async () => {
    let calls = 0;
    const res = await retryWithBackoff(async () => {
      calls++;
      if (calls < 2) throw new Error('Temp');
      return 'ok';
    }, 3, 2);
    assert.strictEqual(res, 'ok');
    assert.strictEqual(calls, 2);
  });

  it('throws error after exceeding max retries', async () => {
    let calls = 0;
    await assert.rejects(
      async () => {
        await retryWithBackoff(async () => {
          calls++;
          throw new Error('Persistent failure');
        }, 2, 2);
      },
      /Persistent failure/
    );
    assert.strictEqual(calls, 3);
  });
});
""",
        explanation="Check attempt > maxRetries in the catch block and throw the caught error."
    )


def _ts_async_event_emitter(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="event-emitter",
        difficulty="medium",
        instruction="Fix EventEmitter so off removes only the target callback and once auto-removes after execution.",
        buggy_code="""export class EventEmitter {
  private listeners: Record<string, Function[]> = {};

  on(event: string, fn: Function): void {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(fn);
  }

  off(event: string, fn: Function): void {
    // Bug: Clears all listeners for this event
    this.listeners[event] = [];
  }

  emit(event: string, ...args: any[]): void {
    const list = this.listeners[event] || [];
    list.forEach(fn => fn(...args));
  }
}
""",
        solution_code="""export class EventEmitter {
  private listeners: Record<string, Function[]> = {};

  on(event: string, fn: Function): void {
    if (!this.listeners[event]) this.listeners[event] = [];
    this.listeners[event].push(fn);
  }

  off(event: string, fn: Function): void {
    if (!this.listeners[event]) return;
    this.listeners[event] = this.listeners[event].filter(cb => cb !== fn);
  }

  once(event: string, fn: Function): void {
    const wrapper = (...args: any[]) => {
      this.off(event, wrapper);
      fn(...args);
    };
    this.on(event, wrapper);
  }

  emit(event: string, ...args: any[]): void {
    const list = [...(this.listeners[event] || [])];
    list.forEach(fn => fn(...args));
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { EventEmitter } from './solution.ts';

describe('EventEmitter', () => {
  it('handles on, off, and once correctly', () => {
    const ee = new EventEmitter();
    let count1 = 0, count2 = 0;
    const fn1 = () => count1++;
    const fn2 = () => count2++;

    ee.on('data', fn1);
    ee.once('data', fn2);

    ee.emit('data');
    assert.strictEqual(count1, 1);
    assert.strictEqual(count2, 1);

    ee.emit('data');
    assert.strictEqual(count1, 2);
    assert.strictEqual(count2, 1); // once was unregistered

    ee.off('data', fn1);
    ee.emit('data');
    assert.strictEqual(count1, 2);
  });
});
""",
        explanation="Filter listener array in off() to retain non-matching functions, and wrap callbacks in once()."
    )


def _ts_async_debounce_cancel(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="functional",
        difficulty="medium",
        instruction="Fix debounce so it forwards arguments correctly and supports explicit cancel().",
        buggy_code="""export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): { (...args: Parameters<T>): void; cancel: () => void } {
  let timer: any = null;

  const debounced = (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    // Bug: Calls fn without args
    timer = setTimeout(() => { fn(); }, delayMs);
  };

  debounced.cancel = () => {
    if (timer) clearTimeout(timer);
  };

  return debounced;
}
""",
        solution_code="""export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delayMs: number
): { (...args: Parameters<T>): void; cancel: () => void } {
  let timer: any = null;

  const debounced = (...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      fn(...args);
    }, delayMs);
  };

  debounced.cancel = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  return debounced;
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { debounce } from './solution.ts';

describe('debounce with cancel', () => {
  it('executes with latest arguments and respects cancel', async () => {
    let result = '';
    const fn = debounce((msg: string) => { result = msg; }, 15);

    fn('first');
    fn('second');
    await new Promise(r => setTimeout(r, 25));
    assert.strictEqual(result, 'second');

    fn('third');
    fn.cancel();
    await new Promise(r => setTimeout(r, 25));
    assert.strictEqual(result, 'second'); // 'third' was cancelled
  });
});
""",
        explanation="Pass captured ...args into fn() and clear timer in cancel()."
    )


def _ts_async_batch_queue(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="medium",
        instruction="Fix AsyncBatchQueue so flush returns a new array slice before resetting internal buffer.",
        buggy_code="""export class AsyncBatchQueue<T> {
  private buffer: T[] = [];
  private batchSize: number;

  constructor(batchSize: number) {
    this.batchSize = batchSize;
  }

  add(item: T): boolean {
    this.buffer.push(item);
    return this.buffer.length >= this.batchSize;
  }

  flush(): T[] {
    // Bug: Mutates buffer by clearing it and returns empty reference
    this.buffer.length = 0;
    return this.buffer;
  }
}
""",
        solution_code="""export class AsyncBatchQueue<T> {
  private buffer: T[] = [];
  private batchSize: number;

  constructor(batchSize: number) {
    if (batchSize <= 0) throw new Error("batchSize must be positive");
    this.batchSize = batchSize;
  }

  add(item: T): boolean {
    this.buffer.push(item);
    return this.buffer.length >= this.batchSize;
  }

  flush(): T[] {
    const items = [...this.buffer];
    this.buffer = [];
    return items;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { AsyncBatchQueue } from './solution.ts';

describe('AsyncBatchQueue', () => {
  it('batches items and flushes non-empty snapshots', () => {
    const q = new AsyncBatchQueue<string>(2);
    assert.strictEqual(q.add('a'), false);
    assert.strictEqual(q.add('b'), true);
    const batch = q.flush();
    assert.deepStrictEqual(batch, ['a', 'b']);
    assert.deepStrictEqual(q.flush(), []);
  });
});
""",
        explanation="Copy buffer elements with [...this.buffer] before reinitializing this.buffer = []."
    )


def _ts_async_rw_lock(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="hard",
        instruction="Fix AsyncReadWriteLock so writers wait until all active readers have released their read lock.",
        buggy_code="""export class AsyncReadWriteLock {
  private writing = false;
  private readers = 0;

  async acquireRead(): Promise<() => void> {
    while (this.writing) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.readers++;
    return () => {
      this.readers = Math.max(0, this.readers - 1);
    };
  }

  async acquireWrite(): Promise<() => void> {
    // Bug: Writer only checks if another writer is active, ignoring active readers!
    while (this.writing) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.writing = true;
    return () => {
      this.writing = false;
    };
  }
}
""",
        solution_code="""export class AsyncReadWriteLock {
  private writing = false;
  private readers = 0;

  async acquireRead(): Promise<() => void> {
    while (this.writing) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.readers++;
    return () => {
      this.readers = Math.max(0, this.readers - 1);
    };
  }

  async acquireWrite(): Promise<() => void> {
    while (this.writing || this.readers > 0) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.writing = true;
    return () => {
      this.writing = false;
    };
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert';
import { AsyncReadWriteLock } from './solution.ts';

describe('AsyncReadWriteLock', () => {
  it('blocks writer when readers are active', async () => {
    const rw = new AsyncReadWriteLock();
    let writeExecuted = false;

    const r1 = await rw.acquireRead();
    
    // Attempt write while read lock is held
    const writePromise = (async () => {
      const releaseWrite = await rw.acquireWrite();
      writeExecuted = true;
      releaseWrite();
    })();

    await new Promise(r => setTimeout(r, 20));
    assert.strictEqual(writeExecuted, false); // Must still be waiting for reader

    r1(); // Release reader
    await writePromise;
    assert.strictEqual(writeExecuted, true);
  });
});
""",
        explanation="acquireWrite must wait while either another writer is active or this.readers > 0."
    )


def _ts_fair_async_rwlock(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="hard",
        instruction="Fix FairAsyncReadWriteLock to prevent writer starvation by queuing incoming readers behind waiting writers.",
        buggy_code="""export class FairAsyncReadWriteLock {
  private activeReaders = 0;
  private isWriting = false;
  private waitingWriters = 0;

  async acquireRead(): Promise<() => void> {
    // Bug: Ignores waitingWriters, allowing incoming readers to starve waiting writers
    while (this.isWriting) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.activeReaders++;
    return () => {
      this.activeReaders = Math.max(0, this.activeReaders - 1);
    };
  }

  async acquireWrite(): Promise<() => void> {
    this.waitingWriters++;
    try {
      while (this.isWriting || this.activeReaders > 0) {
        await new Promise(r => setTimeout(r, 5));
      }
      this.isWriting = true;
      return () => {
        this.isWriting = false;
      };
    } finally {
      this.waitingWriters = Math.max(0, this.waitingWriters - 1);
    }
  }
}
""",
        solution_code="""export class FairAsyncReadWriteLock {
  private activeReaders = 0;
  private isWriting = false;
  private waitingWriters = 0;

  async acquireRead(): Promise<() => void> {
    while (this.isWriting || this.waitingWriters > 0) {
      await new Promise(r => setTimeout(r, 5));
    }
    this.activeReaders++;
    return () => {
      this.activeReaders = Math.max(0, this.activeReaders - 1);
    };
  }

  async acquireWrite(): Promise<() => void> {
    this.waitingWriters++;
    try {
      while (this.isWriting || this.activeReaders > 0) {
        await new Promise(r => setTimeout(r, 5));
      }
      this.isWriting = true;
      return () => {
        this.isWriting = false;
      };
    } finally {
      this.waitingWriters = Math.max(0, this.waitingWriters - 1);
    }
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { FairAsyncReadWriteLock } from './solution.ts';

describe('FairAsyncReadWriteLock', () => {
  it('prevents writer starvation when incoming readers arrive', async () => {
    const lock = new FairAsyncReadWriteLock();
    const order: string[] = [];

    const relR1 = await lock.acquireRead();
    order.push('r1_acquired');

    const writePromise = (async () => {
      const relW = await lock.acquireWrite();
      order.push('w1_acquired');
      await new Promise(r => setTimeout(r, 20));
      order.push('w1_released');
      relW();
    })();

    await new Promise(r => setTimeout(r, 10));

    const read2Promise = (async () => {
      const relR2 = await lock.acquireRead();
      order.push('r2_acquired');
      relR2();
    })();

    await new Promise(r => setTimeout(r, 20));
    order.push('r1_released');
    relR1();

    await Promise.all([writePromise, read2Promise]);

    assert.strictEqual(order.indexOf('w1_acquired') < order.indexOf('r2_acquired'), true);
    assert.deepStrictEqual(order, [
      'r1_acquired',
      'r1_released',
      'w1_acquired',
      'w1_released',
      'r2_acquired'
    ]);
  });
});
""",
        explanation="acquireRead must wait while either another writer is active or this.waitingWriters > 0 to prevent writer starvation."
    )


def _ts_weakref_event_bus(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="memory-management",
        difficulty="hard",
        instruction="Fix MemorySafeEventBus so dead or dereferenced listener targets wrapped in WeakRef are automatically pruned during emit to prevent memory leaks.",
        buggy_code="""export class MemorySafeEventBus {
  private listeners = new Map<string, Array<{ ref: WeakRef<object>; fnName: string }>>();

  on(event: string, target: object, fnName: string): void {
    const list = this.listeners.get(event) || [];
    list.push({ ref: new WeakRef(target), fnName });
    this.listeners.set(event, list);
  }

  emit(event: string, ...args: unknown[]): number {
    const list = this.listeners.get(event) || [];
    let called = 0;
    // Bug: Never cleans up stale listeners when target is garbage collected
    for (const item of list) {
      const obj = item.ref.deref() as Record<string, any> | undefined;
      if (obj && typeof obj[item.fnName] === 'function') {
        obj[item.fnName](...args);
        called++;
      }
    }
    return called;
  }

  getListenerCount(event: string): number {
    return (this.listeners.get(event) || []).length;
  }
}
""",
        solution_code="""export class MemorySafeEventBus {
  private listeners = new Map<string, Array<{ ref: WeakRef<object>; fnName: string }>>();

  on(event: string, target: object, fnName: string): void {
    const list = this.listeners.get(event) || [];
    list.push({ ref: new WeakRef(target), fnName });
    this.listeners.set(event, list);
  }

  emit(event: string, ...args: unknown[]): number {
    const list = this.listeners.get(event) || [];
    const active: Array<{ ref: WeakRef<object>; fnName: string }> = [];
    let called = 0;
    for (const item of list) {
      const obj = item.ref.deref() as Record<string, any> | undefined;
      if (obj) {
        active.push(item);
        if (typeof obj[item.fnName] === 'function') {
          obj[item.fnName](...args);
          called++;
        }
      }
    }
    this.listeners.set(event, active);
    return called;
  }

  getListenerCount(event: string): number {
    return (this.listeners.get(event) || []).length;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MemorySafeEventBus } from './solution.ts';

describe('MemorySafeEventBus', () => {
  it('dispatches to active targets and prunes dead WeakRef targets during emit', () => {
    const bus = new MemorySafeEventBus();
    let handler1Called = 0;

    const aliveTarget = {
      onData: (_payload: string) => {
        handler1Called++;
      }
    };

    bus.on('data', aliveTarget, 'onData');
    assert.strictEqual(bus.getListenerCount('data'), 1);

    const count1 = bus.emit('data', 'test payload');
    assert.strictEqual(count1, 1);
    assert.strictEqual(handler1Called, 1);

    // Simulate collected weakref object
    const dummyRef = {
      ref: { deref: () => undefined } as unknown as WeakRef<object>,
      fnName: 'fake'
    };
    (bus as any).listeners.get('data').push(dummyRef);
    assert.strictEqual(bus.getListenerCount('data'), 2);

    const count2 = bus.emit('data', 'second payload');
    assert.strictEqual(count2, 1);
    assert.strictEqual(handler1Called, 2);
    assert.strictEqual(bus.getListenerCount('data'), 1);
  });
});
""",
        explanation="Filter and reassign active listener array during emit to discard dereferenced WeakRef targets."
    )


def _ts_transactional_kv_store_savepoints(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="hard",
        instruction="Fix TransactionalKVStore so rollback to a named savepoint correctly unwinds changes and restores state at that checkpoint without affecting earlier savepoints.",
        buggy_code="""export class TransactionalKVStore {
  private data = new Map<string, unknown>();
  private savepoints: Array<{ name: string; snapshot: Map<string, unknown> }> = [];

  get(key: string): unknown | undefined {
    return this.data.get(key);
  }

  set(key: string, value: unknown): void {
    this.data.set(key, value);
  }

  savepoint(name: string): void {
    this.savepoints.push({ name, snapshot: new Map(this.data) });
  }

  rollback(name: string): boolean {
    // Bug: Clears entire savepoints array and corrupts stack
    const idx = this.savepoints.findIndex(sp => sp.name === name);
    if (idx === -1) return false;
    this.data = new Map(this.savepoints[idx].snapshot);
    this.savepoints = [];
    return true;
  }

  commit(): void {
    this.savepoints = [];
  }
}
""",
        solution_code="""export class TransactionalKVStore {
  private data = new Map<string, unknown>();
  private savepoints: Array<{ name: string; snapshot: Map<string, unknown> }> = [];

  get(key: string): unknown | undefined {
    return this.data.get(key);
  }

  set(key: string, value: unknown): void {
    this.data.set(key, value);
  }

  savepoint(name: string): void {
    this.savepoints.push({ name, snapshot: new Map(this.data) });
  }

  rollback(name: string): boolean {
    const idx = this.savepoints.findIndex(sp => sp.name === name);
    if (idx === -1) return false;
    this.data = new Map(this.savepoints[idx].snapshot);
    this.savepoints = this.savepoints.slice(0, idx + 1);
    return true;
  }

  commit(): void {
    this.savepoints = [];
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TransactionalKVStore } from './solution.ts';

describe('TransactionalKVStore', () => {
  it('supports nested savepoints and selective atomic rollback', () => {
    const store = new TransactionalKVStore();
    store.set('k1', 'v1');
    store.savepoint('sp1');

    store.set('k1', 'v2');
    store.set('k2', 'hello');
    store.savepoint('sp2');

    store.set('k1', 'v3');
    store.set('k2', 'world');
    assert.strictEqual(store.get('k1'), 'v3');
    assert.strictEqual(store.get('k2'), 'world');

    // Rollback to sp2
    const okSp2 = store.rollback('sp2');
    assert.strictEqual(okSp2, true);
    assert.strictEqual(store.get('k1'), 'v2');
    assert.strictEqual(store.get('k2'), 'hello');

    // Rollback to sp1
    const okSp1 = store.rollback('sp1');
    assert.strictEqual(okSp1, true);
    assert.strictEqual(store.get('k1'), 'v1');
    assert.strictEqual(store.get('k2'), undefined);

    store.commit();
    assert.strictEqual(store.get('k1'), 'v1');
  });
});
""",
        explanation="Slice savepoint array to keep preceding checkpoints active when rolling back to a target savepoint."
    )


def _ts_lfu_cache_o1(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="data-structures",
        difficulty="hard",
        instruction="Fix LFUCache to maintain O(1) time complexity and evict the least frequently and least recently used key when reaching capacity.",
        buggy_code="""interface LFUNode {
  key: string;
  val: unknown;
  freq: number;
}

export class LFUCache {
  private capacity: number;
  private cache = new Map<string, LFUNode>();
  private minFreq = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: string): unknown | null {
    const node = this.cache.get(key);
    if (!node) return null;
    node.freq++;
    return node.val;
  }

  put(key: string, val: unknown): void {
    if (this.capacity <= 0) return;
    if (this.cache.has(key)) {
      const node = this.cache.get(key)!;
      node.val = val;
      node.freq++;
      return;
    }
    // Bug: Linear/arbitrary eviction deleting first key instead of true minFreq LRU
    if (this.cache.size >= this.capacity) {
      const firstKey = this.cache.keys().next().value;
      if (firstKey) this.cache.delete(firstKey);
    }
    this.cache.set(key, { key, val, freq: 1 });
    this.minFreq = 1;
  }
}
""",
        solution_code="""interface LFUNode {
  key: string;
  val: unknown;
  freq: number;
  prev?: LFUNode;
  next?: LFUNode;
}

class DoublyLinkedList {
  head: LFUNode;
  tail: LFUNode;
  size = 0;

  constructor() {
    this.head = { key: '', val: null, freq: 0 };
    this.tail = { key: '', val: null, freq: 0 };
    this.head.next = this.tail;
    this.tail.prev = this.head;
  }

  addNode(node: LFUNode): void {
    node.next = this.head.next;
    node.prev = this.head;
    this.head.next!.prev = node;
    this.head.next = node;
    this.size++;
  }

  removeNode(node: LFUNode): void {
    if (!node.prev || !node.next) return;
    node.prev.next = node.next;
    node.next.prev = node.prev;
    this.size--;
  }

  removeTail(): LFUNode | null {
    if (this.size === 0) return null;
    const node = this.tail.prev!;
    this.removeNode(node);
    return node;
  }
}

export class LFUCache {
  private capacity: number;
  private keyMap = new Map<string, LFUNode>();
  private freqMap = new Map<number, DoublyLinkedList>();
  private minFreq = 0;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  private updateFreq(node: LFUNode): void {
    const curList = this.freqMap.get(node.freq)!;
    curList.removeNode(node);
    if (node.freq === this.minFreq && curList.size === 0) {
      this.minFreq++;
    }
    node.freq++;
    if (!this.freqMap.has(node.freq)) {
      this.freqMap.set(node.freq, new DoublyLinkedList());
    }
    this.freqMap.get(node.freq)!.addNode(node);
  }

  get(key: string): unknown | null {
    const node = this.keyMap.get(key);
    if (!node) return null;
    this.updateFreq(node);
    return node.val;
  }

  put(key: string, val: unknown): void {
    if (this.capacity <= 0) return;
    if (this.keyMap.has(key)) {
      const node = this.keyMap.get(key)!;
      node.val = val;
      this.updateFreq(node);
      return;
    }
    if (this.keyMap.size >= this.capacity) {
      const minList = this.freqMap.get(this.minFreq)!;
      const evicted = minList.removeTail();
      if (evicted) {
        this.keyMap.delete(evicted.key);
      }
    }
    const newNode: LFUNode = { key, val, freq: 1 };
    this.keyMap.set(key, newNode);
    if (!this.freqMap.has(1)) {
      this.freqMap.set(1, new DoublyLinkedList());
    }
    this.freqMap.get(1)!.addNode(newNode);
    this.minFreq = 1;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { LFUCache } from './solution.ts';

describe('LFUCache', () => {
  it('correctly tracks min frequency and evicts LRU among least frequent items', () => {
    const cache = new LFUCache(2);
    cache.put('a', 1);
    cache.put('b', 2);
    assert.strictEqual(cache.get('a'), 1); // 'a' freq becomes 2

    cache.put('c', 3); // capacity exceeded, evicts 'b' (freq 1)
    assert.strictEqual(cache.get('b'), null);
    assert.strictEqual(cache.get('a'), 1);
    assert.strictEqual(cache.get('c'), 3);

    cache.put('d', 4); // 'a' freq is 3, 'c' freq is 2, evicts 'c'
    assert.strictEqual(cache.get('c'), null);
    assert.strictEqual(cache.get('d'), 4);
    assert.strictEqual(cache.get('a'), 1);
  });
});
""",
        explanation="Maintain key map and frequency-to-doubly-linked-list map to achieve true O(1) LFU operations."
    )


def _ts_async_countdown_latch(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="async-await",
        difficulty="hard",
        instruction="Fix AsyncCountDownLatch so wait handles timeout, AbortSignal cancellation, and unblocks awaiting coroutines once count reaches 0.",
        buggy_code="""export class AsyncCountDownLatch {
  private count: number;
  private waiters: Array<() => void> = [];

  constructor(count: number) {
    if (count < 0) throw new Error('Count cannot be negative');
    this.count = count;
  }

  getCount(): number {
    return this.count;
  }

  countDown(): void {
    if (this.count > 0) {
      this.count--;
      if (this.count === 0) {
        for (const resolve of this.waiters) resolve();
        this.waiters = [];
      }
    }
  }

  async wait(timeoutMs?: number, signal?: AbortSignal): Promise<boolean> {
    // Bug: Ignores signal and timeout parameters, waiting indefinitely
    if (this.count === 0) return true;
    return new Promise<boolean>((resolve) => {
      this.waiters.push(() => resolve(true));
    });
  }
}
""",
        solution_code="""export class AsyncCountDownLatch {
  private count: number;
  private waiters: Array<() => void> = [];

  constructor(count: number) {
    if (count < 0) throw new Error('Count cannot be negative');
    this.count = count;
  }

  getCount(): number {
    return this.count;
  }

  countDown(): void {
    if (this.count > 0) {
      this.count--;
      if (this.count === 0) {
        const toResolve = [...this.waiters];
        this.waiters = [];
        for (const resolve of toResolve) {
          resolve();
        }
      }
    }
  }

  async wait(timeoutMs?: number, signal?: AbortSignal): Promise<boolean> {
    if (this.count === 0) return true;
    if (signal?.aborted) return false;

    return new Promise<boolean>((resolve) => {
      let timer: NodeJS.Timeout | null = null;
      let onAbort: (() => void) | null = null;

      const cleanup = () => {
        if (timer) clearTimeout(timer);
        if (signal && onAbort) signal.removeEventListener('abort', onAbort);
      };

      const resolveSuccess = () => {
        cleanup();
        resolve(true);
      };

      this.waiters.push(resolveSuccess);

      if (timeoutMs !== undefined && timeoutMs > 0) {
        timer = setTimeout(() => {
          const idx = this.waiters.indexOf(resolveSuccess);
          if (idx !== -1) this.waiters.splice(idx, 1);
          cleanup();
          resolve(false);
        }, timeoutMs);
      }

      if (signal) {
        onAbort = () => {
          const idx = this.waiters.indexOf(resolveSuccess);
          if (idx !== -1) this.waiters.splice(idx, 1);
          cleanup();
          resolve(false);
        };
        signal.addEventListener('abort', onAbort, { once: true });
      }
    });
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { AsyncCountDownLatch } from './solution.ts';

describe('AsyncCountDownLatch', () => {
  it('unblocks when countdown reaches zero', async () => {
    const latch = new AsyncCountDownLatch(3);
    let done = false;

    const waitPromise = (async () => {
      const ok = await latch.wait(500);
      done = ok;
    })();

    latch.countDown();
    latch.countDown();
    assert.strictEqual(done, false);
    assert.strictEqual(latch.getCount(), 1);

    latch.countDown();
    await waitPromise;
    assert.strictEqual(done, true);
    assert.strictEqual(latch.getCount(), 0);
  });

  it('handles timeout and abort signals cleanly', async () => {
    const latch = new AsyncCountDownLatch(5);
    const controller = new AbortController();

    const timeoutRes = await latch.wait(20);
    assert.strictEqual(timeoutRes, false);

    const abortPromise = latch.wait(500, controller.signal);
    controller.abort();
    const abortRes = await abortPromise;
    assert.strictEqual(abortRes, false);
  });
});
""",
        explanation="Clean up listeners and timers properly in wait method on resolution, timeout, or AbortSignal."
    )


def _ts_microsecond_leaky_bucket(idx: int) -> DatasetSample:
    return DatasetSample(
        sample_id=f"concur_ts_{idx:04d}",
        language=Language.TYPESCRIPT,
        category="rate-limiting",
        difficulty="hard",
        instruction="Fix MicrosecondLeakyBucket so elapsed virtual time correctly leaks water at leakRatePerMs and clamps current volume without dropping below zero or exceeding burst capacity.",
        buggy_code="""export class MicrosecondLeakyBucket {
  private capacity: number;
  private leakRatePerMs: number;
  private currentWater = 0;
  private lastLeakTime = 0;

  constructor(capacity: number, leakRatePerMs: number, initialTimeMs = 0) {
    this.capacity = capacity;
    this.leakRatePerMs = leakRatePerMs;
    this.lastLeakTime = initialTimeMs;
  }

  addPacket(weight: number, nowMs: number): boolean {
    // Bug: Doesn't clamp water level at 0, allowing negative water and infinite burst allowance
    const elapsed = nowMs - this.lastLeakTime;
    const leaked = elapsed * this.leakRatePerMs;
    this.currentWater = this.currentWater - leaked;
    this.lastLeakTime = nowMs;

    if (this.currentWater + weight <= this.capacity) {
      this.currentWater += weight;
      return true;
    }
    return false;
  }

  getWaterLevel(): number {
    return this.currentWater;
  }
}
""",
        solution_code="""export class MicrosecondLeakyBucket {
  private capacity: number;
  private leakRatePerMs: number;
  private currentWater = 0;
  private lastLeakTime = 0;

  constructor(capacity: number, leakRatePerMs: number, initialTimeMs = 0) {
    this.capacity = capacity;
    this.leakRatePerMs = leakRatePerMs;
    this.lastLeakTime = initialTimeMs;
  }

  addPacket(weight: number, nowMs: number): boolean {
    const elapsed = Math.max(0, nowMs - this.lastLeakTime);
    const leaked = elapsed * this.leakRatePerMs;
    this.currentWater = Math.max(0, this.currentWater - leaked);
    this.lastLeakTime = nowMs;

    if (this.currentWater + weight <= this.capacity) {
      this.currentWater += weight;
      return true;
    }
    return false;
  }

  getWaterLevel(): number {
    return this.currentWater;
  }
}
""",
        test_code="""import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MicrosecondLeakyBucket } from './solution.ts';

describe('MicrosecondLeakyBucket', () => {
  it('enforces capacity burst and leak rate across virtual time progression', () => {
    const t0 = 1000;
    const bucket = new MicrosecondLeakyBucket(10, 0.1, t0); // capacity: 10, leak: 0.1 unit/ms

    // Fill to capacity
    assert.strictEqual(bucket.addPacket(6, t0), true);
    assert.strictEqual(bucket.addPacket(4, t0), true);
    assert.strictEqual(bucket.getWaterLevel(), 10);

    // Overflow rejected at same timestamp
    assert.strictEqual(bucket.addPacket(1, t0), false);

    // Advance 50 ms -> leaked 5 units (10 - 5 = 5 units remain)
    assert.strictEqual(bucket.addPacket(3, t0 + 50), true); // 5 + 3 = 8
    assert.strictEqual(bucket.getWaterLevel(), 8);

    // Advance 200 ms -> should clamp at 0 instead of going negative
    assert.strictEqual(bucket.addPacket(10, t0 + 250), true);
    assert.strictEqual(bucket.getWaterLevel(), 10);
  });
});
""",
        explanation="Clamp water level with Math.max(0, this.currentWater - leaked) to prevent negative water volume under long elapsed times."
    )

