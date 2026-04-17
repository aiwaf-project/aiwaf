"""Concurrency stress tests for rate limiting and storage behavior."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from aiwaf.core.runtime_storage import MemoryStorage
from aiwaf.core.runtime_utils import RateLimiter


@pytest.mark.slow
def test_memory_storage_thread_safety_under_parallel_sets_gets():
    store = MemoryStorage()

    def worker(i):
        key = f"k:{i}"
        store.set(key, i)
        return store.get(key)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(worker, range(1000)))

    assert results == list(range(1000))


@pytest.mark.slow
def test_rate_limiter_parallel_requests_trigger_limits():
    limiter = RateLimiter()
    ip = "203.0.113.77"
    path = "/api/load"

    def request_once(_):
        return limiter.is_rate_limited(ip, path, max_requests=50, window_seconds=30)

    with ThreadPoolExecutor(max_workers=32) as pool:
        outcomes = list(pool.map(request_once, range(120)))

    assert any(outcomes)
    assert outcomes.count(True) >= 1
