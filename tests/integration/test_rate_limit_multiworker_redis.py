import os
import time
import uuid
import multiprocessing as mp

import pytest

from aiwaf.core.cache_backend import CacheBackendConfig, make_cache_backend
from aiwaf.core.rate_limit import build_rate_limit_key, evaluate_rate_limit, THROTTLE


def _worker(redis_url: str, key: str, window: int, max_requests: int, flood: int, conn):
    try:
        cache = make_cache_backend(CacheBackendConfig(backend="redis", redis_url=redis_url, key_prefix=""))
        now = time.time()
        timestamps = cache.get(key) or []
        decision = evaluate_rate_limit(
            timestamps=timestamps,
            now=now,
            window_seconds=window,
            max_requests=max_requests,
            flood_threshold=flood,
        )
        cache.set(key, decision.timestamps, ttl_seconds=window)
        conn.send(decision.action)
    except Exception as exc:  # pragma: no cover
        conn.send({"error": str(exc)})
    finally:
        conn.close()


@pytest.mark.slow
def test_rate_limit_redis_backend_shared_across_processes():
    """
    Verifies the Redis cache backend shares rate-limit buckets across processes.

    This is the property you need for "multiple workers": process A increments the
    same counter that process B reads.
    """
    redis_url = os.environ.get("AIWAF_TEST_REDIS_URL") or os.environ.get("AIWAF_REDIS_URL")
    if not redis_url:
        pytest.skip("Set AIWAF_TEST_REDIS_URL to run Redis multi-worker integration test")

    try:
        cache = make_cache_backend(CacheBackendConfig(backend="redis", redis_url=redis_url, key_prefix=""))
        # Best-effort connectivity check (works for redis-py client).
        cache._redis.ping()  # type: ignore[attr-defined]
    except Exception:
        pytest.skip("Redis unavailable or redis package missing")

    unique = uuid.uuid4().hex
    key = build_rate_limit_key("ratelimit", "198.51.100.10", f"/rl/{unique}", key_mode="ip_path", app_key="")

    window = 60
    max_requests = 1
    flood = 100

    ctx = mp.get_context("spawn")

    parent1, child1 = ctx.Pipe(duplex=False)
    p1 = ctx.Process(target=_worker, args=(redis_url, key, window, max_requests, flood, child1))
    p1.start()
    action1 = parent1.recv()
    p1.join(10)
    assert p1.exitcode == 0
    assert action1 != THROTTLE

    parent2, child2 = ctx.Pipe(duplex=False)
    p2 = ctx.Process(target=_worker, args=(redis_url, key, window, max_requests, flood, child2))
    p2.start()
    action2 = parent2.recv()
    p2.join(10)
    assert p2.exitcode == 0

    # Second process sees the first process's timestamp and hits soft limit.
    assert action2 == THROTTLE
