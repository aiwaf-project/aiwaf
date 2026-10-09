"""Concurrency regressions for shared and local sliding-window buckets."""
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import os
import time
import uuid

import pytest
from aiwaf.core.cache_backend import DictCacheBackend, RedisJSONCache
from aiwaf.core.rate_limit import consume_rate_limit


def test_cache_failure_is_sanitized_and_never_allows_a_request():
    from aiwaf.core.rate_limit import RateCacheUnavailable
    class Unavailable:
        def get(self, _key):
            raise ConnectionError('redis://secret-password@internal-host')
    with pytest.raises(RateCacheUnavailable, match='^Rate cache unavailable$'):
        consume_rate_limit(Unavailable(), 'test', now=1, window_seconds=10, max_requests=20, flood_threshold=40)


def test_local_cache_does_not_lose_concurrent_updates():
    class SlowCache(DictCacheBackend):
        def get(self, key):
            value = super().get(key)
            time.sleep(.001)
            return value
    cache = SlowCache({})
    def consume(_):
        return consume_rate_limit(cache, "burst", now=100.0, window_seconds=10,
                                  max_requests=20, flood_threshold=40).action
    with ThreadPoolExecutor(max_workers=16) as pool:
        actions = list(pool.map(consume, range(65)))
    assert actions.count("allow") == 20
    assert actions.count("throttle") == 20
    assert actions.count("flood_block") == 25


def _redis_batch(args):
    url, key = args
    cache = RedisJSONCache(url, key_prefix="")
    return [consume_rate_limit(cache, key, now=time.time(), window_seconds=60,
                               max_requests=20, flood_threshold=40).action for _ in range(16)]


def test_redis_is_atomic_across_concurrent_processes():
    url = os.environ.get("AIWAF_TEST_REDIS_URL")
    if not url:
        pytest.skip("Set AIWAF_TEST_REDIS_URL for the live Redis concurrency regression")
    key = "aiwaf:test:atomic:" + uuid.uuid4().hex
    cache = RedisJSONCache(url, key_prefix="")
    # Keep a legacy JSON bucket: upgrades must not reset existing rate state.
    cache.set(key, [time.time()], ttl_seconds=60)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        actions = [action for batch in pool.map(_redis_batch, [(url, key)] * 4) for action in batch]
    assert actions.count("allow") == 19
    assert actions.count("throttle") == 20
    assert actions.count("flood_block") == 25
    assert len(cache.get(key)) == 65


def test_builtin_redis_client_supports_atomic_script(monkeypatch):
    from aiwaf.core.cache_backend import _SimpleRedisClient
    client = _SimpleRedisClient("localhost", 6379)
    calls = []
    monkeypatch.setattr(client, "_exec", lambda *parts: calls.append(parts) or b"[1.0]")
    assert client.eval("return ARGV[1]", 1, "key", "value") == b"[1.0]"
    assert calls == [("EVAL", "return ARGV[1]", "1", "key", "value")]


def _django_redis_batch(args):
    # Pytest adds tests/ to sys.path; its django fixture package must not shadow
    # the installed framework when a fresh interpreter starts a worker.
    import sys
    from pathlib import Path
    fixtures = Path(__file__).resolve().parents[1]
    sys.path[:] = [p for p in sys.path if Path(p).resolve() != fixtures]
    from django.core.cache.backends.redis import RedisCache
    from aiwaf.django.rate_cache import consume_django_rate_limit
    url, key = args
    cache = RedisCache(url, {'KEY_PREFIX': 'aiwaf:test:django:atomic'})
    return [consume_django_rate_limit(cache, key, now=time.time(), window_seconds=60,
                                     max_requests=20, flood_threshold=40).action for _ in range(16)]


def test_django_redis_preserves_serialized_buckets_across_processes():
    url = os.environ.get('AIWAF_TEST_REDIS_URL')
    if not url:
        pytest.skip('Set AIWAF_TEST_REDIS_URL for live Django Redis regression')
    from django.core.cache.backends.redis import RedisCache
    cache = RedisCache(url, {'KEY_PREFIX': 'aiwaf:test:django:atomic'})
    key = uuid.uuid4().hex
    cache.set(key, [time.time()], timeout=60)
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        actions = [a for batch in pool.map(_django_redis_batch, [(url, key)] * 4) for a in batch]
    assert actions.count('allow') == 19
    assert actions.count('throttle') == 20
    assert actions.count('flood_block') == 25
    assert len(cache.get(key)) == 65
    cache.delete(key)


def test_django_redis_outage_is_sanitized_even_before_transaction(monkeypatch):
    from django.core.cache.backends.redis import RedisCache
    from aiwaf.django.rate_cache import consume_django_rate_limit
    from aiwaf.core.rate_limit import RateCacheUnavailable
    cache = RedisCache('redis://127.0.0.1:1', {})
    def unavailable(*_args, **_kwargs):
        raise ConnectionError('redis://secret:password@private-host')
    monkeypatch.setattr(cache._cache, 'get_client', unavailable)
    with pytest.raises(RateCacheUnavailable, match='^Rate cache unavailable$') as error:
        consume_django_rate_limit(cache, 'isolated', now=time.time(), window_seconds=60,
                                 max_requests=20, flood_threshold=40)
    assert error.value.__cause__ is None
