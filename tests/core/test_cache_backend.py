import types

from aiwaf.core.cache_backend import InMemoryTTLCache


def test_inmemory_ttl_cache_expires(monkeypatch):
    clock = types.SimpleNamespace(now=1000.0)

    monkeypatch.setattr("aiwaf.core.cache_backend.time.time", lambda: clock.now)

    cache = InMemoryTTLCache(key_prefix="")
    cache.set("k", {"v": 1}, ttl_seconds=10)
    assert cache.get("k") == {"v": 1}

    clock.now += 9.9
    assert cache.get("k") == {"v": 1}

    clock.now += 0.2
    assert cache.get("k") is None

