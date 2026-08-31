from aiwaf.core.runtime_storage import initialize_storage
from aiwaf.fast.middleware.honeypot_timing_middleware import _HoneypotStateCache


def test_honeypot_cache_pop_returns_and_deletes_values():
    storage = initialize_storage("memory")
    storage.set("honeypot_get:test", 3)
    cache = _HoneypotStateCache()
    assert cache.pop("honeypot_get:test") == 3
    assert cache.pop("missing", "default") == "default"
