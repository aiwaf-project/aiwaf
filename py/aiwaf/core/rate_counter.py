"""Shared rate counts for the Rust-owned pipeline; the rate decision stays in aiwaf-rust."""
import hashlib

from .honeypot_cache import synchronous


def bounded_key(namespace, key):
    """A fixed-length key, so long paths also work with Memcached and cache key prefixes."""
    return namespace + ":" + hashlib.sha256(key.encode("utf-8")).hexdigest()


class RateCounter:
    """The interface a native pipeline's ``rate_counter`` provides.

    ``count`` records one request for ``key`` at ``now`` and returns how many requests the
    key has in the last ``window_seconds``, this one included. It must be atomic across
    workers, which is what makes the limit shared.
    """

    def count(self, key: str, now: float, window_seconds: float) -> int:
        raise NotImplementedError


class RedisRateCounter(RateCounter):
    """Counts in Redis with the regular middleware's atomic sliding-window script."""

    def __init__(self, url, *, namespace="aiwaf:native:rate:v1"):
        if not isinstance(url, str) or not url:
            raise ValueError("rate_redis_url is required for the Redis rate backend")
        if not isinstance(namespace, str) or not namespace or len(namespace.encode("utf-8")) > 100:
            raise ValueError("rate_cache_namespace must be a nonempty string of at most 100 bytes")
        from .cache_backend import RedisJSONCache
        self.cache = RedisJSONCache(url, key_prefix="")
        self.namespace = namespace

    def count(self, key, now, window_seconds):
        # The thresholds aren't used for counting; aiwaf-rust applies its own.
        decision = synchronous(self.cache.consume_rate_limit(
            bounded_key(self.namespace, key), now=now, window_seconds=window_seconds,
            max_requests=0, flood_threshold=0))
        return int(decision.count)
