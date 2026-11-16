"""Atomic sliding windows with Django's native Redis serialization preserved."""
import math

from ..core.rate_limit import consume_rate_limit, evaluate_rate_limit, RateCacheUnavailable


class _LocalCache:
    def __init__(self, cache):
        self.cache = cache

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value, ttl_seconds):
        self.cache.set(key, value, timeout=ttl_seconds)


def consume_django_rate_limit(cache, key, **options):
    try:
        return _consume_django_rate_limit(cache, key, **options)
    except RateCacheUnavailable:
        raise
    except Exception:
        raise RateCacheUnavailable('Rate cache unavailable') from None


def _consume_django_rate_limit(cache, key, **options):
    from django.core.cache.backends.redis import RedisCache

    if not isinstance(cache, RedisCache):
        # Django's default cache is a lazy ConnectionProxy.
        from django.core.cache import caches
        from django.utils.connection import ConnectionProxy
        if isinstance(cache, ConnectionProxy):
            cache = caches[cache._alias]
    if not isinstance(cache, RedisCache):
        return consume_rate_limit(_LocalCache(cache), key, **options)

    from redis.exceptions import WatchError, RedisError
    client = cache._cache
    redis_key = cache.make_and_validate_key(key)
    connection = client.get_client(redis_key, write=True)
    # WATCH/MULTI keeps existing pickle/custom-serialized buckets readable and
    # retries conflicting writes across threads, processes and app instances.
    with connection.pipeline() as transaction:
        while True:
            try:
                transaction.watch(redis_key)
                raw = transaction.get(redis_key)
                timestamps = client._serializer.loads(raw) if raw is not None else []
                decision = evaluate_rate_limit(timestamps=timestamps, **options)
                transaction.multi()
                transaction.set(redis_key, client._serializer.dumps(decision.timestamps),
                                px=math.ceil(max(float(options['window_seconds']), 1.0) * 1000))
                transaction.execute()
                return decision
            except WatchError:
                continue
            except RedisError:
                raise RateCacheUnavailable('Rate cache unavailable') from None


class DjangoRateCounter:
    """Native rate counts in a Django cache, updated as the regular rate limiter updates them.

    Shared across workers when the cache is (Redis, Memcached, database); LocMemCache counts
    per process, as it does for the regular middleware.
    """

    def __init__(self, alias="default", *, namespace="aiwaf:native:rate:v1"):
        from django.core.cache import caches
        from django.core.exceptions import ImproperlyConfigured

        if not isinstance(alias, str) or not alias.strip() or alias not in caches:
            raise ImproperlyConfigured("rate_cache_alias must name a configured Django cache")
        self.alias = alias
        self.namespace = namespace

    def count(self, key, now, window_seconds):
        from django.core.cache import caches
        from ..core.rate_counter import bounded_key

        decision = consume_django_rate_limit(caches[self.alias], bounded_key(self.namespace, key), now=now,
                                             window_seconds=window_seconds, max_requests=0, flood_threshold=0)
        return int(decision.count)
