"""Django cache compatibility and sanitized rate-cache failures."""
import sys
from unittest.mock import patch

from django.core.cache import cache
from django.core.cache.backends.locmem import LocMemCache
from django.core.cache.backends.redis import RedisCache
from django.test import SimpleTestCase, override_settings

from aiwaf.core.rate_limit import RateCacheUnavailable
from aiwaf.django.rate_cache import consume_django_rate_limit


class RateCacheTests(SimpleTestCase):
    options = dict(now=100.0, window_seconds=60, max_requests=2, flood_threshold=4)

    def test_local_cache_retains_window_and_enforces_budget(self):
        backend = LocMemCache('rate-cache-unit', {})
        backend.clear()
        backend.set('client', [1.0, 99.0], timeout=60)
        actions = [consume_django_rate_limit(backend, 'client', **self.options).action
                   for _ in range(4)]
        self.assertEqual(actions, ['allow', 'throttle', 'throttle', 'flood_block'])
        self.assertEqual(backend.get('client'), [99.0, 100.0, 100.0, 100.0, 100.0])
        backend.clear()

    @override_settings(CACHES={'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'rate-cache-proxy-unit',
    }})
    def test_lazy_default_cache_uses_configured_backend(self):
        cache.clear()
        self.assertEqual(consume_django_rate_limit(cache, 'client', **self.options).action, 'allow')
        self.assertEqual(cache.get('client'), [100.0])
        cache.clear()

    def test_local_backend_failure_is_sanitized(self):
        backend = LocMemCache('rate-cache-unavailable', {})
        with patch.object(backend, 'get', side_effect=ConnectionError('secret backend URL')):
            with self.assertRaisesRegex(RateCacheUnavailable, '^Rate cache unavailable$') as caught:
                consume_django_rate_limit(backend, 'client', **self.options)
        self.assertIsNone(caught.exception.__cause__)

    def test_existing_typed_failure_is_preserved(self):
        failure = RateCacheUnavailable('Rate cache unavailable')
        backend = LocMemCache('rate-cache-typed-failure', {})
        with patch('aiwaf.django.rate_cache.consume_rate_limit', side_effect=failure):
            with self.assertRaises(RateCacheUnavailable) as caught:
                consume_django_rate_limit(backend, 'client', **self.options)
        self.assertIs(caught.exception, failure)

    def test_redis_outage_before_transaction_is_sanitized(self):
        backend = RedisCache('redis://127.0.0.1:1', {})
        with patch.object(backend._cache, 'get_client',
                          side_effect=ConnectionError('redis://secret:password@private-host')):
            with self.assertRaisesRegex(RateCacheUnavailable, '^Rate cache unavailable$') as caught:
                consume_django_rate_limit(backend, 'isolated', **self.options)
        self.assertIsNone(caught.exception.__cause__)

    def test_missing_optional_redis_driver_is_sanitized(self):
        backend = RedisCache('redis://127.0.0.1:1', {})
        with patch.dict(sys.modules, {'redis': None}):
            with self.assertRaisesRegex(RateCacheUnavailable, '^Rate cache unavailable$') as caught:
                consume_django_rate_limit(backend, 'isolated', **self.options)
        self.assertIsNone(caught.exception.__cause__)
