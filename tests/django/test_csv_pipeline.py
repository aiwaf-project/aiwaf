from tempfile import TemporaryDirectory
from unittest import skipUnless
from unittest.mock import patch
from django.http import JsonResponse
from django.test import SimpleTestCase, TestCase, RequestFactory, override_settings
from django.urls import path
from aiwaf.django.decorators import aiwaf_exempt

try:
    import aiwaf_rust
except ImportError:
    aiwaf_rust = None


def echo(request):
    return JsonResponse({'body':request.body.decode(), 'ok':True})


def item(request, uuid):
    return JsonResponse({'uuid': str(uuid)})


urlpatterns = [path('echo', echo), path('exempt', aiwaf_exempt(echo)), path('items/<str:uuid>/', item)]


@skipUnless(aiwaf_rust and hasattr(aiwaf_rust, 'NativeCSVPipeline'), 'Native pipeline required')
class NativePipelineTests(SimpleTestCase):
    def test_native_django_policy_and_original_body(self):
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__,
            AIWAF_CSV_PIPELINE={'data_dir':directory, 'rate_max':10000, 'rate_flood':20000,
                                'min_form_time':0, 'required_headers':[]}):
            middleware = NativeAIWAFMiddleware(echo)
            factory = RequestFactory()
            with patch('aiwaf.core.sql_injection.SQLInjectionPolicy.inspect', side_effect=AssertionError('Python SQL fallback')):
                response = middleware(factory.post('/echo', data='{"hello":"world"}', content_type='application/json'))
                assert response.status_code == 200
                assert b'hello' in response.content
                assert middleware(factory.get('/echo?q=1+union+select+1')).status_code == 403
                assert middleware(factory.get('/exempt?q=1+union+select+1')).status_code == 200
                assert middleware(factory.get('/wp-content/plugins/pods/init.php')).status_code == 403

    def test_native_django_bounded_body(self):
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__,
            AIWAF_CSV_PIPELINE={'data_dir':directory, 'max_bytes':30, 'required_headers':[]}):
            middleware = NativeAIWAFMiddleware(echo)
            assert middleware(RequestFactory().post('/echo', data='x'*31, content_type='application/json')).status_code == 413


@skipUnless(aiwaf_rust and hasattr(aiwaf_rust, 'NativeCSVPipeline'), 'Native pipeline required')
class NativePipelineFromSettingsTests(SimpleTestCase):
    def test_from_settings_uses_the_regular_settings_and_django_exemptions(self):
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, override_settings(
                ROOT_URLCONF=__name__, AIWAF_CSV_PIPELINE='from_settings', AIWAF_CSV_DATA_DIR=directory,
                MIDDLEWARE=['aiwaf.django.csv_pipeline.NativeAIWAFMiddleware'],
                AIWAF_STORAGE_MODE='csv', AIWAF_RATE_MAX=1, AIWAF_RATE_FLOOD=5, AIWAF_REQUIRED_HEADERS=[],
                AIWAF_MIN_FORM_TIME=0, AIWAF_PATH_MANIFEST=directory + '/none.json',
                CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}):
            with self.assertLogs('aiwaf.native', level='INFO') as logs:
                middleware = NativeAIWAFMiddleware(echo)
            self.assertIn('Native pipeline settings', logs.output[0])
            # The test settings exempt 127.0.0.1, RequestFactory's default address, and that carries over too.
            factory = RequestFactory(REMOTE_ADDR='203.0.113.40')
            self.assertEqual([middleware(factory.get('/echo')).status_code for _ in range(2)], [200, 429])
            # Django's built-in login exemption carries over, so it isn't rate limited.
            self.assertEqual([middleware(factory.get('/login/')).status_code for _ in range(3)], [200, 200, 200])
            self.assertEqual(middleware(RequestFactory().get('/echo')).status_code, 200)


@skipUnless(aiwaf_rust and hasattr(aiwaf_rust, 'NativeCSVPipeline') and getattr(aiwaf_rust.NativeCSVPipeline, 'supports', None)
            and aiwaf_rust.NativeCSVPipeline.supports('uuid_fact'), 'Native route UUID fact required')
class NativePipelineRouteUuidTests(SimpleTestCase):
    def test_route_uuid_parameters_are_checked_as_in_the_regular_middleware(self):
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        from django.urls import resolve
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__, AIWAF_CSV_PIPELINE={
                'data_dir': directory, 'enabled': ['uuid_tamper'], 'manifest_path': directory + '/none.json'}):
            middleware = NativeAIWAFMiddleware(lambda request: resolve(request.path_info).func(request, **resolve(request.path_info).kwargs))
            factory = RequestFactory(REMOTE_ADDR='203.0.113.50')
            self.assertEqual(middleware(factory.get('/items/not-a-uuid/')).status_code, 403)
            ok = RequestFactory(REMOTE_ADDR='203.0.113.51').get('/items/550e8400-e29b-41d4-a716-446655440000/')
            self.assertEqual(middleware(ok).status_code, 200)


@skipUnless(aiwaf_rust and hasattr(aiwaf_rust, 'NativeCSVPipeline') and getattr(aiwaf_rust.NativeCSVPipeline, 'supports', None)
            and aiwaf_rust.NativeCSVPipeline.supports('host_store'), 'Native host stores required')
class NativePipelineDjangoStoreTests(TestCase):
    def test_native_pipeline_uses_the_regular_django_stores(self):
        from aiwaf.django.blocklist_manager import BlocklistManager
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        from aiwaf.django.storage import get_exemption_store
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__, AIWAF_STORAGE_MODE='models',
                AIWAF_EXEMPT_IPS=[], AIWAF_CSV_PIPELINE={'data_dir': directory, 'store': 'django', 'exempt_paths': [],
                'enabled': ['ip_keyword_block'], 'manifest_path': directory + '/none.json'}):
            middleware = NativeAIWAFMiddleware(echo)
            get = lambda path, ip: middleware(RequestFactory(REMOTE_ADDR=ip).get(path)).status_code
            # A block stored in the database applies natively.
            BlocklistManager.block('203.0.113.70', 'stored block')
            self.assertEqual(get('/echo', '203.0.113.70'), 403)
            # A native block lands in the database the regular middleware reads.
            self.assertEqual(get('/wp-admin/shell.php', '203.0.113.71'), 403)
            self.assertTrue(BlocklistManager.is_blocked('203.0.113.71'))
            # A stored exemption applies natively too.
            get_exemption_store().add_exemption('203.0.113.72', 'trusted')
            self.assertEqual(get('/wp-admin/shell.php', '203.0.113.72'), 200)
            with self.assertRaises(ValueError):
                with override_settings(AIWAF_CSV_PIPELINE={'data_dir': directory, 'store': 'redis'}):
                    NativeAIWAFMiddleware(echo)

    def test_keywords_learned_natively_land_in_the_django_keyword_store(self):
        from django.http import HttpResponseNotFound
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        from aiwaf.django.storage import get_keyword_store
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__, AIWAF_STORAGE_MODE='models',
                AIWAF_EXEMPT_IPS=[], AIWAF_CSV_PIPELINE={'data_dir': directory, 'store': 'django', 'exempt_paths': [],
                'enabled': ['ai_anomaly'], 'manifest_path': directory + '/none.json'}):
            middleware = NativeAIWAFMiddleware(lambda request: HttpResponseNotFound())
            middleware(RequestFactory(REMOTE_ADDR='203.0.113.73').get('/old/phpmyadmin-panelx'))
            self.assertIn('panelx', get_keyword_store().get_top_keywords(10))


@skipUnless(aiwaf_rust and hasattr(aiwaf_rust, 'NativeCSVPipeline')
            and getattr(aiwaf_rust.NativeCSVPipeline, 'supports', lambda _: False)('rate_counter'),
            'aiwaf-rust with rate counters required')
class NativePipelineDjangoCacheRateTests(SimpleTestCase):
    def test_rate_limits_are_counted_in_the_django_cache_and_shared(self):
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        # A cache shared between processes, as Redis or Memcached would be.
        with TemporaryDirectory() as directory, TemporaryDirectory() as cache_dir, override_settings(
                ROOT_URLCONF=__name__, CACHES={"default": {"BACKEND": "django.core.cache.backends.filebased.FileBasedCache",
                                                           "LOCATION": cache_dir}},
                AIWAF_CSV_PIPELINE={'data_dir': directory, 'enabled': ['rate_limit'], 'exempt_paths': [], 'exempt_ips': [],
                                    'rate_backend': 'django_cache', 'rate_max': 2, 'rate_flood': 10}):
            # Two middleware instances stand in for two workers sharing one cache.
            workers = [NativeAIWAFMiddleware(echo), NativeAIWAFMiddleware(echo)]
            factory = RequestFactory()
            statuses = [workers[i % 2](factory.get('/echo', REMOTE_ADDR='198.51.100.7')).status_code for i in range(3)]
            assert statuses == [200, 200, 429]

    def test_a_process_local_cache_uses_the_engines_own_memory(self):
        # LocMemCache keeps values in this process only, as the engine's memory does.
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, override_settings(
                ROOT_URLCONF=__name__, CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
                AIWAF_CSV_PIPELINE={'data_dir': directory, 'enabled': ['rate_limit', 'honeypot'], 'exempt_paths': [],
                                    'exempt_ips': [], 'rate_backend': 'django_cache', 'rate_max': 2, 'rate_flood': 10}):
            middleware = NativeAIWAFMiddleware(echo)
            assert middleware.pipeline.profile["rate_backend"] == "memory"
            assert middleware.pipeline.profile["honeypot_backend"] == "memory"
            factory = RequestFactory()
            statuses = [middleware(factory.get('/echo', REMOTE_ADDR='198.51.100.8')).status_code for _ in range(3)]
            assert statuses == [200, 200, 429]

    def test_an_unknown_cache_alias_fails_at_startup(self):
        from django.core.exceptions import ImproperlyConfigured
        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, override_settings(ROOT_URLCONF=__name__, AIWAF_CSV_PIPELINE={
                'data_dir': directory, 'rate_backend': 'django_cache', 'rate_cache_alias': 'missing'}):
            with self.assertRaises(ImproperlyConfigured):
                NativeAIWAFMiddleware(echo)


class NativeUrlResolutionCacheTests(SimpleTestCase):
    def test_recent_paths_are_resolved_once_per_urlconf_language_and_path(self):
        from unittest.mock import patch
        from django.utils import translation
        from aiwaf.django import csv_pipeline

        csv_pipeline._RESOLVED.clear()
        with patch.object(csv_pipeline, "resolve", wraps=csv_pipeline.resolve) as resolver:
            self.assertEqual(csv_pipeline._resolved("/echo", __name__)[1:], (True, None))
            self.assertEqual(csv_pipeline._resolved("/echo", __name__)[1:], (True, None))
            self.assertEqual(resolver.call_count, 1)
            self.assertEqual(csv_pipeline._resolved("/missing", __name__), (None, False, None))
            self.assertEqual(csv_pipeline._resolved("/items/abc/", __name__)[2], "abc")
            # i18n_patterns resolve per language, so the language is part of the key.
            with translation.override("fr"):
                csv_pipeline._resolved("/echo", __name__)
            self.assertEqual(resolver.call_count, 4)

    def test_the_cache_is_bounded(self):
        from unittest.mock import patch
        from aiwaf.django import csv_pipeline

        csv_pipeline._RESOLVED.clear()
        with patch.object(csv_pipeline, "_RESOLVED_LIMIT", 3):
            for index in range(5):
                csv_pipeline._resolved(f"/missing/{index}", __name__)
            self.assertEqual(len(csv_pipeline._RESOLVED), 3)
