"""The regular Django middleware runs as the native pipeline when it can do so exactly."""
import os
from tempfile import TemporaryDirectory
from unittest import skipUnless
from unittest.mock import patch

from django.core.exceptions import MiddlewareNotUsed
from django.http import JsonResponse
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.urls import path

try:
    import aiwaf_rust
    from aiwaf.core.native_settings import DEFAULT_FEATURES
    FULL_ENGINE = all(aiwaf_rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
        if hasattr(getattr(aiwaf_rust, "NativeCSVPipeline", None), "supports") else False
except ImportError:
    FULL_ENGINE = False


def item(request):
    return JsonResponse({"ok": True})


urlpatterns = [path("items", item)]
REGULAR = ["django.contrib.sessions.middleware.SessionMiddleware",
           "django.contrib.auth.middleware.AuthenticationMiddleware",
           "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
           "aiwaf.django.middleware.RateLimitMiddleware"]
NATIVE = "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware"


def settings_for(directory, **extra):
    values = dict(ROOT_URLCONF=__name__, MIDDLEWARE=REGULAR, AIWAF_CSV_DATA_DIR=directory,
                  AIWAF_RATE_MAX=1, AIWAF_RATE_FLOOD=10, AIWAF_EXEMPT_IPS=[], AIWAF_EXEMPT_PATHS=[])
    return override_settings(**{**values, **extra})


def access_log(directory):
    """A web server access log with a request in it, which aiwaf train reads."""
    path = os.path.join(directory, "access.log")
    with open(path, "w") as file:
        file.write('198.51.100.1 - - [16/Nov/2026:10:00:00 +0000] "GET /items HTTP/1.1" 200 12 "-" "Mozilla/5.0"\n')
    return path


def middleware_types(client):
    client.get("/items")  # Loads the middleware chain.
    handler, names = client.handler._middleware_chain, []
    while handler is not None:
        # Django wraps each middleware with functools.wraps.
        while hasattr(handler, "__wrapped__"):
            handler = handler.__wrapped__
        names.append(type(handler).__name__)
        handler = getattr(handler, "get_response", None)
    return names


@skipUnless(FULL_ENGINE, "aiwaf-rust with every native feature required")
class NativeDefaultTests(SimpleTestCase):
    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("AIWAF_NATIVE_PIPELINE", None)

    def test_the_regular_middleware_runs_as_native_when_everything_carries_over(self):
        with TemporaryDirectory() as directory, settings_for(directory):
            with self.assertLogs("aiwaf.native", "INFO") as logs:
                client = Client(REMOTE_ADDR="198.51.100.31")
                names = middleware_types(client)
            self.assertIn("running the native pipeline for Django", "\n".join(logs.output))
            # One native middleware where the last regular entry was, after sessions and auth.
            self.assertEqual(names.count("NativeAIWAFMiddleware"), 1)
            self.assertNotIn("IPAndKeywordBlockMiddleware", names)
            self.assertNotIn("RateLimitMiddleware", names)
            self.assertEqual(client.get("/items").status_code, 429)  # AIWAF_RATE_MAX carried over.

    def test_it_can_be_turned_off(self):
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_NATIVE_PIPELINE=False):
            names = middleware_types(Client(REMOTE_ADDR="198.51.100.32"))
            self.assertNotIn("NativeAIWAFMiddleware", names)
        os.environ["AIWAF_NATIVE_PIPELINE"] = "off"
        with TemporaryDirectory() as directory, settings_for(directory):
            self.assertNotIn("NativeAIWAFMiddleware", middleware_types(Client(REMOTE_ADDR="198.51.100.33")))

    def test_settings_that_dont_carry_over_keep_the_regular_middleware(self):
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_SOMETHING_UNKNOWN=True):
            names = middleware_types(Client(REMOTE_ADDR="198.51.100.34"))
            self.assertIn("IPAndKeywordBlockMiddleware", names)
            self.assertNotIn("NativeAIWAFMiddleware", names)

    def test_classes_used_directly_or_alongside_explicit_native_stay_regular(self):
        from aiwaf.django.middleware import IPAndKeywordBlockMiddleware
        from aiwaf.django.native_default import native_plan

        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=[]):
            self.assertIsInstance(IPAndKeywordBlockMiddleware(item), IPAndKeywordBlockMiddleware)
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=REGULAR + [NATIVE]):
            self.assertIsNone(native_plan())

    def test_only_the_host_entry_runs_and_others_step_aside(self):
        from aiwaf.django.middleware import IPAndKeywordBlockMiddleware, RateLimitMiddleware

        with TemporaryDirectory() as directory, settings_for(directory):
            with self.assertRaises(MiddlewareNotUsed):
                IPAndKeywordBlockMiddleware(item)
            self.assertEqual(type(RateLimitMiddleware(get_response=item)).__name__, "NativeAIWAFMiddleware")

    def test_an_engine_without_every_feature_keeps_the_regular_middleware(self):
        from aiwaf.core import native_settings

        with patch.object(native_settings, "_engine_supports", lambda feature: feature != "rate_counter"):
            with TemporaryDirectory() as directory, settings_for(directory):
                self.assertNotIn("NativeAIWAFMiddleware", middleware_types(Client(REMOTE_ADDR="198.51.100.35")))

    def test_a_loaded_model_carries_over_only_if_aiwaf_rust_made_it(self):
        from aiwaf.django import middleware as regular
        from aiwaf.django.native_default import native_plan

        stages = REGULAR + ["aiwaf.django.middleware.AIAnomalyMiddleware"]
        forest = aiwaf_rust.IsolationForest(n_estimators=5, random_state=1)
        forest.fit([[float(i), 0.0, 0.0, 1.0, 1.0, 0.0] for i in range(16)])
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, AIWAF_MIN_AI_LOGS=0,
                                                             AIWAF_ACCESS_LOG=access_log(directory)):
            with patch.object(regular, "MODEL", forest):
                self.assertIs(native_plan()[2], forest)
            with patch.object(regular, "MODEL", object()):
                self.assertIsNone(native_plan())
        # Without enough logs the regular anomaly middleware drops its model, so native runs without it.
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, AIWAF_MIN_AI_LOGS=5,
                                                             AIWAF_ACCESS_LOG=access_log(directory)):
            for model in (forest, object()):
                with patch.object(regular, "MODEL", model), \
                        patch.object(regular.AIAnomalyMiddleware, "_check_ai_log_sufficiency", lambda self: (False, 0)):
                    self.assertIsNone(native_plan()[2])

    def test_settings_native_reads_the_same_way_or_never_reads_dont_keep_the_regular_middleware(self):
        same = dict(AIWAF_ACCESS_LOG="/var/log/nginx/access.log", AIWAF_MODEL_PATH="/tmp/model.json",
                    AIWAF_MODEL_STORAGE="file", AIWAF_MIN_AI_LOGS=500, AIWAF_DISABLE_AI=True,
                    AIWAF_AI_CONTAMINATION=0.05, AIWAF_MIN_TRAIN_LOGS=50, AIWAF_RUST_ISOLATION_FOREST=True,
                    AIWAF_GEOIP_DB_PATH="/tmp/geo.mmdb", AIWAF_GEO_CACHE_SECONDS=60, AIWAF_GEO_CACHE_PREFIX="geo:",
                    AIWAF_LOG_FORMAT="json", AIWAF_LOG_LEVEL="INFO", AIWAF_RUST_REQUEST_ENGINE=True,
                    AIWAF_BLOCKLIST_STORE_EXTENDED_INFO=True, AIWAF_BLOCKLIST_MAX_HEADERS=10,
                    AIWAF_BLOCKLIST_REDACT_HEADERS=["Cookie"], AIWAF_ALLOWED_PATH_KEYWORDS=["widgets"])
        with TemporaryDirectory() as directory, settings_for(directory, **same):
            self.assertIn("NativeAIWAFMiddleware", middleware_types(Client(REMOTE_ADDR="198.51.100.40")))

    def test_requests_are_saved_for_training_only_without_an_access_log_to_train_from(self):
        # Without one, aiwaf train reads the requests the regular anomaly middleware saves to RequestLog.
        from aiwaf.django.native_default import native_plan

        stages = REGULAR + ["aiwaf.django.middleware.AIAnomalyMiddleware"]
        with TemporaryDirectory() as directory:
            missing, empty = os.path.join(directory, "none.log"), os.path.join(directory, "empty.log")
            open(empty, "w").close()
            for extra in ({}, {"AIWAF_ACCESS_LOG": missing}, {"AIWAF_ACCESS_LOG": empty}):
                with settings_for(directory, MIDDLEWARE=stages, **extra):
                    self.assertIs(native_plan()[1]["save_request_log"], True)
            with settings_for(directory, MIDDLEWARE=stages, AIWAF_ACCESS_LOG=access_log(directory)):
                self.assertNotIn("save_request_log", native_plan()[1])
            # A rotated log counts, as the trainer reads those too.
            os.rename(access_log(directory), os.path.join(directory, "access.log.1"))
            with settings_for(directory, MIDDLEWARE=stages, AIWAF_ACCESS_LOG=os.path.join(directory, "access.log")):
                self.assertNotIn("save_request_log", native_plan()[1])
            # Without the anomaly stage nothing is saved.
            with settings_for(directory):
                self.assertNotIn("save_request_log", native_plan()[1])

    def test_header_caps_and_the_anomaly_window_carry_over(self):
        from aiwaf.core import native_settings
        from aiwaf.django.native_default import native_plan

        stages = REGULAR + ["aiwaf.django.middleware.HeaderValidationMiddleware"]
        caps = dict(AIWAF_MAX_HEADER_BYTES=4096, AIWAF_MAX_HEADER_COUNT=40, AIWAF_MAX_USER_AGENT_LENGTH=1000,
                    AIWAF_MAX_ACCEPT_LENGTH=512, AIWAF_WINDOW_SECONDS=30)
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, **caps):
            config = native_plan()[1]
            self.assertEqual([config[key] for key in ("header_max_bytes", "header_max_count", "header_max_user_agent",
                                                      "header_max_accept", "anomaly_window")], [4096, 40, 1000, 512, 30.0])
            # A long User-Agent the app allows passes natively, as it does in the regular check.
            client = Client(REMOTE_ADDR="198.51.100.46", HTTP_USER_AGENT="Mozilla/5.0 Chrome/130 " + "a" * 700,
                            HTTP_ACCEPT="text/html", HTTP_ACCEPT_LANGUAGE="en", HTTP_ACCEPT_ENCODING="gzip",
                            HTTP_CONNECTION="keep-alive")
            self.assertIn("NativeAIWAFMiddleware", middleware_types(client))
            self.assertEqual(client.get("/items").status_code, 429)  # The rate limit, not the header check.
        # An aiwaf-rust with fixed caps keeps the regular middleware; a window under a second too.
        supports = native_settings._engine_supports
        with patch.object(native_settings, "_engine_supports", lambda feature: feature != "header_limits" and supports(feature)):
            with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, AIWAF_MAX_HEADER_COUNT=40):
                self.assertIsNone(native_plan())
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_WINDOW_SECONDS=0.5):
            self.assertIsNone(native_plan())

    def test_the_regular_keyword_lists_carry_over(self):
        from aiwaf.django.native_default import native_plan

        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_ALLOWED_PATH_KEYWORDS=["widgets"],
                                                             AIWAF_EXEMPT_KEYWORDS=["gadgets"]):
            legitimate = native_plan()[1]["legitimate_keywords"]
        self.assertTrue({"widgets", "gadgets", "items", "profile"} <= set(legitimate))

    def test_the_geoip_cache_settings_are_read_as_the_regular_middleware_reads_them(self):
        from aiwaf.django.middleware import RateLimitMiddleware

        with TemporaryDirectory() as directory, settings_for(directory):
            self.assertEqual(RateLimitMiddleware(item).geo_cache, ("aiwaf:geo:", 3600))
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_GEO_CACHE_SECONDS=60,
                                                             AIWAF_GEO_CACHE_PREFIX="geo:"):
            self.assertEqual(RateLimitMiddleware(item).geo_cache, ("geo:", 60))

    def test_a_native_pipeline_that_cant_start_falls_back_to_the_regular_middleware(self):
        from aiwaf.django import csv_pipeline, native_default

        self.addCleanup(native_default._FAILED.clear)
        with TemporaryDirectory() as directory, settings_for(directory):
            with patch.object(csv_pipeline, "CSVPipeline", side_effect=ValueError("bad profile")), \
                    self.assertLogs("aiwaf.native", "WARNING") as logs:
                client = Client(REMOTE_ADDR="198.51.100.42")
                names = middleware_types(client)
            self.assertIn("couldn't start for Django (ValueError: bad profile)", logs.output[0])
            self.assertEqual(len(logs.output), 1)
            # Every regular stage runs, not only the one that would have hosted the native pipeline.
            self.assertIn("IPAndKeywordBlockMiddleware", names)
            self.assertIn("RateLimitMiddleware", names)
            self.assertNotIn("NativeAIWAFMiddleware", names)
            self.assertEqual(client.get("/items").status_code, 429)

    def test_the_reason_an_app_stays_regular_is_logged_once(self):
        from aiwaf.core import native_settings
        from aiwaf.django.native_default import native_plan

        native_settings._REPORTED.clear()
        self.addCleanup(native_settings._REPORTED.clear)
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_SOMETHING_UNKNOWN=True):
            with self.assertLogs("aiwaf.native", "INFO") as logs:
                self.assertIsNone(native_plan())
                self.assertIsNone(native_plan())
            self.assertEqual(len(logs.output), 1)
            self.assertIn("running the regular middleware for Django", logs.output[0])
            self.assertIn("AIWAF_SOMETHING_UNKNOWN: not carried", logs.output[0])
        # An app that turned native off itself isn't told why it's off.
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_NATIVE_PIPELINE=False):
            with self.assertNoLogs("aiwaf.native", "INFO"):
                self.assertIsNone(native_plan())

    def test_a_rule_limit_above_the_flood_threshold_starts_and_blocks_where_the_regular_one_does(self):
        # Took sites down on 1.2.0: a '/' rule raising MAX above the default flood threshold of 40.
        rules = [{"PREFIX": "/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 100}}]
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_PATH_RULES=rules, AIWAF_RATE_MAX=20,
                                                             AIWAF_RATE_FLOOD=40):
            client = Client(REMOTE_ADDR="198.51.100.36")
            self.assertIn("NativeAIWAFMiddleware", middleware_types(client))
            # middleware_types made the first request, so these are requests 2 to 41.
            statuses = [client.get("/items").status_code for _ in range(40)]
            self.assertEqual(statuses[-2:], [200, 403])

    def test_a_cache_native_honeypot_timing_cant_use_keeps_the_regular_middleware(self):
        stages = REGULAR + ["aiwaf.django.middleware.HoneypotTimingMiddleware"]
        for options in ({"BACKEND": "django.core.cache.backends.dummy.DummyCache"},
                        {"BACKEND": "django.core.cache.backends.filebased.FileBasedCache", "LOCATION": "/tmp/aiwaf-cache-test",
                         "OPTIONS": {"IGNORE_EXCEPTIONS": True}}):
            with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, CACHES={"default": options}):
                names = middleware_types(Client(REMOTE_ADDR="198.51.100.37"))
                self.assertIn("HoneypotTimingMiddleware", names)
                self.assertNotIn("NativeAIWAFMiddleware", names)
        # A plain LocMemCache isn't called (native keeps timing in its own memory), so even with
        # ignored errors it switches.
        for options in ({"BACKEND": "django.core.cache.backends.locmem.LocMemCache"},
                        {"BACKEND": "django.core.cache.backends.locmem.LocMemCache", "OPTIONS": {"IGNORE_EXCEPTIONS": True}}):
            with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, CACHES={"default": options}):
                self.assertIn("NativeAIWAFMiddleware", middleware_types(Client(REMOTE_ADDR="198.51.100.38")))

    def test_a_data_directory_native_cant_create_keeps_the_regular_middleware(self):
        # The regular middleware with model storage writes no files; native keeps files there.
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_CSV_DATA_DIR="/proc/aiwaf-cant-create"):
            names = middleware_types(Client(REMOTE_ADDR="198.51.100.39"))
            self.assertIn("IPAndKeywordBlockMiddleware", names)
            self.assertNotIn("NativeAIWAFMiddleware", names)


@skipUnless(FULL_ENGINE, "aiwaf-rust with every native feature required")
class NativeDefaultDatabaseTests(TestCase):
    """What the native pipeline writes to the regular models, as the regular middleware does."""

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("AIWAF_NATIVE_PIPELINE", None)

    def test_requests_the_anomaly_stage_checks_are_saved_to_request_log(self):
        from aiwaf.django.models import RequestLog

        stages = REGULAR + ["aiwaf.django.middleware.AIAnomalyMiddleware"]
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages, AIWAF_RATE_MAX=100,
                                                             AIWAF_RATE_FLOOD=200, AIWAF_STORAGE_MODE="models",
                                                             AIWAF_EXEMPT_PATHS=["/items/exempt"]):
            client = Client(REMOTE_ADDR="198.51.100.47", HTTP_USER_AGENT="Mozilla/5.0 Chrome/130",
                            HTTP_REFERER="https://example.com/")
            self.assertIn("NativeAIWAFMiddleware", middleware_types(client))
            client.get("/items")
            client.get("/wp-admin.php")  # Blocked natively, so not saved, as in the regular middleware.
        rows = list(RequestLog.objects.order_by("id").values_list("ip_address", "method", "path", "status_code",
                                                                  "user_agent", "referer"))
        self.assertEqual(rows, [("198.51.100.47", "GET", "/items", 200, "Mozilla/5.0 Chrome/130", "https://example.com/")] * 2)

    def test_a_block_records_the_requests_headers_when_the_app_asks(self):
        from aiwaf.django.models import BlocklistEntry

        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_STORAGE_MODE="models",
                                                             AIWAF_BLOCKLIST_STORE_EXTENDED_INFO=True):
            client = Client(REMOTE_ADDR="198.51.100.48", HTTP_USER_AGENT="Mozilla/5.0 Chrome/130",
                            HTTP_COOKIE="session=secret")
            self.assertIn("NativeAIWAFMiddleware", middleware_types(client))
            self.assertEqual(client.get("/wp-login.php").status_code, 403)
        entry = BlocklistEntry.objects.get(ip_address="198.51.100.48")
        info = entry.extended_request_info
        self.assertEqual(info["headers"]["User-Agent"], "Mozilla/5.0 Chrome/130")
        self.assertNotIn("secret", str(info))


@skipUnless(FULL_ENGINE, "aiwaf-rust with every native feature required")
class NativeDefaultLoggerTests(SimpleTestCase):
    """The logger middleware logs beside the native pipeline what it logs beside the regular one."""

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("AIWAF_NATIVE_PIPELINE", None)

    def logged(self, directory, middleware, native, ip):
        from aiwaf.core.logs import csv_log_path

        log = os.path.join(directory, f"requests-{native}.log")
        with settings_for(directory, MIDDLEWARE=middleware, AIWAF_MIDDLEWARE_LOGGING=True, AIWAF_MIDDLEWARE_LOG=log,
                          AIWAF_MIDDLEWARE_DB=False, AIWAF_NATIVE_PIPELINE=None if native else False):
            client = Client(REMOTE_ADDR=ip, HTTP_USER_AGENT="Mozilla/5.0 Chrome/130")
            names = middleware_types(client)
            self.assertEqual("NativeAIWAFMiddleware" in names, native)
            self.assertIn("AIWAFLoggerMiddleware", names)
            client.get("/items")  # Over the rate limit of one.
        with open(csv_log_path(log)) as file:
            return [line.split(",")[3:5] for line in file.read().splitlines()[1:]]

    def test_listed_after_or_before_every_other_aiwaf_middleware(self):
        logger = "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware"
        with TemporaryDirectory() as directory:
            # Last: blocks happen before it, so it logs only what reaches it.
            last = [self.logged(directory, REGULAR + [logger], native, f"198.51.100.{60 + native}") for native in (False, True)]
            self.assertEqual(last[1], last[0])
            self.assertEqual(last[1], [["/items", "200"]])
        with TemporaryDirectory() as directory:
            # First: it logs the blocks too.
            first = REGULAR[:2] + [logger] + REGULAR[2:]
            rows = [self.logged(directory, first, native, f"198.51.100.{62 + native}") for native in (False, True)]
            self.assertEqual(rows[1], rows[0])
            self.assertEqual(rows[1], [["/items", "200"], ["/items", "429"]])

    def test_between_other_aiwaf_middleware_it_needs_an_engine_that_names_the_blocking_stage(self):
        from aiwaf.core import native_settings
        from aiwaf.django.native_default import native_plan

        between = REGULAR[:3] + ["aiwaf.django.middleware_logger.AIWAFLoggerMiddleware"] + REGULAR[3:]
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=between, AIWAF_MIDDLEWARE_LOGGING=True):
            self.assertEqual(native_plan()[1]["logger_outer_stages"], ["ip_keyword_block"])
            supports = native_settings._engine_supports
            with patch.object(native_settings, "_engine_supports", lambda f: f != "blocked_stage" and supports(f)):
                self.assertIsNone(native_plan())


class MonitorProfileTests(SimpleTestCase):
    """AIWAF_PROFILE=monitor runs the regular middleware and logs what it would block."""

    def setUp(self):
        from aiwaf.core import profiles

        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("AIWAF_NATIVE_PIPELINE", None)
        self.addCleanup(profiles.activate, "balanced")

    def test_monitor_lets_through_what_it_would_block_and_says_why(self):
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_PROFILE="monitor"):
            client = Client(REMOTE_ADDR="198.51.100.161")
            with self.assertLogs("aiwaf.monitor", "WARNING") as logs:
                names = middleware_types(client)
                statuses = [client.get("/items").status_code for _ in range(3)]
                # A probe blocklists the IP for later requests; in monitor it isn't blocklisted.
                statuses += [client.get("/.env").status_code, client.get("/items").status_code]
            self.assertEqual(statuses, [200, 200, 200, 404, 200])
            self.assertNotIn("NativeAIWAFMiddleware", names)
            self.assertIn("MonitoredMiddleware", names)
            output = "\n".join(logs.output)
            self.assertIn("AIWAF monitor: would block GET /items from 198.51.100.161 with 429 (RateLimitMiddleware", output)
            self.assertIn("AIWAF monitor: would add 198.51.100.161 to the blocklist (Keyword block", output)

    def test_the_balanced_profile_still_blocks(self):
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_PROFILE="balanced", AIWAF_NATIVE_PIPELINE=False):
            client = Client(REMOTE_ADDR="198.51.100.162")
            self.assertEqual([client.get("/items").status_code for _ in range(2)], [200, 429])

    def test_an_explicit_native_pipeline_is_refused_in_monitor(self):
        from django.core.exceptions import ImproperlyConfigured

        from aiwaf.django.csv_pipeline import NativeAIWAFMiddleware
        with TemporaryDirectory() as directory, settings_for(directory, AIWAF_PROFILE="monitor"):
            with self.assertRaisesMessage(ImproperlyConfigured, "AIWAF_PROFILE=monitor runs the regular middleware"):
                NativeAIWAFMiddleware(item)

    def test_process_view_blocks_are_logged_and_let_through(self):
        from django.core.exceptions import PermissionDenied
        from django.http import HttpResponse
        from django.test import RequestFactory

        from aiwaf.django.native_default import MonitoredMiddleware

        class Blocking:
            def __init__(self, get_response):
                self.get_response = get_response

            def __call__(self, request):
                return self.get_response(request)

            def process_view(self, request, view, args, kwargs):
                if request.path == "/denied":
                    raise PermissionDenied("uuid tamper")
                return HttpResponse("blocked", status=403) if request.path == "/blocked" else None

            def process_exception(self, request, exception):
                return HttpResponse("handled", status=500)

        middleware = MonitoredMiddleware(Blocking, lambda request: HttpResponse("ok"))
        factory = RequestFactory(REMOTE_ADDR="198.51.100.163")
        with self.assertLogs("aiwaf.monitor", "WARNING") as logs:
            for path in ("/blocked", "/denied", "/fine"):
                self.assertIsNone(middleware.process_view(factory.get(path), item, (), {}))
        self.assertEqual(middleware.process_exception(factory.get("/x"), ValueError()).status_code, 500)
        self.assertEqual(middleware(factory.get("/x")).status_code, 200)
        output = "\n".join(logs.output)
        self.assertIn("would block GET /blocked from 198.51.100.163 with 403 (Blocking: blocked)", output)
        self.assertIn("would block GET /denied from 198.51.100.163 with 403 (Blocking: uuid tamper)", output)
