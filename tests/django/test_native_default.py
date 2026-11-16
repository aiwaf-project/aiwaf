"""The regular Django middleware runs as the native pipeline when it can do so exactly."""
import os
from tempfile import TemporaryDirectory
from unittest import skipUnless
from unittest.mock import patch

from django.core.exceptions import MiddlewareNotUsed
from django.http import JsonResponse
from django.test import Client, SimpleTestCase, override_settings
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
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=stages):
            with patch.object(regular, "MODEL", forest):
                self.assertIs(native_plan()[2], forest)
            with patch.object(regular, "MODEL", object()):
                self.assertIsNone(native_plan())
