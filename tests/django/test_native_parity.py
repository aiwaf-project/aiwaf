"""The regular Django middleware and the native pipeline it switches to answer the same requests alike."""
import os
from tempfile import TemporaryDirectory
from unittest import skipUnless
from unittest.mock import patch

from django.http import Http404, JsonResponse
from django.test import Client, TestCase
from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from .test_native_default import FULL_ENGINE, REGULAR, middleware_types, settings_for

KNOWN = "6f1c2a3e-4b5d-4c6e-8f70-112233445566"


@csrf_exempt
def item(request, **kwargs):
    if kwargs.get("uuid") not in (None, KNOWN):
        raise Http404
    return JsonResponse({"ok": True})


urlpatterns = [path("items", item), path("items/<int:n>", item), path("form", item), path("user/<str:uuid>/", item)]
BROWSER = {"HTTP_USER_AGENT": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130 Safari/537.36", "HTTP_ACCEPT": "text/html",
           "HTTP_ACCEPT_LANGUAGE": "en", "HTTP_ACCEPT_ENCODING": "gzip", "HTTP_CONNECTION": "keep-alive"}
HEADER = "aiwaf.django.middleware.HeaderValidationMiddleware"
# (method, path, headers replacing the browser's; None removes one)
HEADER_CASES = [
    ("get", "/items", {}),
    ("get", "/items", {"HTTP_ACCEPT": None}),
    ("get", "/items", {"HTTP_USER_AGENT": ""}),
    ("get", "/items", {"HTTP_USER_AGENT": None}),
    ("get", "/items", {"HTTP_ACCEPT_LANGUAGE": None, "HTTP_ACCEPT_ENCODING": None, "HTTP_CONNECTION": None}),
    ("get", "/items", {"HTTP_ACCEPT": "*/*", "HTTP_ACCEPT_LANGUAGE": None, "HTTP_ACCEPT_ENCODING": None}),
    ("get", "/items", {"HTTP_USER_AGENT": "Googlebot/2.1", "HTTP_ACCEPT": None}),
    ("get", "/items", {"HTTP_USER_AGENT": "Mozilla/4.0"}),
    ("get", "/items", {"HTTP_USER_AGENT": "Mozilla/5"}),
    ("get", "/items", {"HTTP_USER_AGENT": "curl/8.0"}),
    ("get", "/static/app.css", {"HTTP_USER_AGENT": "curl/8.0"}),
    ("get", "/items/app.js", {"HTTP_USER_AGENT": "curl/8.0"}),
    ("get", "/items", {"SERVER_PROTOCOL": "HTTP/1.0"}),
    ("get", "/items", {"HTTP_USER_AGENT": "Mozilla/5.0 Chrome/130 " + "a" * 600}),
    ("get", "/items", {"HTTP_ACCEPT": "text/html," + "a" * 5000}),
    ("get", "/items", {f"HTTP_X_EXTRA_{i}": "1" for i in range(120)}),
    ("get", "/items", {"HTTP_CONNECTION": None, "HTTP_CACHE_CONTROL": None}),
    ("post", "/items", {"HTTP_ACCEPT": None}),
]


@skipUnless(FULL_ENGINE, "aiwaf-rust with every native feature required")
class ParityTestCase(TestCase):
    base = 0  # Every run gets its own client addresses, since blocks are shared through the stores.

    def setUp(self):
        patcher = patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("AIWAF_NATIVE_PIPELINE", None)
        from django.core.cache import cache

        # Rate counts and timing live in the process's cache, so each run starts clean.
        cache.clear()

    def statuses(self, native, middleware, cases, **settings):
        ParityTestCase.base += 1
        results = []
        settings = {"AIWAF_RATE_MAX": 1000, "AIWAF_RATE_FLOOD": 2000, **settings}
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=middleware, ROOT_URLCONF=__name__,
                                                             AIWAF_STORAGE_MODE="models",
                                                             AIWAF_NATIVE_PIPELINE=None if native else False, **settings):
            probe = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.250", **BROWSER)
            self.assertEqual("NativeAIWAFMiddleware" in middleware_types(probe), native, settings)
            for index, (method, path, changes) in enumerate(cases):
                headers = {**BROWSER, **changes}
                headers = {key: value for key, value in headers.items() if value is not None}
                client = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.{index + 1}")
                response = getattr(client, method)(path, **headers)
                results.append(response.status_code)
        return results

    def assertAlike(self, middleware, cases, **settings):
        regular = self.statuses(False, middleware, cases, **settings)
        native = self.statuses(True, middleware, cases, **settings)
        differences = {index: (cases[index], regular[index], native[index])
                       for index in range(len(cases)) if regular[index] != native[index]}
        self.assertEqual(differences, {}, settings)
        if os.environ.get("AIWAF_PARITY_PRINT"):
            print(settings, regular)
        return regular


class HeaderParityTests(ParityTestCase):
    def test_default_required_headers(self):
        self.assertAlike(REGULAR + [HEADER], HEADER_CASES)

    def test_no_required_headers(self):
        self.assertAlike(REGULAR + [HEADER], HEADER_CASES, AIWAF_REQUIRED_HEADERS=[])

    def test_a_higher_score_and_the_apps_caps(self):
        self.assertAlike(REGULAR + [HEADER], HEADER_CASES, AIWAF_HEADER_QUALITY_MIN_SCORE=7, AIWAF_MAX_USER_AGENT_LENGTH=1000,
                         AIWAF_MAX_ACCEPT_LENGTH=8000, AIWAF_MAX_HEADER_COUNT=200)

    def test_required_headers_per_method(self):
        self.assertAlike(REGULAR + [HEADER], HEADER_CASES,
                         AIWAF_REQUIRED_HEADERS={"GET": ["HTTP_USER_AGENT"], "DEFAULT": ["HTTP_USER_AGENT", "HTTP_ACCEPT"]})


STAGES = ["django.contrib.sessions.middleware.SessionMiddleware",
          "django.contrib.auth.middleware.AuthenticationMiddleware",
          "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
          "aiwaf.django.middleware.RateLimitMiddleware",
          "aiwaf.django.middleware.AIAnomalyMiddleware",
          "aiwaf.django.middleware.HoneypotTimingMiddleware",
          "aiwaf.django.middleware.UUIDTamperMiddleware"]


class SequenceParityTests(ParityTestCase):
    """Each case is a sequence of requests from one client; every response's status is compared."""

    def sequences(self, native, middleware, cases, **settings):  # noqa: D102
        ParityTestCase.base += 1
        settings = {"AIWAF_RATE_MAX": 1000, "AIWAF_RATE_FLOOD": 2000, **settings}
        results = []
        with TemporaryDirectory() as directory, settings_for(directory, MIDDLEWARE=middleware, ROOT_URLCONF=__name__,
                                                             AIWAF_STORAGE_MODE="models",
                                                             AIWAF_NATIVE_PIPELINE=None if native else False, **settings):
            probe = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.250", **BROWSER)
            self.assertEqual("NativeAIWAFMiddleware" in middleware_types(probe), native, settings)
            for index, requests in enumerate(cases):
                client = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.{index + 1}", **BROWSER)
                statuses = []
                for method, url, *data in requests:
                    statuses.append(getattr(client, method)(url, *data).status_code)
                results.append(statuses)
        return results

    def assertSequencesAlike(self, middleware, cases, **settings):
        regular = self.sequences(False, middleware, cases, **settings)
        native = self.sequences(True, middleware, cases, **settings)
        differences = {index: (cases[index], regular[index], native[index])
                       for index in range(len(cases)) if regular[index] != native[index]}
        self.assertEqual(differences, {}, settings)
        if os.environ.get("AIWAF_PARITY_PRINT"):
            print(settings, regular)
        return regular

    def test_keywords_and_injection(self):
        self.assertSequencesAlike(STAGES, [
            [("get", "/items")], [("get", "/wp-login.php"), ("get", "/items")], [("get", "/.env")],
            [("get", "/backup.bak")], [("get", "/items", {"q": "' OR 1=1--"})], [("get", "/items", {"q": "union select"})],
            [("post", "/form", {"name": "x'; DROP TABLE users;--"})], [("get", "/nothing-here")],
            [("get", "/xmlrpc.php")], [("get", "/items/5")], [("get", "/ShElL")],
        ])

    def test_rate_limits(self):
        self.assertSequencesAlike(STAGES, [[("get", "/items")] * 8, [("get", f"/items/{n}") for n in range(8)]],
                                  AIWAF_RATE_MAX=3, AIWAF_RATE_FLOOD=6)

    def test_honeypot_timing(self):
        self.assertSequencesAlike(STAGES, [
            [("get", "/form"), ("post", "/form", {"a": "1"})], [("post", "/form", {"a": "1"})],
        ], AIWAF_MIN_FORM_TIME=5)

    def test_uuid_tampering(self):
        self.assertSequencesAlike(STAGES, [
            [("get", f"/user/{KNOWN}/")], [("get", "/user/not-a-uuid/"), ("get", "/items")],
            [("get", "/user/6f1c2a3e-4b5d-4c6e-8f70-000000000000/")] * 7,
        ])

    def test_exempt_paths(self):
        self.assertSequencesAlike(STAGES + [HEADER], [
            [("get", "/form", {"q": "' OR 1=1--"})], [("get", "/form/wp-login.php")], [("get", "/forms")],
            [("get", "/items/.env")], [("get", "/form")] * 6,
        ], AIWAF_EXEMPT_PATHS=["/form"], AIWAF_RATE_MAX=3, AIWAF_RATE_FLOOD=6)


class OlderEngineTests(ParityTestCase):
    def test_an_aiwaf_rust_that_cant_skip_static_files_keeps_the_regular_header_check(self):
        from aiwaf.core import native_settings
        from aiwaf.django.native_default import native_plan

        supports = native_settings._engine_supports
        with patch.object(native_settings, "_engine_supports", lambda f: f not in ("header_skip", "method_headers") and supports(f)):
            with TemporaryDirectory() as directory:
                with settings_for(directory, MIDDLEWARE=REGULAR + [HEADER]):
                    self.assertIsNone(native_plan())
                with settings_for(directory, AIWAF_REQUIRED_HEADERS={"GET": ["HTTP_USER_AGENT"]}):
                    self.assertIsNone(native_plan())
                # Without the header check there's nothing to skip.
                with settings_for(directory):
                    self.assertIsNotNone(native_plan())


# Probes that look like scans, then requests that reuse their words: what each pipeline learns
# from the probes decides whether the later requests are blocked.
LEARNING = [
    [("get", "/backup-old/config.php.bak"), ("get", "/wordpress-setup/install.php"), ("get", "/dbadmin/shell.php"),
     ("get", "/temp-files/../../etc/passwd"), ("get", "/vendor/phpunit/eval-stdin.php")],
    [("get", "/items", {"cmd": "whoami"}), ("get", "/oldsite/exec?cmd=id"), ("get", "/console-debug/system.php")],
    [("get", "/nothing/backup-old"), ("get", "/wordpress-setup"), ("get", "/dbadmin"), ("get", "/oldsite"),
     ("get", "/console-debug"), ("get", "/vendor/phpunit"), ("get", "/temp-files"), ("get", "/items")],
    [("get", "/items/oldsite"), ("get", "/items/nothing"), ("get", "/form/oldsite.php"), ("get", "/userx/oldsite")],
]


class LearningParityTests(ParityTestCase):
    sequences = SequenceParityTests.sequences

    def learned(self, native, middleware, **settings):
        from aiwaf.django.models import BlacklistEntry, DynamicKeyword

        DynamicKeyword.objects.all().delete()
        BlacklistEntry.objects.all().delete()
        statuses = self.sequences(native, middleware, LEARNING, **settings)
        keywords = sorted(DynamicKeyword.objects.values_list("keyword", flat=True))
        return statuses, keywords

    def assertLearnsAlike(self, middleware, **settings):
        regular, native = self.learned(False, middleware, **settings), self.learned(True, middleware, **settings)
        if os.environ.get("AIWAF_PARITY_PRINT"):
            print(settings, regular, native)
        self.assertEqual(native[1], regular[1], "learned keywords")
        self.assertEqual(native[0], regular[0], "statuses")

    def test_keyword_learning(self):
        self.assertLearnsAlike(STAGES)

    def test_keyword_learning_without_the_anomaly_stage(self):
        self.assertLearnsAlike([entry for entry in STAGES if "AIAnomaly" not in entry])


def trained_forest():
    """An aiwaf-rust model trained on ordinary browsing: short paths, no keywords, few 404s."""
    import aiwaf_rust

    forest = aiwaf_rust.IsolationForest(n_estimators=50, random_state=7)
    forest.fit([[float(8 + i % 12), 0.0, 0.0, 0.02 + (i % 5) / 100, float(1 + i % 3), 0.0] for i in range(200)])
    return forest


# A scan: 404s for paths with the anomaly stage's keywords, but none of the keyword stage's.
SCAN = [[("get", f"/{word}/area{n}") for n, word in enumerate(
    ["backup", "database", "mysql", "dump", "setup", "install", "installer", "adminer", "cgi-bin", "scripts", "base64",
     "backup", "database", "mysql", "dump", "setup"])] + [("get", "/items")], [("get", "/items"), ("get", "/nothing-here")]]


class ModelParityTests(ParityTestCase):
    sequences = SequenceParityTests.sequences
    assertSequencesAlike = SequenceParityTests.assertSequencesAlike

    def test_the_anomaly_models_blocks(self):
        from aiwaf.django import middleware as regular

        with TemporaryDirectory() as directory:
            log = os.path.join(directory, "access.log")
            with open(log, "w") as file:
                file.write('198.51.100.1 - - [16/Nov/2026:10:00:00 +0000] "GET /items HTTP/1.1" 200 12 "-" "Mozilla/5.0"\\n')
            settings = dict(AIWAF_MIN_AI_LOGS=0, AIWAF_ACCESS_LOG=log, AIWAF_ENABLE_KEYWORD_LEARNING=False)
            with patch.object(regular, "MODEL", trained_forest()):
                answers = self.assertSequencesAlike(STAGES, SCAN, **settings)
            without = self.assertSequencesAlike(STAGES, SCAN, **settings)
        # The model changes what's blocked, so it was exercised.
        self.assertNotEqual(answers, without)


GEO = "aiwaf.django.middleware.GeoBlockMiddleware"


class GeoParityTests(ParityTestCase):
    """Countries come from a stand-in for the GeoIP lookup both pipelines use."""

    def countries(self, native, cases, **settings):
        from aiwaf.core import geoip

        ParityTestCase.base += 1
        current = {"country": None}
        results = []
        with patch.object(geoip, "lookup_country", lambda *args, **kwargs: current["country"]), TemporaryDirectory() as directory, \
                settings_for(directory, MIDDLEWARE=[GEO] + STAGES, ROOT_URLCONF=__name__, AIWAF_STORAGE_MODE="models",
                             AIWAF_RATE_MAX=1000, AIWAF_RATE_FLOOD=2000, AIWAF_NATIVE_PIPELINE=None if native else False,
                             **settings):
            probe = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.250", **BROWSER)
            self.assertEqual("NativeAIWAFMiddleware" in middleware_types(probe), native, settings)
            for index, country in enumerate(cases):
                current["country"] = country
                client = Client(REMOTE_ADDR=f"198.{50 + self.base // 250}.{self.base % 250}.{index + 1}", **BROWSER)
                results.append(client.get("/items").status_code)
        return results

    def assertCountriesAlike(self, cases, **settings):
        regular, native = self.countries(False, cases, **settings), self.countries(True, cases, **settings)
        self.assertEqual(native, regular, settings)
        if os.environ.get("AIWAF_PARITY_PRINT"):
            print(settings, regular)
        return regular

    def test_block_and_allow_lists(self):
        cases = ["US", "CN", "cn", "GB", None, "", "JP"]
        blocked = self.assertCountriesAlike(cases, AIWAF_GEO_BLOCK_ENABLED=True, AIWAF_GEO_BLOCK_COUNTRIES=["CN", "jp"])
        self.assertIn(403, blocked)
        allowed = self.assertCountriesAlike(cases, AIWAF_GEO_BLOCK_ENABLED=True, AIWAF_GEO_ALLOW_COUNTRIES=["US", "GB"])
        self.assertIn(403, allowed)
        self.assertCountriesAlike(cases, AIWAF_GEO_BLOCK_ENABLED=False, AIWAF_GEO_BLOCK_COUNTRIES=["CN"])

    def test_countries_blocked_in_the_store(self):
        from aiwaf.django.models import GeoBlockedCountry

        GeoBlockedCountry.objects.create(country_code="GB")
        self.assertIn(403, self.assertCountriesAlike(["US", "GB", "CN"], AIWAF_GEO_BLOCK_ENABLED=True,
                                                     AIWAF_GEO_BLOCK_COUNTRIES=["CN"]))

    def test_only_countries_blocked_in_the_store(self):
        from aiwaf.django.models import GeoBlockedCountry

        GeoBlockedCountry.objects.create(country_code="GB")
        self.assertCountriesAlike(["US", "GB", "CN"], AIWAF_GEO_BLOCK_ENABLED=True)
