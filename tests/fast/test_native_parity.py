"""FastAPI's regular middleware and the native pipeline it switches to answer the same requests alike."""
import os
import sys

import pytest

pytest.importorskip("fastapi")
rust = pytest.importorskip("aiwaf_rust")
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from aiwaf.core.native_settings import DEFAULT_FEATURES  # noqa: E402

FULL_ENGINE = all(rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
    if hasattr(rust.NativeCSVPipeline, "supports") else False
pytestmark = pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")

KNOWN = "6f1c2a3e-4b5d-4c6e-8f70-112233445566"
BROWSER = {"user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130 Safari/537.36", "accept": "text/html",
           "accept-language": "en", "accept-encoding": "gzip", "connection": "keep-alive"}
HEADER_CASES = [
    [("get", "/items", {})], [("get", "/items", {"accept": None})], [("get", "/items", {"user-agent": ""})],
    [("get", "/items", {"user-agent": None})],
    [("get", "/items", {"accept-language": None, "accept-encoding": None, "connection": None})],
    [("get", "/items", {"accept": "*/*", "accept-language": None, "accept-encoding": None})],
    [("get", "/items", {"user-agent": "Googlebot/2.1", "accept": None})], [("get", "/items", {"user-agent": "Mozilla/4.0"})],
    [("get", "/items", {"user-agent": "Mozilla/5.0"})], [("get", "/items", {"user-agent": "Mozilla/5"})],
    [("get", "/items", {"user-agent": "curl/8.0"})], [("get", "/static/app.css", {"user-agent": "curl/8.0"})],
    [("get", "/items/app.js", {"user-agent": "curl/8.0"})],
    [("get", "/items", {"user-agent": "Mozilla/5.0 Chrome/130 " + "a" * 600})],
    [("get", "/items", {f"x-extra-{i}": "1" for i in range(120)})], [("post", "/form", {"accept": None})],
]
SEQUENCES = [
    [("get", "/items", {})], [("get", "/wp-login.php", {}), ("get", "/items", {})], [("get", "/.env", {})],
    [("get", "/items?q=' OR 1=1--", {})], [("get", "/items?q=union select", {})], [("get", "/nothing-here", {})],
    [("get", "/xmlrpc.php", {})], [("get", "/ShElL", {})], [("post", "/form?name=x'; DROP TABLE users;--", {})],
    [("get", f"/user/{KNOWN}", {})], [("get", "/user/not-a-uuid", {}), ("get", "/items", {})],
    [("get", "/user/6f1c2a3e-4b5d-4c6e-8f70-000000000000", {})] * 7,
]
learned = {}
SECTIONS = ["ip_keyword_block", "rate_limiting", "ai_anomaly", "honeypot", "uuid_tamper", "header_validation"]
counter = [0]
# The country the stand-in GeoIP lookup reports for the request being sent.
GEO = {"country": None}
STORED_COUNTRIES = []
LOGGING = {"on": False}


@pytest.fixture(autouse=True)
def automatic(monkeypatch, tmp_path):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)


def run(tmp_path, monkeypatch, native, cases, **values):
    from aiwaf.core.runtime_config import AIWAFConfig
    from aiwaf.fast import AIWAF

    from aiwaf.core.runtime_storage import initialize_storage

    monkeypatch.setenv("AIWAF_DATA_DIR", str(tmp_path / f"data-{native}"))
    # Each run starts with empty stores: blocks and learned keywords are kept in memory.
    initialize_storage(backend="memory")
    config = AIWAFConfig(load_from_env=False)
    for section in SECTIONS:
        config.set(f"{section}.enabled", True)
    config.set("logging_middleware.enabled", LOGGING["on"])
    config.set("logging_middleware.log_dir", str(tmp_path / f"logs-{native}"))
    config.set("rate_limiting.max_requests", 1000)
    config.set("rate_limiting.flood_threshold", 2000)
    config.set("path_manifest", str(tmp_path / "none.json"))
    for key, value in values.items():
        config.set(key, value)
    app = FastAPI()

    @app.api_route("/items", methods=["GET", "POST"])
    @app.api_route("/items/{n}", methods=["GET"])
    @app.api_route("/form", methods=["GET", "POST"])
    async def item(request: Request):
        return {"ok": True}

    @app.get("/user/{uuid}")
    async def user(uuid: str):
        if uuid != KNOWN:
            raise HTTPException(status_code=404)
        return {"ok": True}

    aiwaf = AIWAF(app, config=config, native_pipeline=None if native else False)
    # Countries stored as blocked, added once AIWAF has set up its storage.
    from aiwaf.core.runtime_storage import get_geo_block_store

    for country in STORED_COUNTRIES:
        get_geo_block_store().add_country(country)
    assert bool(aiwaf.native) == native, values
    client = TestClient(app)
    client.headers.clear()  # Only the headers each case names.
    results = []
    for requests in cases:
        counter[0] += 1
        ip = f"198.{70 + counter[0] // 250}.{counter[0] % 250}.1"
        statuses = []
        for method, url, changes in requests:
            changes = dict(changes)
            GEO["country"] = changes.pop("_country", None)
            headers = {key: value for key, value in {**BROWSER, **changes, "x-forwarded-for": ip}.items() if value is not None}
            client_ip = TestClient(app, client=(ip, 5000))
            client_ip.headers.clear()
            statuses.append(client_ip.request(method.upper(), url, headers=headers).status_code)
        results.append(statuses)
    return results


def alike(tmp_path, monkeypatch, cases, **values):
    regular = run(tmp_path, monkeypatch, False, cases, **values)
    from aiwaf.core.runtime_storage import get_keyword_store

    learned[False] = sorted(get_keyword_store().get_all_keywords())
    native = run(tmp_path, monkeypatch, True, cases, **values)
    learned[True] = sorted(get_keyword_store().get_all_keywords())
    differences = {index: (cases[index], regular[index], native[index])
                   for index in range(len(cases)) if regular[index] != native[index]}
    assert differences == {}
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(values, regular)
    return regular


@pytest.mark.parametrize("values", [{}, {"header_validation.quality_threshold": 7}])
def test_headers(tmp_path, monkeypatch, values):
    alike(tmp_path, monkeypatch, HEADER_CASES, **values)


def test_keywords_injection_and_uuids(tmp_path, monkeypatch):
    alike(tmp_path, monkeypatch, SEQUENCES)


def test_rate_limits(tmp_path, monkeypatch):
    alike(tmp_path, monkeypatch, [[("get", "/items", {})] * 8, [("get", f"/items/{n}", {}) for n in range(8)]],
          **{"rate_limiting.max_requests": 3, "rate_limiting.flood_threshold": 6})


def test_honeypot_timing(tmp_path, monkeypatch):
    alike(tmp_path, monkeypatch, [[("get", "/form", {}), ("post", "/form", {})], [("post", "/form", {})]],
          **{"honeypot.min_form_time": 5})


def test_an_aiwaf_rust_that_cant_match_the_header_check_keeps_the_regular_middleware(tmp_path, monkeypatch):
    from aiwaf.core import native_settings

    supports = native_settings._engine_supports
    monkeypatch.setattr(native_settings, "_engine_supports", lambda f: f != "extra_user_agents" and supports(f))
    with pytest.raises(AssertionError):
        run(tmp_path, monkeypatch, True, [])


LEARNING = [
    [("get", "/backup-old/config.php.bak", {}), ("get", "/wordpress-setup/install.php", {}), ("get", "/dbadmin/shell.php", {}),
     ("get", "/temp-files/../../etc/passwd", {}), ("get", "/vendor/phpunit/eval-stdin.php", {})],
    [("get", "/items?cmd=whoami", {}), ("get", "/oldsite/exec?cmd=id", {}), ("get", "/console-debug/system.php", {}),
     ("get", "/reports-archive/data?q=reports-archive UNION SELECT", {}), ("get", "/legacy-filemanager/xmlrpc-relay", {})],
    [("get", "/nothing/backup-old", {}), ("get", "/wordpress-setup", {}), ("get", "/dbadmin", {}), ("get", "/oldsite", {}),
     ("get", "/console-debug", {}), ("get", "/reports-archive", {}), ("get", "/legacy-filemanager", {}), ("get", "/items", {})],
    # FastAPI learns a segment that the query repeats beside script words.
    [("get", "/catalog-pages/browse?sort=catalog-pages alert", {})], [("get", "/catalog-pages", {})],
]


def test_keyword_learning(tmp_path, monkeypatch):
    statuses = alike(tmp_path, monkeypatch, LEARNING)
    assert learned[True] == learned[False]
    assert learned[True], "nothing was learned, so the rule wasn't exercised"
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(statuses, learned[False])


@pytest.fixture
def countries(monkeypatch):
    """Countries from a stand-in for the GeoIP lookup both pipelines use."""
    from aiwaf.core import geoip

    monkeypatch.setattr(geoip, "lookup_country", lambda *args, **kwargs: GEO["country"])
    monkeypatch.setattr(sys.modules[__name__], "SECTIONS", SECTIONS + ["geo_block"])


GEO_CASES = [[("get", "/items", {"_country": country})] for country in ["US", "CN", "cn", "GB", None, "", "JP"]]


@pytest.mark.parametrize("values", [
    {"geo_block.block_countries": ["CN", "jp"]},
    {"geo_block.allow_countries": ["US", "GB"]},
    {"geo_block.block_countries": ["CN"], "geo_block.enabled": False},
])
def test_geo_blocking(tmp_path, monkeypatch, countries, values):
    alike(tmp_path, monkeypatch, GEO_CASES, **values)


def test_countries_blocked_in_the_store(tmp_path, monkeypatch, countries):
    monkeypatch.setattr(sys.modules[__name__], "STORED_COUNTRIES", ["GB"])
    answers = alike(tmp_path, monkeypatch, GEO_CASES, **{"geo_block.block_countries": ["CN"]})
    assert answers[3] == [403]


LOGGED = [[("get", "/items", {})], [("get", "/nothing-here", {})], [("get", "/wp-login.php", {})], [("post", "/form", {})],
          [("get", "/items?q=a", {})]]


def trained_on(tmp_path, native):
    """The requests aiwaf train reads from a pipeline's request log: path and status."""
    from aiwaf.fast.trainer import _read_logs

    records = _read_logs(tmp_path / f"logs-{native}")
    return sorted((str(r.get("path")).split("?")[0], str(r.get("status") or r.get("status_code"))) for r in records)


@pytest.mark.parametrize("log_format", ["combined", "json", "csv"])
def test_what_the_trainer_reads_from_the_request_log(tmp_path, monkeypatch, log_format):
    monkeypatch.setitem(LOGGING, "on", True)
    alike(tmp_path, monkeypatch, LOGGED, **{"logging_middleware.log_format": log_format})
    regular, native = trained_on(tmp_path, False), trained_on(tmp_path, True)
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(log_format, regular, native)
    assert native == regular
