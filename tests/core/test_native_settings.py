"""Converting regular-middleware settings to a native profile."""
import json
import logging

import pytest

from aiwaf.core.defaults import DEFAULT_EXEMPT_PATHS_DJANGO
from aiwaf.core.native_settings import native_config, resolve

rust = pytest.importorskip("aiwaf_rust")


def convert(settings=None, framework="django", **kwargs):
    return native_config(settings or {}, framework, environ=kwargs.pop("environ", {}), **kwargs)


def outcomes(conversion):
    return {note.setting: note.outcome for note in conversion.notes}


def assert_engine_accepts(tmp_path, conversion):
    """Every converted profile must start a real native pipeline."""
    from aiwaf.core.csv_pipeline import CSVPipeline

    config = {**conversion.config, "data_dir": str(tmp_path / "data"), "manifest_path": str(tmp_path / "none.json")}
    config.pop("store", None)  # A Django adapter key: which stores to use.
    if config.get("rate_backend") == "django_cache":  # The Django adapter supplies this counter.
        config["rate_backend"] = "memory"
    CSVPipeline(config)


def test_django_keeps_its_built_in_exemptions_and_keywords(tmp_path):
    conversion = convert({"AIWAF_EXEMPT_PATHS": ["/status/"]})
    assert conversion.config["exempt_paths"] == [*DEFAULT_EXEMPT_PATHS_DJANGO, "/status/"]
    assert "conflg" in conversion.config["static_keywords"]
    from aiwaf.core.native_settings import _engine_supports

    if _engine_supports("host_store"):
        # Django's native middleware uses the regular stores, so stored entries carry over.
        assert conversion.config["store"] == "django"
        assert "AIWAF_STORAGE_MODE" not in outcomes(conversion)
        assert outcomes(convert({"AIWAF_STORAGE_MODE": "models"}))["AIWAF_STORAGE_MODE"] == "carried"
    else:
        # An older aiwaf-rust can't use the Django stores, so it keeps native CSV files and says so.
        assert "store" not in conversion.config
        assert outcomes(conversion)["AIWAF_STORAGE_MODE"] == "changed"
    assert_engine_accepts(tmp_path, conversion)


def test_flask_reports_static_file_patterns_it_cant_exempt(tmp_path):
    from aiwaf.core.native_settings import _engine_supports

    conversion = convert(framework="flask")
    assert "/health" in conversion.config["exempt_paths"]
    assert not any("*" in path for path in conversion.config["exempt_paths"])
    note = next(n for n in conversion.notes if n.setting == "AIWAF_EXEMPT_PATHS")
    if _engine_supports("host_store"):
        # The regular Flask store matches *.css itself, so the patterns carry over.
        assert conversion.config["store"] == "flask" and note.outcome == "carried"
    else:
        assert note.outcome == "changed" and "*.css" in note.detail
    assert "AIWAF_STORAGE_MODE" not in outcomes(conversion)
    assert_engine_accepts(tmp_path, conversion)
    # A configured list replaces Flask's defaults, as in the regular middleware.
    assert convert({"AIWAF_EXEMPT_PATHS": ["/up"]}, "flask").config["exempt_paths"] == ["/up"]


@pytest.mark.parametrize("setting,value,key,expected", [
    ("AIWAF_RATE_WINDOW", "30", "rate_window", 30.0),
    ("AIWAF_RATE_MAX", 30, "rate_max", 30),
    ("AIWAF_RATE_FLOOD", 100, "rate_flood", 100),
    ("AIWAF_RATE_KEY_MODE", "ip", "rate_key_mode", "ip"),
    ("AIWAF_RATE_SOFT_BLOCK_BLACKLIST", True, "soft_block_blocklist", True),
    ("AIWAF_ENABLE_IP_BLOCKING", "false", "ip_blocking", False),
    ("AIWAF_MIN_FORM_TIME", 2, "min_form_time", 2.0),
    ("AIWAF_ENABLE_KEYWORD_LEARNING", False, "learning", False),
    ("AIWAF_DYNAMIC_TOP_N", 5, "top_n", 5),
    ("AIWAF_UUID_SCORE_BLOCK_THRESHOLD", 7, "uuid_threshold", 7),
    ("AIWAF_SQL_INJECTION_MODE", "monitor", "sql_mode", "monitor"),
    ("AIWAF_EXEMPT_KEYWORDS", "health, status", "exempt_keywords", ["health", "status"]),
    ("AIWAF_GEO_BLOCK_COUNTRIES", ["CN"], "geo_block", ["CN"]),
    ("AIWAF_REQUIRED_HEADERS", ["User-Agent", "HTTP_ACCEPT"], "required_headers", ["HTTP_USER_AGENT", "HTTP_ACCEPT"]),
    ("AIWAF_PATH_RULES", [{"PREFIX": "/api/", "DISABLE": ["honeypot"]}], "path_rules", [{"PREFIX": "/api/", "DISABLE": ["honeypot"]}]),
])
def test_settings_with_the_same_meaning_carry_over(tmp_path, setting, value, key, expected):
    conversion = convert({setting: value})
    assert conversion.config[key] == expected
    assert outcomes(conversion)[setting] == "carried"
    if key not in ("sql_mode",) or getattr(rust.NativeCSVPipeline, "supports", None):
        assert_engine_accepts(tmp_path, conversion)


def test_values_that_cant_be_read_and_unknown_settings_are_reported():
    conversion = convert({"AIWAF_RATE_MAX": "lots", "AIWAF_MODEL_PATH": "model.pkl"})
    assert "rate_max" not in conversion.config
    assert outcomes(conversion)["AIWAF_RATE_MAX"] == "not carried"
    assert outcomes(conversion)["AIWAF_MODEL_PATH"] == "not carried"
    assert "AIWAF_MODEL_PATH" in conversion.report()


def test_ip_ranges_cant_be_exempt_natively():
    conversion = convert({"AIWAF_EXEMPT_IPS": "203.0.113.5, 10.0.0.0/8, 192.168.*.*"})
    assert conversion.config["exempt_ips"] == ["203.0.113.5"]
    note = next(n for n in conversion.notes if n.setting == "AIWAF_EXEMPT_IPS")
    assert note.outcome == "changed" and "10.0.0.0/8" in note.detail and "192.168.*.*" in note.detail


def test_environment_settings_are_read_when_not_set():
    assert convert(environ={"AIWAF_SQL_INJECTION_MODE": "off"}).config["sql_mode"] == "off"
    assert convert({"AIWAF_SQL_INJECTION_MODE": "block"}, environ={"AIWAF_SQL_INJECTION_MODE": "off"}).config["sql_mode"] == "block"


def test_django_stages_come_from_the_regular_middleware_list():
    middleware = ["aiwaf.django.middleware.RateLimitMiddleware", "aiwaf.django.middleware.HeaderValidationMiddleware",
                  "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware"]
    assert convert(middleware=middleware).config["enabled"] == ["header_validation", "rate_limit"]
    with_logging = convert({"AIWAF_MIDDLEWARE_LOGGING": True}, middleware=middleware)
    assert with_logging.config["enabled"] == ["header_validation", "logging", "rate_limit"]
    assert outcomes(with_logging)["AIWAF_MIDDLEWARE_LOGGING"] == "changed"
    # After switching to the native middleware, there's no regular chain to copy.
    switched = convert(middleware=["aiwaf.django.csv_pipeline.NativeAIWAFMiddleware"])
    assert "enabled" not in switched.config and outcomes(switched)["MIDDLEWARE"] == "changed"


def test_resolve_converts_only_when_asked(caplog):
    assert resolve({"rate_max": 3}, "flask", {"AIWAF_RATE_MAX": 50}) == {"rate_max": 3}
    with caplog.at_level(logging.INFO, logger="aiwaf.native"):
        converted = resolve({"from_settings": True, "rate_max": 3}, "flask", {"AIWAF_RATE_MAX": 50, "AIWAF_RATE_FLOOD": 99})
    assert (converted["rate_max"], converted["rate_flood"]) == (3, 99)
    assert "from_settings" not in converted
    assert any("Native pipeline settings" in r.getMessage() for r in caplog.records)
    with pytest.raises(ValueError):
        native_config({}, "bottle")


def test_flask_native_pipeline_from_app_config(tmp_path, caplog):
    flask = pytest.importorskip("flask")
    from aiwaf.flask.csv_pipeline import NativeAIWAF

    app = flask.Flask(__name__)
    app.config.update(AIWAF_DATA_DIR=str(tmp_path / "data"), AIWAF_RATE_MAX=1, AIWAF_RATE_FLOOD=5,
                      AIWAF_REQUIRED_HEADERS=[], AIWAF_MIN_FORM_TIME=0, AIWAF_PATH_MANIFEST=str(tmp_path / "none.json"))
    app.add_url_rule("/hello", "hello", lambda: {"ok": True})
    with caplog.at_level(logging.INFO, logger="aiwaf.native"):
        NativeAIWAF(app, "from_settings")
    assert any("Native pipeline settings" in r.getMessage() for r in caplog.records)
    client = app.test_client()
    assert [client.get("/hello").status_code for _ in range(2)] == [200, 429]


def test_a_limit_above_the_flood_threshold_becomes_the_threshold(tmp_path):
    conversion = convert({"AIWAF_RATE_MAX": 50})
    assert (conversion.config["rate_max"], conversion.config.get("rate_flood", 40)) == (40, 40)
    assert [n.outcome for n in conversion.notes if n.setting == "AIWAF_RATE_MAX"] == ["carried"]
    assert_engine_accepts(tmp_path, conversion)


# FastAPI's AIWAFConfig

def fastapi_config(**sections):
    from aiwaf.core.runtime_config import AIWAFConfig

    config = AIWAFConfig(load_from_env=False)
    for path, value in sections.items():
        config.set(path.replace("__", "."), value)
    return config


def test_fastapi_defaults_convert_header_exemptions_to_path_rules(tmp_path):
    from aiwaf.core.native_settings import native_config_fastapi

    from aiwaf.core.native_settings import _engine_supports

    conversion = native_config_fastapi(fastapi_config(), environ={})
    assert "geo_block" not in conversion.config["enabled"] and "rate_limit" in conversion.config["enabled"]
    assert {"PREFIX": "/health", "DISABLE": ["header_validation"]} in conversion.config["path_rules"]
    assert "/health" not in conversion.config.get("exempt_paths", [])
    assert conversion.config["exempt_ips"] == ["127.0.0.1", "::1"]
    if _engine_supports("host_store"):
        # The regular exemption store loads and matches the private ranges, so nothing is lost.
        assert conversion.config["store"] == "fastapi"
        assert "exemptions.auto_exempt_patterns" not in outcomes(conversion)
        assert "storage.backend" not in outcomes(conversion)
    else:
        note = next(n for n in conversion.notes if n.setting == "exemptions.auto_exempt_patterns")
        assert note.outcome == "changed" and "10.*.*.*" in note.detail
        assert outcomes(conversion)["storage.backend"] == "changed"
    assert_engine_accepts(tmp_path, conversion)


def test_fastapi_reports_only_settings_that_differ_from_the_defaults():
    from aiwaf.core.native_settings import native_config_fastapi

    conversion = native_config_fastapi(fastapi_config(rate_limiting__max_requests=5, security__max_header_length=1000,
                                                      rate_limiting__cache_backend="redis"), environ={})
    assert conversion.config["rate_max"] == 5
    assert outcomes(conversion)["rate_limiting.max_requests"] == "carried"
    assert outcomes(conversion)["security.max_header_length"] == "not carried"
    # Redis without a URL can't be shared, so it's reported.
    assert outcomes(conversion)["rate_limiting.cache_backend"] == "not carried"
    assert "header_validation.quality_threshold" not in outcomes(conversion)


def test_fastapi_native_pipeline_from_its_config(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware

    app = FastAPI()
    for route in ("/health", "/items"):
        app.add_api_route(route, lambda: {"ok": True})
    config = fastapi_config(honeypot__min_form_time=0)
    app.add_middleware(NativeAIWAFMiddleware, config={"from_settings": True, "data_dir": str(tmp_path / "data"),
                                                      "manifest_path": str(tmp_path / "none.json")},
                       aiwaf_config=config)
    client = TestClient(app, headers={})
    client.headers.clear()
    # Header validation is off for /health only, as in the regular middleware.
    assert client.get("/health").status_code == 200
    assert client.get("/items").status_code == 403



# Rate limits are counted where the regular rate limiter counts them.

def _shared_rates():
    from aiwaf.core.native_settings import _engine_supports
    return _engine_supports("rate_counter")


def test_django_counts_rate_limits_in_its_cache(tmp_path):
    conversion = convert()
    assert conversion.config.get("rate_backend") == ("django_cache" if _shared_rates() else None)
    assert_engine_accepts(tmp_path, conversion)


@pytest.mark.parametrize("settings", [
    {"AIWAF_RATE_CACHE_BACKEND": "redis", "AIWAF_REDIS_URL": "redis://cache:6379/1"},
    {"AIWAF_RATE_CACHE_BACKEND": "redis"},
    {"AIWAF_RATE_CACHE_BACKEND": "csv"},
    {"AIWAF_RATE_CACHE_BACKEND": "memory"},
])
def test_flask_rate_cache_backends(tmp_path, settings):
    conversion = convert(settings, "flask")
    backend, url = settings["AIWAF_RATE_CACHE_BACKEND"], settings.get("AIWAF_REDIS_URL")
    note = next(n for n in conversion.notes if n.setting == "AIWAF_RATE_CACHE_BACKEND")
    if backend == "redis" and url and _shared_rates():
        assert conversion.config["rate_backend"] == "redis" and conversion.config["rate_redis_url"] == url
        assert note.outcome == "carried"
    elif backend == "redis":
        assert "rate_backend" not in conversion.config and note.outcome == "not carried"
        assert ("No Redis URL" if _shared_rates() else "newer aiwaf-rust") in note.detail
    else:
        assert conversion.config.get("rate_backend", "memory") == backend and note.outcome == "carried"
    if conversion.config.get("rate_backend") != "redis":
        assert_engine_accepts(tmp_path, conversion)


def test_flask_redis_rates_on_an_older_engine_are_reported(monkeypatch):
    from aiwaf.core import native_settings

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: False)
    conversion = convert({"AIWAF_RATE_CACHE_BACKEND": "redis", "AIWAF_REDIS_URL": "redis://cache"}, "flask")
    assert "newer aiwaf-rust" in next(n.detail for n in conversion.notes if n.setting == "AIWAF_RATE_CACHE_BACKEND")
    assert "rate_backend" not in convert({}, "django").config


@pytest.mark.parametrize("backend", ["redis", "csv"])
def test_fastapi_rate_cache_backends(backend):
    from aiwaf.core.native_settings import native_config_fastapi

    conversion = native_config_fastapi(fastapi_config(rate_limiting__cache_backend=backend,
                                                      rate_limiting__redis_url="redis://cache:6379/2"), environ={})
    if backend == "csv" or _shared_rates():
        assert conversion.config["rate_backend"] == backend
        assert outcomes(conversion)["rate_limiting.cache_backend"] == "carried"
    if backend == "redis" and _shared_rates():
        assert conversion.config["rate_redis_url"] == "redis://cache:6379/2"


# The native pipeline as the default

def test_native_by_default_needs_every_feature_no_difference_and_no_opt_out(monkeypatch):
    from aiwaf.core import native_settings
    from aiwaf.core.native_settings import NativeConversion, native_by_default

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: True)
    clean = NativeConversion(config={})
    assert native_by_default(clean, environ={})
    for choice in (False, "false", "0", "OFF", "no"):
        assert not native_by_default(clean, choice=choice, environ={})
    assert not native_by_default(clean, environ={"AIWAF_NATIVE_PIPELINE": "false"})
    # The app's own choice wins over the environment.
    assert native_by_default(clean, choice=True, environ={"AIWAF_NATIVE_PIPELINE": "false"})
    differs = NativeConversion(config={})
    differs.note("AIWAF_X", "not carried", "")
    assert not native_by_default(differs, environ={})
    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: feature != "rate_counter")
    assert not native_by_default(clean, environ={})


def test_only_aiwaf_rust_models_carry_over(monkeypatch):
    import sys
    from aiwaf.core.native_settings import NativeConversion, carry_model, is_rust_model

    forest = rust.IsolationForest(n_estimators=5, random_state=1)
    conversion = NativeConversion(config={})
    assert carry_model(conversion, None, "AIWAF_MODEL_PATH") is None
    assert carry_model(conversion, forest, "AIWAF_MODEL_PATH") is forest
    assert not conversion.differences()
    assert carry_model(conversion, object(), "AIWAF_MODEL_PATH") is None
    assert [n.setting for n in conversion.differences()] == ["AIWAF_MODEL_PATH"]
    monkeypatch.setitem(sys.modules, "aiwaf_rust", None)
    assert not is_rust_model(forest)


def test_fastapi_logs_natively_where_its_trainer_reads():
    from aiwaf.core.native_settings import native_config_fastapi

    conversion = native_config_fastapi(fastapi_config(logging_middleware__enabled=True,
                                                      logging_middleware__log_dir="/var/log/app"), environ={})
    assert conversion.config["log_path"] == "/var/log/app/aiwaf_native_access.csv"
    assert outcomes(conversion)["logging_middleware"] == "carried"



@pytest.mark.parametrize("value, store, expected", [
    (True, True, "carried"), (False, True, "carried"), ("yes", True, "carried"),
    (True, False, "carried"), (False, False, "changed"), ("maybe", False, "carried"),
])
def test_flask_use_csv_follows_where_flask_keeps_its_stores(monkeypatch, value, store, expected):
    from aiwaf.core import native_settings

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: store)
    conversion = convert({"AIWAF_USE_CSV": value}, "flask")
    assert outcomes(conversion)["AIWAF_USE_CSV"] == expected
    assert ("store" in conversion.config) == store
    # Django has no such setting, so it isn't reported there as unknown.
    assert "AIWAF_USE_CSV" not in outcomes(convert({}, "flask"))


# Whenever the converter reports no differences, the native pipeline must start: the automatic
# switch relies on it. Edge values the regular middleware tolerates are tried one by one and in
# random combinations.

_EDGE = {float: [0, 0.5, -1, 1e9, "10", "abc", 3], int: [0, -1, 10 ** 7, "5", 2.5, True],
         str: ["BLOCK", "Monitor", "observe", "", "IP", "ip", "off"], bool: [0, 1, "yes", "maybe"],
         list: ["", "a, b", [], ["*.css"], [123], "X-Custom"]}
_SPECIAL = {
    "AIWAF_PATH_RULES": [[{"PREFIX": "/", "RATE_LIMIT": {"WINDOW": 0}}], [{"PREFIX": "/", "RATE_LIMIT": {"MAX": -1}}],
                         [{"PREFIX": "/", "RATE_LIMIT": {"MAX": "5"}}], [{"PREFIX": "/", "RATE_LIMIT": {"MAX": 100}}],
                         [{"PREFIX": "/", "RATE_LIMIT": {"FLOOD": 10 ** 6}}], ["not a dict"]],
    "AIWAF_INJECTION_RISK_MODE": ["LOG", "warn"],
}


def _native_starts(tmp_path, conversion):
    from aiwaf.core.csv_pipeline import CSVPipeline

    config = dict(conversion.config)
    config.pop("store", None)
    if config.get("rate_backend") == "django_cache":
        config["rate_backend"] = "memory"
    config["data_dir"] = str(tmp_path / f"d{len(list(tmp_path.iterdir()))}")
    config.setdefault("manifest_path", str(tmp_path / "none.json"))
    CSVPipeline(config)


@pytest.mark.parametrize("framework", ["django", "flask"])
def test_settings_without_differences_always_start_natively(tmp_path, monkeypatch, framework):
    import random
    from aiwaf.core import native_settings

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: True)
    cases = [{name: value} for name, (key, kind) in native_settings._DIRECT.items()
             for value in _SPECIAL.get(name, _EDGE[kind])]
    names, rng = list(native_settings._DIRECT.items()), random.Random(7)
    cases += [{name: rng.choice(_SPECIAL.get(name, _EDGE[kind]) + [1, 10, 100]) for name, (key, kind) in rng.sample(names, 4)}
              for _ in range(300)]
    for settings in cases:
        conversion = convert(settings, framework)
        if not conversion.differences():
            try:
                _native_starts(tmp_path, conversion)
            except Exception as error:  # pragma: no cover - the failure message names the case
                pytest.fail(f"{settings} has no differences but native failed: {error}")


def test_values_native_cant_take_are_reported_and_normalized_ones_carry():
    conversion = convert({"AIWAF_SQL_INJECTION_MODE": "BLOCK", "AIWAF_RATE_KEY_MODE": "IP_ONLY",
                          "AIWAF_INJECTION_RISK_MODE": "LOG", "AIWAF_RATE_WINDOW": 0.5, "AIWAF_PAYLOAD_MAX_BYTES": 0,
                          "AIWAF_PATH_RULES": [{"PREFIX": "/a", "RATE_LIMIT": {"WINDOW": 0}, "DISABLE": ["geo_block"]},
                                               "nope"]}, "flask")
    assert conversion.config["sql_mode"] == "block" and conversion.config["rate_key_mode"] == "ip"
    assert conversion.config["injection_risk_mode"] == "log"
    assert "rate_window" not in conversion.config and "max_bytes" not in conversion.config
    assert {"PREFIX": "/a", "DISABLE": ["geo_block"]} in conversion.config["path_rules"]
    reported = {n.setting for n in conversion.differences()}
    assert {"AIWAF_RATE_WINDOW", "AIWAF_PAYLOAD_MAX_BYTES", "AIWAF_PATH_RULES"} <= reported
    assert {"AIWAF_SQL_INJECTION_MODE", "AIWAF_RATE_KEY_MODE"}.isdisjoint(reported)
    environ = convert({}, "flask", environ={"AIWAF_SQL_INJECTION_MODE": "observe"})
    assert "AIWAF_SQL_INJECTION_MODE" in {n.setting for n in environ.differences()}


def test_fastapi_settings_without_differences_always_start_natively(tmp_path, monkeypatch):
    import random
    from aiwaf.core import native_settings
    from aiwaf.core.native_settings import native_config_fastapi

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: True)
    rng, names = random.Random(11), list(native_settings._FASTAPI_DIRECT.items())
    for _ in range(300):
        chosen = {path.replace(".", "__"): rng.choice(_EDGE[kind] + [1, 10, 100]) for path, (key, kind) in rng.sample(names, 3)}
        conversion = native_config_fastapi(fastapi_config(**chosen), environ={})
        if not conversion.differences():
            try:
                _native_starts(tmp_path, conversion)
            except Exception as error:  # pragma: no cover
                pytest.fail(f"{chosen} has no differences but native failed: {error}")
    sql = native_config_fastapi(fastapi_config(ip_keyword_block__sql_injection_mode="Monitor"), environ={})
    assert sql.config["sql_mode"] == "monitor"


def test_native_by_default_needs_a_data_directory_it_can_write(tmp_path, monkeypatch):
    from aiwaf.core import native_settings
    from aiwaf.core.native_settings import NativeConversion, native_by_default

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: True)
    ok = NativeConversion(config={"data_dir": str(tmp_path / "data"), "log_path": str(tmp_path / "logs" / "a.csv")})
    assert native_by_default(ok, environ={}) and (tmp_path / "logs").is_dir()
    blocked = NativeConversion(config={"data_dir": "/proc/aiwaf-cant-create"})
    assert not native_by_default(blocked, environ={})
    assert [n.setting for n in blocked.differences()] == ["data_dir"]
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    monkeypatch.setattr(native_settings.os, "access", lambda path, mode: False)
    assert not native_by_default(NativeConversion(config={"data_dir": str(read_only)}), environ={})
