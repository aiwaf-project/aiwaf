"""Differential policy, persistence and concurrency checks for the native chain."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import pytest
from aiwaf.core import header_validation
from aiwaf.core.sql_injection import SQLInjectionPolicy

rust = pytest.importorskip("aiwaf_rust")
pytestmark = pytest.mark.skipif(not hasattr(rust, "NativeCSVPipeline"), reason="new native pipeline required")


def engine(tmp_path, **config):
    return rust.NativeCSVPipeline(json.dumps({"data_dir": str(tmp_path), "enabled": [], "exempt_paths": [], **config}))


def begin(e, *, ip="8.8.8.8", path="/missing", query=b"", body=b"", now=100.0,
          method="GET", headers=None, country="", exists=False, disabled=(), exempt=False,
          authenticated=False, methods=("GET", "POST"), exhausted=False):
    headers = {**(headers or {}), "AIWAF_COUNTRY": country}
    return e.begin(ip, path, method, headers, query, body, exists, "GET" in methods, "POST" in methods,
                   method in methods, authenticated, set(disabled), exempt, now, "2026-10-12T00:00:00+00:00", exhausted)


def rows(tmp_path, file):
    with (tmp_path / file).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("query,body", [
    (b"q=hello", b""), (b"q=UNION%20SELECT", b""), (b"q=%2527%2520or%25201%253D1", b""),
    (b"q=%FF", b""), (b"", b'{"input":"uni\\u006fn select"}'),
    (b"", '{"input":"你好 UNION SELECT"}'.encode()),
    (b"", '{"input":"UNION SELECT"}'.encode("utf-16")),
    (b"", b'{"union select":null}'), (b"", b'[null,12,false,"hello"]'),
    (b"", b'{"a":"[\\"union select\\"]"}'), (b"", b'bad json'),
    (b"q=union/**/select", b""), (b"q=%27--", b""),
    (b"q=hello", b'{"input":";drop table users"}'),
    (b"", b'{"number":NaN,"input":"uni\\u006fn select"}'),
    (b"", b'{"number":Infinity,"input":"uni\\u006fn select"}'),
    (b"", b'{"number":-Infinity,"input":"uni\\u006fn select"}'),
    (b"", b'{"surrogate":"\\ud800","input":"uni\\u006fn select"}'),
    (b"", b'{"surrogate":"\\udc00","input":"uni\\u006fn select"}'),
    (b"", b'{"surrogate":"\\ud83d\\ude00","input":"uni\\u006fn select"}'),
    (b"", b'{"literal":"NaN \\\\ud800 \\\"abc\\\"","input":"uni\\u006fn select"}'),
    (b"", b'{"z":"uni\\u006fn select","a":"\\u0027--"}'),
    (b"", b'{"a":"uni\\u006fn select","z":"\\u0027--"}'),
    (b"", b'{"a":"uni\\u006fn select","a":"hello"}'),
    (b"", b'{"number":1e999,"input":"uni\\u006fn select"}'),
    (b"", b'{"number":' + b'9' * 500 + b',"input":"uni\\u006fn select"}'),
])
def test_sql_matches_existing_inspector(tmp_path, query, body):
    e = engine(tmp_path, enabled=["ip_keyword_block"], learning=False)
    ctx, status, payload = begin(e, query=query, body=body, exists=True)
    expected = SQLInjectionPolicy().inspect(query, body)
    assert status == (expected.status if expected else 0)
    if expected:
        assert json.loads(payload)["error"] == expected.rule
    e.finish(ctx, status or 200, 100.1, "0")


def test_budget_and_no_keyword_write_on_sql_block(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block"], max_bytes=30)
    assert begin(e, body=b"x" * 31)[1] == 413
    assert begin(e, path="/../shell/shell.php", query=b"q=1 union select 1")[1] == 403
    assert rows(tmp_path, "keywords.csv") == []
    assert begin(e, exhausted=True)[1] == 413


def test_nested_structure_limit(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block"])
    assert begin(e, body=(b"[" * 18 + b'"hello"' + b"]" * 18))[1] == 413
    assert begin(e, body=json.dumps([None] * 4100).encode())[1] == 413
    assert begin(e, body=b'[' * 200 + b'null' + b']' * 200)[1] == 413


@pytest.mark.parametrize('encoding', ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16-le', 'utf-16-be', 'utf-32', 'utf-32-le', 'utf-32-be'])
@pytest.mark.parametrize('budget', [30, 64, 256])
def test_encoding_and_budget_parity(tmp_path, encoding, budget):
    e = engine(tmp_path, enabled=['ip_keyword_block'], max_bytes=budget, learning=False)
    for text in ('{"field":"hello"}', '{"field":"uni\\u006fn select"}'):
        body = text.encode(encoding)
        _, status, payload = begin(e, body=body, exists=True)
        expected = SQLInjectionPolicy(max_bytes=budget).inspect(b'', body)
        assert status == (expected.status if expected else 0)
        if expected:
            assert json.loads(payload)['error'] == expected.rule


def test_anomaly_model_history_and_response_learning(tmp_path):
    from aiwaf.core.anomaly import evaluate_anomaly
    keywords = ['.php', 'wp-', 'admin', 'shell']
    model = rust.IsolationForest(n_estimators=20, random_state=42)
    model.fit([[10+i%10, i%2, .01+i%4/1000, i%3, 1+i%4, i%2] for i in range(128)])
    e = engine(tmp_path, enabled=['ai_anomaly'], anomaly_keywords=keywords)
    e.set_model(model)
    path = '/wp-content/admin/shell.php'
    history = [(100-i*.1, path, 404, .1) for i in range(30)]
    e.seed_history('8.8.8.8', [(t, p, s, rt, True) for t, p, s, rt in history])
    expected = evaluate_anomaly(ip='8.8.8.8', path=path, status_code=404, response_time=.1, now=101.,
        history=history, window_seconds=60., model=model, static_keywords=keywords, malicious_keywords=keywords,
        keyword_learning_enabled=False, path_exists=lambda _:False, is_exempt_path=lambda _:False,
        is_malicious_context=lambda _:False)
    ctx, status, _ = begin(e, path=path, now=100.9)
    assert status == 0
    assert (e.finish(ctx, 404, 101., '0')[0] == 403) == expected.block
    e.set_model(None)
    assert e.diagnostics()[0] is False
    ctx, _, _ = begin(e, ip='9.9.9.9', path='/config/freshkeyword', now=101.)
    e.finish(ctx, 404, 101.1, '0')
    assert 'freshkeyword' in {r['keyword'] for r in rows(tmp_path, 'keywords.csv')}


def test_keyword_learning_and_reputation_files(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block"])
    assert begin(e, path="/../shell/shell.php")[1] == 403
    assert "shell" in {row["keyword"] for row in rows(tmp_path, "keywords.csv")}
    blocked = rows(tmp_path, "blacklist.csv")
    assert blocked[0]["ip"] == "8.8.8.8"
    assert int(blocked[0]["offenses"]) == 1
    assert begin(e, path="/known", exists=True)[1] == 403
    assert begin(e, path="/known", exists=True, now=10000)[1] == 0


@pytest.mark.parametrize("backend", ["memory", "csv"])
def test_concurrent_rate_consumption_is_atomic(tmp_path, backend):
    e = engine(tmp_path, enabled=["rate_limit"], rate_backend=backend, rate_max=5, rate_flood=100)
    def consume(_):
        ctx, status, _ = begin(e)
        e.finish(ctx, status or 200, 100.1, "0")
        return status
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(consume, range(30)))
    assert statuses.count(0) == 5
    assert statuses.count(429) == 25
    if backend == "csv":
        second = engine(tmp_path, enabled=["rate_limit"], rate_backend="csv", rate_max=5, rate_flood=100)
        assert begin(second)[1] == 429
    assert begin(e, now=111)[1] == 0


def test_rate_flood_and_soft_blocklist(tmp_path):
    e = engine(tmp_path, enabled=["rate_limit"], rate_max=1, rate_flood=2)
    assert [begin(e)[1] for _ in range(3)] == [0, 429, 403]
    assert rows(tmp_path, "blacklist.csv")[0]["reason"] == "Flood pattern"


def test_exemption_and_disabled_stages(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block", "header_validation"], exempt_paths=["/safe"])
    assert begin(e, path="/safe/child", query=b"q=1 union select 1")[1] == 0
    assert begin(e, query=b"q=1 union select 1", disabled=["ip_keyword_block", "header_validation"])[1] == 0
    rust.csv_write(str(tmp_path / "whitelist.csv"), ["ip", "added_date"], [["8.8.8.8", "today"]], False)
    assert begin(e, query=b"q=1 union select 1")[1] == 0


def test_geo_first_and_header_policy_parity(tmp_path):
    e = engine(tmp_path, enabled=["geo_block", "ip_keyword_block", "header_validation"], geo_block=["cn"])
    ctx, status, _ = begin(e, country="CN", query=b"q=1 union select 1")
    assert status == 403
    assert rows(tmp_path, "blacklist.csv")[0]["reason"] == "Geo blocked: CN"
    h = engine(tmp_path / "h", enabled=["header_validation"])
    for headers in ({}, {"HTTP_USER_AGENT":"curl/8.0", "HTTP_ACCEPT":"*/*"},
                    {"HTTP_USER_AGENT":"Mozilla/5.0", "HTTP_ACCEPT":"text/html", "HTTP_ACCEPT_LANGUAGE":"en-US"}):
        expected = header_validation.evaluate_header_policy(headers, min_score=3)
        assert bool(begin(h, headers=headers, ip=str(len(headers)))[1]) == bool(expected)


@pytest.mark.parametrize('backend', ['csv', 'memory'])
def test_honeypot_expiry_auth_and_method(tmp_path, backend):
    e = engine(tmp_path, enabled=["honeypot"], honeypot_backend=backend)
    assert begin(e, now=100)[1] == 0
    assert begin(e, method="POST", now=100.05)[1] == 403
    assert begin(e, method="POST", now=102)[1] == 0
    assert begin(e, method="POST", now=341)[1] == 409
    assert begin(e, method="POST", now=342)[1] == 0
    assert begin(e, authenticated=True, method="POST", methods=["GET"])[1] == 0
    assert begin(e, method="POST", methods=["GET"])[1] == 405


@pytest.mark.parametrize('backend', ['csv', 'memory'])
def test_honeypot_ttl_and_worker_state(tmp_path, backend):
    e = engine(tmp_path, enabled=['honeypot'], honeypot_backend=backend)
    assert begin(e, now=100)[1] == 0
    second = engine(tmp_path, enabled=['honeypot'], honeypot_backend=backend)
    assert begin(second, method='POST', now=100.05)[1] == (403 if backend == 'csv' else 0)
    assert begin(e, method='POST', now=400)[1] == 409  # inclusive TTL edge
    assert begin(e, now=500)[1] == 0
    assert begin(e, method='POST', now=801)[1] == 0  # expired timestamp
    if backend == 'memory':
        assert not (tmp_path / 'pipeline_state.csv').exists()


def test_memory_honeypot_concurrent_refresh_and_expiry(tmp_path):
    e = engine(tmp_path, enabled=['honeypot'], honeypot_backend='memory')
    begin(e, now=100)
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: begin(e, method='POST', now=341)[1], range(32)))
    assert statuses.count(409) == 1
    assert statuses.count(0) == 31
    # A later GET survives expiration of the old timestamp and governs POST.
    begin(e, now=342)
    assert begin(e, method='POST', now=342.05)[1] == 403


def test_uuid_response_scoring_and_context_ownership(tmp_path):
    e = engine(tmp_path, enabled=["uuid_tamper"], uuid_threshold=2)
    assert begin(e, query=b"uuid=invalid")[1] == 403
    query = b"uuid=12345678-1234-1234-1234-123456789abc"
    c, _, _ = begin(e, ip="9.9.9.9", query=query)
    assert e.finish(c, 404, 100.1, "0")[0] == 404
    with pytest.raises(RuntimeError, match="already finalized"):
        e.finish(c, 404, 100.1, "0")
    c2, _, _ = begin(e, ip="9.9.9.9", query=query)
    assert e.finish(c2, 404, 100.1, "0")[0] == 403
    other = engine(tmp_path / "other")
    c3, _, _ = begin(e)
    with pytest.raises(ValueError, match="context"):
        other.finish(c3, 200, 100.1, "0")


def test_native_logging_and_missing_capability(tmp_path, monkeypatch):
    e = engine(tmp_path, enabled=["logging"])
    c, _, _ = begin(e, path="/CaseSensitive", headers={"HTTP_USER_AGENT":"browser"})
    e.finish(c, 200, 100.123, "42")
    log = rows(tmp_path, "requests.csv")
    assert log[0]["path"] == "/CaseSensitive"
    assert log[0]["content_length"] == "42"
    assert log[0]["status_code"] == "200"
    assert log[0]["blocked"] == "False"
    from aiwaf.core.csv_pipeline import CSVPipeline
    monkeypatch.delattr(rust, "NativeCSVPipeline")
    with pytest.raises(RuntimeError, match="requires aiwaf-rust"):
        CSVPipeline({"data_dir": str(tmp_path)})


def test_callback_values_cannot_silently_exempt_a_request():
    from aiwaf.core.csv_pipeline import validate_fact_callback, fact_value
    async def async_value(_):
        return True
    with pytest.raises(TypeError, match='synchronous'):
        validate_fact_callback(async_value)
    with pytest.raises(TypeError, match='synchronous'):
        fact_value(lambda value: async_value(value), 'request')
    validate_fact_callback(lambda _:False)
    assert fact_value(lambda _:False, 'request') is False


def test_model_shape_and_existing_log_schema_fail_at_startup(tmp_path):
    e = engine(tmp_path)
    with pytest.raises(ValueError, match='six request features'):
        e.set_model(rust.IsolationForest())
    rust.csv_write(str(tmp_path / 'requests.csv'), ['wrong'], [], False)
    with pytest.raises(ValueError, match='log CSV schema'):
        engine(tmp_path, enabled=['logging'])


def test_clock_adjustment_does_not_turn_success_into_500(tmp_path):
    e = engine(tmp_path, enabled=['logging', 'ai_anomaly'])
    ctx, _, _ = begin(e, now=100.)
    assert e.finish(ctx, 200, 99.9, '0', True)[0] == 200
    assert float(rows(tmp_path, 'requests.csv')[0]['response_time_ms']) >= 0


@pytest.mark.parametrize("config", [
    {"unknown_key":True}, {"sql_mode":"loud"}, {"rate_backend":"redis"},
    {"enabled":["misspelling"]}, {"rate_window":0}, {"max_bytes":0},
    {"rate_max":30, "rate_flood":20}, {"exempt_paths":["/foo/*"]},
    {"honeypot_backend":"redis"},
])
def test_unsupported_profiles_fail_at_startup(tmp_path, config):
    with pytest.raises(ValueError):
        engine(tmp_path, **config)


# Path rules and route checks: the adapter works out per-request facts; the engine applies them.
needs_route_facts = pytest.mark.skipif(not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None),
                                       reason="aiwaf-rust without route_rule and rate overrides")

@needs_route_facts
def test_route_rule_blocks_in_the_keyword_stage_whatever_the_sql_mode(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block", "logging"], learning=False, sql_mode="off")
    ctx, status, payload = e.begin("8.8.8.8", "/run", "GET", {"AIWAF_COUNTRY": ""}, b"", b"", True, True, True,
                                   True, False, set(), False, 100.0, "2026-10-12T00:00:00+00:00", False,
                                   "command_chain")
    assert (status, json.loads(payload)) == (403, {"error": "command_chain"})
    e.finish(ctx, status, 100.1, "0")
    assert rows(tmp_path, "requests.csv")[-1]["block_reason"] == "command_chain"
    # A route without a finding, or with the keyword stage turned off, isn't blocked.
    ctx, status, _ = e.begin("8.8.8.8", "/run", "GET", {"AIWAF_COUNTRY": ""}, b"", b"", True, True, True,
                             True, False, {"ip_keyword_block"}, False, 101.0, "2026-10-12T00:00:00+00:00", False,
                             "command_chain")
    assert status == 0
    e.finish(ctx, 200, 101.1, "0")


@needs_route_facts
def test_rate_overrides_replace_the_profile_limits_for_one_request(tmp_path):
    e = engine(tmp_path, enabled=["rate_limit"], rate_max=20, rate_flood=40)

    def hit(now, **limits):
        ctx, status, _ = e.begin("8.8.8.8", "/api", "GET", {"AIWAF_COUNTRY": ""}, b"", b"", True, True, True,
                                 True, False, set(), False, now, "2026-10-12T00:00:00+00:00", False, None,
                                 limits.get("window"), limits.get("max"), limits.get("flood"))
        e.finish(ctx, status or 200, now, "0")
        return status

    assert [hit(100.0 + i, max=2, flood=10) for i in range(3)] == [0, 0, 429]
    with pytest.raises(ValueError):
        hit(110.0, max=5, flood=2)
    with pytest.raises(ValueError):
        hit(110.0, window=0.5)


@needs_route_facts
def test_long_override_windows_survive_state_cleanup(tmp_path):
    e = engine(tmp_path, enabled=["rate_limit"], rate_window=10, rate_max=20, rate_flood=40)

    def hit(now):
        ctx, status, _ = e.begin("8.8.8.8", "/slow", "GET", {"AIWAF_COUNTRY": ""}, b"", b"", True, True, True,
                                 True, False, set(), False, now, "2026-10-12T00:00:00+00:00", False, None,
                                 300.0, 2, 10)
        e.finish(ctx, status or 200, now, "0")
        return status

    # Cleanup runs every 30 seconds. It must keep entries the 300-second window still counts.
    assert [hit(100.0), hit(101.0), hit(160.0)] == [0, 0, 429]


def _adapter(tmp_path, stores=None, **config):
    from aiwaf.core.csv_pipeline import CSVPipeline

    return CSVPipeline({"data_dir": str(tmp_path / "data"), "exempt_paths": [], "learning": False,
                        "manifest_path": str(tmp_path / "missing.json"), **config}, stores=stores)


def _request(pipeline, path, *, query=b"", headers=None):
    base = {"HTTP_USER_AGENT": "Mozilla/5.0 Chrome/130 Safari/537.36", "HTTP_ACCEPT": "text/html",
            "HTTP_ACCEPT_LANGUAGE": "en", "HTTP_ACCEPT_ENCODING": "gzip", "HTTP_CONNECTION": "keep-alive"}
    ctx, status, payload = pipeline.begin(ip="8.8.8.8", path=path, method="GET",
                                          headers=base if headers is None else headers, query=query, body=b"",
                                          exists=True, methods={"GET"})
    pipeline.finish(ctx, status or 200)
    return status, payload


def test_adapter_path_rules_disable_stages_for_a_prefix(tmp_path):
    pipeline = _adapter(tmp_path, enabled=["header_validation"],
                        path_rules=[{"PREFIX": "/health/", "DISABLE": ["header_validation"]}])
    assert _request(pipeline, "/health/live", headers={})[0] == 0
    assert _request(pipeline, "/other", headers={})[0] == 403


@needs_route_facts
def test_adapter_path_rules_override_rate_limits(tmp_path):
    pipeline = _adapter(tmp_path, enabled=["rate_limit"],
                        path_rules=[{"PREFIX": "/api/", "RATE_LIMIT": {"MAX": 1, "FLOOD": 5}}])
    assert [_request(pipeline, "/api/items")[0] for _ in range(2)] == [0, 429]
    assert [_request(pipeline, "/home")[0] for _ in range(2)] == [0, 0]
    with pytest.raises(ValueError):
        _adapter(tmp_path, enabled=["rate_limit"], path_rules=[{"PREFIX": "/x/", "RATE_LIMIT": {"MAX": 9, "FLOOD": 3}}])
    with pytest.raises(ValueError):
        _adapter(tmp_path, path_rules=[{"PREFIX": "/x/", "RATE_LIMIT": {"MAX": "many"}}])


@needs_route_facts
def test_adapter_runs_route_checks_from_the_manifest(tmp_path, caplog):
    import logging
    from aiwaf.core import route_risks

    route_risks.clear_cache()
    manifest = tmp_path / "paths.json"
    manifest.write_text(json.dumps({"routes": {"/files": {"view": "app.files", "injection_risks": [
        {"sink": "path", "input": "name", "confidence": "high"}]}}}), encoding="utf-8")
    blocking = _adapter(tmp_path, enabled=["ip_keyword_block"], manifest_path=str(manifest))
    status, payload = _request(blocking, "/files", query=b"name=../../etc/passwd")
    assert (status, json.loads(payload)) == (403, {"error": "path_traversal"})
    assert _request(blocking, "/files", query=b"name=report.pdf")[0] == 0
    with caplog.at_level(logging.WARNING, logger="aiwaf.injection_risk"):
        logging_only = _adapter(tmp_path, enabled=["ip_keyword_block"], manifest_path=str(manifest),
                                injection_risk_mode="log")
        assert _request(logging_only, "/files", query=b"name=../../etc/passwd")[0] == 0
    assert any(json.loads(r.getMessage())["rule"] == "path_traversal" for r in caplog.records)
    # Turned off, or with the keyword stage off, the route check doesn't run.
    assert _request(_adapter(tmp_path, enabled=["ip_keyword_block"], manifest_path=str(manifest),
                             injection_risk_mode="off"), "/files", query=b"name=../x")[0] == 0
    assert _request(_adapter(tmp_path, enabled=["header_validation"], manifest_path=str(manifest)),
                    "/files", query=b"name=../x")[0] == 0
    with pytest.raises(ValueError):
        _adapter(tmp_path, injection_risk_mode="loud")
    route_risks.clear_cache()


class _OlderEngine:
    """Stands in for a published aiwaf-rust without route_rule or rate overrides."""
    calls = []

    def __init__(self, profile_json, *cache):
        pass

    def begin(self, *args):
        _OlderEngine.calls.append(len(args))
        return object(), 0, "null"

    def finish(self, ctx, status, now, length, monotonic):
        return status, "null"


def _with_older_engine(monkeypatch):
    import types
    from aiwaf.core import csv_pipeline as module

    _OlderEngine.calls = []
    monkeypatch.setattr(module.importlib, "import_module",
                        lambda name: types.SimpleNamespace(NativeCSVPipeline=_OlderEngine))


def test_adapter_keeps_working_with_older_published_engines(tmp_path, monkeypatch):
    _with_older_engine(monkeypatch)
    pipeline = _adapter(tmp_path, enabled=["header_validation"],
                        path_rules=[{"PREFIX": "/health/", "DISABLE": ["header_validation"]}])
    assert _request(pipeline, "/health/live")[0] == 0
    assert _OlderEngine.calls == [16]


def test_adapter_refuses_features_an_older_engine_lacks(tmp_path, monkeypatch):
    from aiwaf.core import route_risks

    _with_older_engine(monkeypatch)
    with pytest.raises(RuntimeError, match="rate_overrides"):
        _adapter(tmp_path, path_rules=[{"PREFIX": "/api/", "RATE_LIMIT": {"MAX": 1}}])
    route_risks.clear_cache()
    manifest = tmp_path / "paths.json"
    manifest.write_text(json.dumps({"routes": {"/files": {"injection_risks": [{"sink": "path", "input": ""}]}}}),
                        encoding="utf-8")
    with pytest.raises(RuntimeError, match="route_rule"):
        _adapter(tmp_path, manifest_path=str(manifest))
    # Turning route checks off doesn't need the newer engine.
    _adapter(tmp_path, manifest_path=str(manifest), injection_risk_mode="off")
    route_risks.clear_cache()


@needs_route_facts
def test_engine_reports_its_optional_facts():
    assert rust.NativeCSVPipeline.supports("route_rule") and rust.NativeCSVPipeline.supports("rate_overrides")
    assert not rust.NativeCSVPipeline.supports("something_else")


needs_sql_monitor = pytest.mark.skipif(
    not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None)
    or not rust.NativeCSVPipeline.supports("sql_monitor"), reason="aiwaf-rust without SQL monitor mode")


@needs_sql_monitor
@pytest.mark.parametrize("query,headers,path,exhausted,rule", [
    (b"q=%27%20OR%201%3D1--", {}, "/search", False, "sql_breakout_single_quote"),
    (b"", {"HTTP_COOKIE": "session=' OR 1=1--"}, "/search", False, "sql_breakout_single_quote"),
    (b"", {}, "/post/1 OR 1=1", False, "sql_breakout_numeric"),
    (b"", {}, "/upload", True, "payload_inspection_limit"),
])
def test_monitor_mode_reports_sql_findings_without_blocking(tmp_path, query, headers, path, exhausted, rule):
    e = engine(tmp_path, enabled=["ip_keyword_block", "logging"], learning=False, sql_mode="monitor")
    ctx, status, payload = begin(e, query=query, headers=headers, path=path, exists=True, exhausted=exhausted)
    assert (status, payload, ctx.sql_monitor) == (0, "null", rule)
    e.finish(ctx, 200, 100.1, "0")
    assert rows(tmp_path, "requests.csv")[-1]["blocked"] == "False"
    clean, status, _ = begin(e, path="/search", exists=True)
    assert (status, clean.sql_monitor) == (0, None)
    e.finish(clean, 200, 100.2, "0")
    # The same request is blocked in block mode, so monitor mode changes only the outcome.
    blocking = engine(tmp_path / "block", enabled=["ip_keyword_block"], learning=False)
    assert begin(blocking, query=query, headers=headers, path=path, exists=True, exhausted=exhausted)[1] != 0


@needs_sql_monitor
def test_adapter_logs_monitored_sql_findings_as_the_regular_chain_does(tmp_path, caplog):
    import logging

    pipeline = _adapter(tmp_path, enabled=["ip_keyword_block"], sql_mode="monitor")
    with caplog.at_level(logging.WARNING, logger="aiwaf.payload"):
        assert _request(pipeline, "/search", query=b"q=%27%20OR%201%3D1--")[0] == 0
    assert [r.getMessage() for r in caplog.records if r.name == "aiwaf.payload"] == [
        "aiwaf.payload rule=sql_breakout_single_quote mode=monitor"]


def test_adapter_refuses_monitor_mode_on_an_older_engine(tmp_path, monkeypatch):
    _with_older_engine(monkeypatch)
    with pytest.raises(RuntimeError, match="sql_monitor"):
        _adapter(tmp_path, sql_mode="monitor")


needs_uuid_fact = pytest.mark.skipif(
    not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None)
    or not rust.NativeCSVPipeline.supports("uuid_fact"), reason="aiwaf-rust without the route UUID fact")


@needs_uuid_fact
def test_a_route_uuid_is_scored_like_the_query_parameter(tmp_path):
    e = engine(tmp_path, enabled=["uuid_tamper"], uuid_threshold=2)

    def hit(uuid, status, now, ip="203.0.113.90"):
        ctx, blocked, _ = e.begin(ip, "/items/x", "GET", {"AIWAF_COUNTRY": ""}, b"", b"", True, True, True, True,
                                  False, set(), False, now, "2026-10-12T00:00:00+00:00", False, None, None, None,
                                  None, uuid=uuid)
        return blocked or e.finish(ctx, status, now, "0")[0]

    assert hit("not-a-uuid", 200, 100.0) == 403
    valid = "550e8400-e29b-41d4-a716-446655440000"
    # Repeated 404s for well-formed UUIDs build the score until the threshold blocks.
    assert [hit(valid, 404, 101.0 + i, ip="203.0.113.91") for i in range(2)] == [404, 403]
    # The route's UUID wins over ?uuid=.
    ctx, status, _ = e.begin("203.0.113.92", "/items/x", "GET", {"AIWAF_COUNTRY": ""}, b"uuid=not-a-uuid", b"", True,
                             True, True, True, False, set(), False, 110.0, "2026-10-12T00:00:00+00:00", False, None,
                             None, None, None, uuid=valid)
    assert status == 0
    e.finish(ctx, 200, 110.0, "0")


def test_route_uuids_on_an_older_engine_are_skipped_with_one_warning(tmp_path, monkeypatch, caplog):
    import logging

    _with_older_engine(monkeypatch)
    pipeline = _adapter(tmp_path, enabled=["uuid_tamper"])
    with caplog.at_level(logging.WARNING, logger="aiwaf.native"):
        for _ in range(2):
            ctx, status, _ = pipeline.begin(ip="8.8.8.8", path="/items/x", method="GET", headers={}, query=b"",
                                            body=b"", exists=True, methods={"GET"}, uuid="not-a-uuid")
    assert _OlderEngine.calls == [16, 16]
    assert [r.getMessage() for r in caplog.records].count(
        "Route UUIDs need a newer aiwaf-rust to be checked; only ?uuid= is checked. Run: pip install -U aiwaf-rust") == 1


# Host stores: facts in, effects out.

needs_host_store = pytest.mark.skipif(
    not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None)
    or not rust.NativeCSVPipeline.supports("host_store"), reason="aiwaf-rust without host stores")


def host_begin(e, path="/x", *, ip="203.0.113.60", country="", query=b"", exists=True, exempt=False, **facts):
    return e.begin(ip, path, "GET", {"AIWAF_COUNTRY": country}, query, b"", exists, True, True, True, False, set(),
                   exempt, 100.0, "2026-10-12T00:00:00+00:00", False, None, None, None, None, None,
                   facts.get("blocked"), facts.get("dynamic_keywords"), facts.get("geo_countries"))


@needs_host_store
def test_host_store_mode_takes_facts_and_returns_effects(tmp_path):
    e = engine(tmp_path, enabled=["ip_keyword_block", "geo_block"], policy_store="host")
    # No policy CSV files: the host's stores hold that state.
    assert not any((tmp_path / name).exists() for name in ("blacklist.csv", "whitelist.csv", "keywords.csv"))
    ctx, status, _ = host_begin(e, blocked=True)
    assert status == 403 and ctx.take_effects() == []
    ctx, status, _ = host_begin(e, country="cn", geo_countries={"CN"})
    assert status == 403
    assert ctx.take_effects() == [("block", ["203.0.113.60", "Geo blocked: CN"])]
    assert ctx.take_effects() == []
    # A keyword block is reported for the host to store; an exemption from the host skips it.
    ctx, status, _ = host_begin(e, "/admin-backup.php", dynamic_keywords=[], exists=False)
    assert status == 403 and ctx.take_effects()[0][0] == "block"
    ctx, status, _ = host_begin(e, "/admin-backup.php", dynamic_keywords=[], exists=False, exempt=True)
    assert status == 0


@needs_host_store
@pytest.mark.parametrize("path,query,exists", [
    ("/admin-backup.php", b"", False), ("/reports", b"q=union select panelx", True), ("/panelx/config", b"", False),
    ("/blog/panelx", b"", True), ("/../panelx", b"", False), ("/home", b"", True),
    ("/panelx", b"", False),  # Blocked only through the learned keyword, like /panelx/config.
])
def test_learned_keywords_from_the_host_decide_as_the_csv_file_does(tmp_path, path, query, exists):
    learned = ["panelx", "zzprobe.php"]
    csv_engine = engine(tmp_path / "csv", enabled=["ip_keyword_block"])
    with (tmp_path / "csv" / "keywords.csv").open("a", encoding="utf-8") as f:
        f.writelines(f"{keyword},2026-10-12\n" for keyword in learned)
    host_engine = engine(tmp_path / "host", enabled=["ip_keyword_block"], policy_store="host")
    from_csv = begin(csv_engine, path=path, query=query, exists=exists)[1:]
    from_host = host_begin(host_engine, path, query=query, exists=exists, dynamic_keywords=learned)[1:]
    assert from_host == from_csv


@needs_host_store
def test_host_store_mode_reports_learned_keywords(tmp_path):
    e = engine(tmp_path, enabled=["ai_anomaly"], policy_store="host")
    # phpmyadmin makes the 404 suspicious; panelx is the new word, the rest are already known.
    ctx, status, _ = host_begin(e, "/old/phpmyadmin-panelx", exists=False, dynamic_keywords=[])
    e.finish(ctx, 404, 100.1, "0")
    assert ("keywords", ["panelx"]) in ctx.take_effects()
    assert not (tmp_path / "keywords.csv").exists()


class FakeStores:
    def __init__(self, *, blocked=(), exempt=(), keywords=(), countries=()):
        self.blocked, self.exempt = set(blocked), set(exempt)
        self.keywords, self.countries = list(keywords), list(countries)
        self.calls, self.blocks, self.learned = [], [], []

    def is_blocked(self, ip):
        self.calls.append(("is_blocked", ip))
        return ip in self.blocked

    def is_exempted(self, ip, path):
        return ip in self.exempt

    def top_keywords(self, n):
        self.calls.append(("top_keywords", n))
        return self.keywords[:n]

    def blocked_countries(self):
        self.calls.append(("blocked_countries",))
        return self.countries

    def block(self, ip, reason):
        self.blocks.append((ip, reason))
        self.blocked.add(ip)

    def learn(self, keywords):
        self.learned.extend(keywords)


@needs_host_store
def test_adapter_reads_and_writes_the_hosts_stores(tmp_path):
    stores = FakeStores(blocked={"203.0.113.61"}, keywords=["panelx"])
    from aiwaf.core.csv_pipeline import CSVPipeline

    pipeline = CSVPipeline({"data_dir": str(tmp_path / "data"), "exempt_paths": [], "enabled": ["ip_keyword_block"],
                            "manifest_path": str(tmp_path / "none.json")}, stores=stores)

    def get(path, ip):
        ctx, status, _ = pipeline.begin(ip=ip, path=path, method="GET", headers={}, query=b"", body=b"",
                                        exists=False, methods={"GET"})
        pipeline.finish(ctx, status or 200)
        return status

    assert get("/home", "203.0.113.61") == 403
    assert get("/admin-backup.php", "203.0.113.62") == 403
    assert stores.blocks == [("203.0.113.62", stores.blocks[0][1])]
    assert get("/home", "203.0.113.62") == 403  # The store now has the block.
    # Keywords are cached between requests.
    assert sum(call[0] == "top_keywords" for call in stores.calls) == 1
    # Exemption answers are cached for store_cache_seconds, so a new exemption waits for it.
    stores.exempt.add("203.0.113.61")
    assert get("/home", "203.0.113.61") == 403
    pipeline._exempted.items.clear()
    assert get("/home", "203.0.113.61") == 0


def _store_pipeline(tmp_path, stores, **config):
    from aiwaf.core.csv_pipeline import CSVPipeline

    return CSVPipeline({"data_dir": str(tmp_path / "data"), "exempt_paths": [], "learning": False,
                        "manifest_path": str(tmp_path / "none.json"), **config}, stores=stores)


def _store_get(pipeline, path, ip, country="", exists=True):
    ctx, status, _ = pipeline.begin(ip=ip, path=path, method="GET", headers={}, query=b"", body=b"",
                                    exists=exists, methods={"GET"}, country=country)
    pipeline.finish(ctx, status or 200)
    return status


def _reads(stores):
    return [call[0] for call in stores.calls]


@needs_host_store
def test_only_the_store_facts_a_request_uses_are_read(tmp_path):
    stores = FakeStores(exempt={"203.0.113.70"})
    stores.is_exempted = lambda ip, path, base=stores.is_exempted: (stores.calls.append(("is_exempted",)), base(ip, path))[1]
    pipeline = _store_pipeline(tmp_path, stores, enabled=["ip_keyword_block", "geo_block", "logging"], learning=True,
                               path_rules=[{"PREFIX": "/static/", "DISABLE": ["ip_keyword_block", "geo_block"]}],
                               store_cache_seconds=0)
    _store_get(pipeline, "/a", "203.0.113.71")
    assert sorted(_reads(stores)) == ["blocked_countries", "is_blocked", "is_exempted", "top_keywords"]
    stores.calls.clear()
    # Only logging runs here, so nothing is read.
    _store_get(pipeline, "/static/app.css", "203.0.113.71")
    assert _reads(stores) == []
    # An exempt request runs logging alone, so only the exemption is read.
    _store_get(pipeline, "/a", "203.0.113.70")
    assert _reads(stores) == ["is_exempted"]


@needs_host_store
def test_blocklist_answers_are_cached_and_own_blocks_apply_at_once(tmp_path):
    stores = FakeStores(keywords=[])
    pipeline = _store_pipeline(tmp_path, stores, enabled=["ip_keyword_block"], store_cache_seconds=60)
    assert _store_get(pipeline, "/a", "203.0.113.80") == 0
    assert _store_get(pipeline, "/b", "203.0.113.80") == 0
    assert _reads(stores).count("is_blocked") == 1
    # A block made elsewhere waits for the cache; this pipeline's own block applies at once.
    stores.blocked.add("203.0.113.80")
    assert _store_get(pipeline, "/c", "203.0.113.80") == 0
    assert _store_get(pipeline, "/admin-backup.php", "203.0.113.81", exists=False) == 403
    assert stores.blocks and _store_get(pipeline, "/d", "203.0.113.81") == 403
    assert _reads(stores).count("is_blocked") == 2


def test_cached_lookups_are_bounded():
    from aiwaf.core.csv_pipeline import _Lookups

    lookups = _Lookups(60, limit=2)
    for key in "abc":
        lookups.get(key, lambda: True)
    assert list(lookups.items) == ["b", "c"]
    lookups.get("b", lambda: False)  # A hit keeps its value and becomes the newest.
    assert list(lookups.items) == ["c", "b"] and lookups.get("b", lambda: False) is True
    off = _Lookups(0)
    off.set("a", True)
    assert off.get("a", lambda: False) is False and not off.items


def test_adapter_refuses_incomplete_stores_and_older_engines(tmp_path, monkeypatch):
    from aiwaf.core.csv_pipeline import CSVPipeline

    with pytest.raises(TypeError, match="learn"):
        CSVPipeline({"data_dir": str(tmp_path)}, stores=type("S", (), {"is_blocked": lambda s, ip: False})())
    _with_older_engine(monkeypatch)
    with pytest.raises(RuntimeError, match="host_store"):
        _adapter(tmp_path, stores=FakeStores())


def test_the_stores_base_class_documents_every_method():
    from aiwaf.core.csv_pipeline import PipelineStores

    base = PipelineStores()
    for call in (lambda: base.is_blocked("ip"), lambda: base.is_exempted("ip", "/"), lambda: base.top_keywords(1),
                 base.blocked_countries, lambda: base.block("ip", "why"), lambda: base.learn(["kw"])):
        with pytest.raises(NotImplementedError):
            call()


# Shared rate counts: a counter (such as Redis) counts, aiwaf-rust decides.

needs_rate_counter = pytest.mark.skipif(
    not getattr(getattr(rust, "NativeCSVPipeline", None), "supports", None)
    or not rust.NativeCSVPipeline.supports("rate_counter"), reason="aiwaf-rust without rate counters")


class SharedCounter:
    """One sliding window per key, shared by every pipeline given it, as Redis is."""

    def __init__(self):
        import threading
        self.windows, self.calls, self.lock = {}, [], threading.Lock()

    def count(self, key, now, window_seconds):
        with self.lock:
            self.calls.append((key, window_seconds))
            window = [t for t in self.windows.get(key, []) if now - t < window_seconds] + [now]
            self.windows[key] = window
            return len(window)


def _counted(tmp_path, counter, name, **config):
    from aiwaf.core.csv_pipeline import CSVPipeline

    return CSVPipeline({"data_dir": str(tmp_path / name), "exempt_paths": [], "learning": False,
                        "manifest_path": str(tmp_path / "missing.json"), "enabled": ["rate_limit"],
                        "rate_backend": "counter", "rate_max": 3, "rate_flood": 5, **config}, rate_counter=counter)


@needs_rate_counter
def test_workers_sharing_a_counter_share_one_limit(tmp_path):
    counter = SharedCounter()
    workers = [_counted(tmp_path, counter, "a"), _counted(tmp_path, counter, "b")]
    statuses = [_request(workers[i % 2], "/api")[0] for i in range(6)]
    # Three requests pass in total, not three per worker; the sixth is a flood.
    assert statuses == [0, 0, 0, 429, 429, 403]
    assert counter.calls[0] == ("ratelimit:8.8.8.8:/api", 10.0)


@needs_rate_counter
def test_path_rule_windows_reach_the_counter(tmp_path):
    counter = SharedCounter()
    pipeline = _counted(tmp_path, counter, "a", rate_key_mode="ip",
                        path_rules=[{"PREFIX": "/api/", "RATE_LIMIT": {"WINDOW": 30, "MAX": 1}}])
    assert [_request(pipeline, "/api/x")[0] for _ in range(2)] == [0, 429]
    assert counter.calls[-1] == ("ratelimit:8.8.8.8", 30.0)


@needs_rate_counter
@pytest.mark.parametrize("result", [RuntimeError("redis down"), "many"])
def test_a_failing_counter_answers_503_as_the_regular_limiter_does(tmp_path, result):
    class Broken:
        def count(self, key, now, window_seconds):
            if isinstance(result, Exception):
                raise result
            return result

    status, payload = _request(_counted(tmp_path, Broken(), "a"), "/api")
    assert (status, json.loads(payload)) == (503, {"error": "temporarily_unavailable"})


def test_rate_counter_configuration_is_checked_at_startup(tmp_path):
    from aiwaf.core.csv_pipeline import CSVPipeline

    base = {"data_dir": str(tmp_path / "data"), "manifest_path": str(tmp_path / "missing.json")}

    class Async:
        async def count(self, key, now, window_seconds):
            return 1

    with pytest.raises(ValueError, match="requires a rate_counter"):
        CSVPipeline({**base, "rate_backend": "counter"})
    with pytest.raises(ValueError, match="requires rate_backend=counter"):
        CSVPipeline(base, rate_counter=SharedCounter())
    with pytest.raises(ValueError, match="Choose rate_backend=counter"):
        CSVPipeline({**base, "rate_backend": "redis", "rate_redis_url": "redis://localhost"}, rate_counter=SharedCounter())
    with pytest.raises(TypeError, match="synchronous count"):
        CSVPipeline({**base, "rate_backend": "counter"}, rate_counter=Async())
    with pytest.raises(ValueError, match="rate_redis_url is required"):
        CSVPipeline({**base, "rate_backend": "redis"})
    with pytest.raises(ValueError, match="rate_cache_namespace"):
        CSVPipeline({**base, "rate_backend": "redis", "rate_redis_url": "redis://localhost", "rate_cache_namespace": ""})


def test_adapter_refuses_a_rate_counter_on_an_older_engine(tmp_path, monkeypatch):
    _with_older_engine(monkeypatch)
    with pytest.raises(RuntimeError, match="rate_counter"):
        _adapter(tmp_path, rate_backend="redis", rate_redis_url="redis://localhost")

