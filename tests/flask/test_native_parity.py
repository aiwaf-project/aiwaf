"""Flask's regular middleware and the native pipeline it switches to answer the same requests alike."""
import os
import sys

import pytest
from flask import Flask, abort, jsonify, request
from flask.testing import FlaskClient

# The test client as Flask has it: the shared fixture adds browser headers to every request,
# and these cases send exactly the headers they name. Taken when the module is imported, before
# any fixture runs.
OPEN = FlaskClient.open

rust = pytest.importorskip("aiwaf_rust")
from aiwaf.core.native_settings import DEFAULT_FEATURES  # noqa: E402

FULL_ENGINE = all(rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
    if hasattr(rust.NativeCSVPipeline, "supports") else False
pytestmark = pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")

KNOWN = "6f1c2a3e-4b5d-4c6e-8f70-112233445566"
BROWSER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/130 Safari/537.36", "Accept": "text/html",
           "Accept-Language": "en", "Accept-Encoding": "gzip", "Connection": "keep-alive"}
HEADER_CASES = [
    [("get", "/items", {})],
    [("get", "/items", {"Accept": None})],
    [("get", "/items", {"User-Agent": ""})],
    [("get", "/items", {"User-Agent": None})],
    [("get", "/items", {"Accept-Language": None, "Accept-Encoding": None, "Connection": None})],
    [("get", "/items", {"Accept": "*/*", "Accept-Language": None, "Accept-Encoding": None})],
    [("get", "/items", {"User-Agent": "Googlebot/2.1", "Accept": None})],
    [("get", "/items", {"User-Agent": "Mozilla/4.0"})],
    [("get", "/items", {"User-Agent": "Mozilla/5"})],
    [("get", "/items", {"User-Agent": "curl/8.0"})],
    [("get", "/static/app.css", {"User-Agent": "curl/8.0"})],
    [("get", "/items", {"User-Agent": "Mozilla/5.0 Chrome/130 " + "a" * 600})],
    [("get", "/items", {f"X-Extra-{i}": "1" for i in range(120)})],
    [("post", "/form", {"Accept": None})],
]
SEQUENCES = [
    [("get", "/items", {})], [("get", "/wp-login.php", {}), ("get", "/items", {})], [("get", "/.env", {})],
    [("get", "/items?q=' OR 1=1--", {})], [("get", "/items?q=union select", {})], [("get", "/nothing-here", {})],
    [("get", "/xmlrpc.php", {})], [("get", "/ShElL", {})], [("post", "/form?name=x'; DROP TABLE users;--", {})],
    [("get", f"/items?uuid={KNOWN}", {})], [("get", "/items?uuid=not-a-uuid", {}), ("get", "/items", {})],
    [("get", "/items?uuid=6f1c2a3e-4b5d-4c6e-8f70-000000000000", {})] * 7,
]
counter = [0]
# The country the stand-in GeoIP lookup reports for the request being sent.
GEO = {"country": None}
# Every stage but geo blocking and logging, which other tests compare.
STAGES = ["ip_keyword_block", "rate_limit", "ai_anomaly", "honeypot", "uuid_tamper", "header_validation"]


@pytest.fixture(autouse=True)
def automatic(monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)


def run(tmp_path, native, cases, **config):
    from aiwaf.flask import AIWAF

    app = Flask(__name__)
    app.config.update(AIWAF_DATA_DIR=str(tmp_path / f"data-{native}"), AIWAF_LOG_DIR=str(tmp_path / f"logs-{native}"),
                      AIWAF_PATH_MANIFEST=str(tmp_path / "none.json"), AIWAF_RATE_MAX=1000, AIWAF_RATE_FLOOD=2000)
    app.config.update(config)

    @app.route("/items")
    @app.route("/items/<int:n>")
    @app.route("/form", methods=["GET", "POST"])
    @app.route("/user/<uuid>")
    def item(n=None, uuid=None):
        uuid = uuid or request.args.get("uuid")
        if uuid not in (None, KNOWN):
            abort(404)
        return jsonify(ok=True)

    aiwaf = AIWAF(app, native_pipeline=None if native else False, middlewares=STAGES)
    assert (aiwaf.native is not None) == native, config
    client = app.test_client()
    results = []
    for requests in cases:
        counter[0] += 1
        ip = f"198.{60 + counter[0] // 250}.{counter[0] % 250}.1"
        statuses = []
        for method, url, changes in requests:
            changes = dict(changes)
            GEO["country"] = changes.pop("_country", None)
            headers = {key: value for key, value in {**BROWSER, **changes}.items() if value is not None}
            response = OPEN(client, url, method=method.upper(), headers=headers, environ_base={"REMOTE_ADDR": ip})
            statuses.append(response.status_code)
        results.append(statuses)
    return results


def alike(tmp_path, cases, **config):
    regular, native = run(tmp_path, False, cases, **config), run(tmp_path, True, cases, **config)
    differences = {index: (cases[index], regular[index], native[index])
                   for index in range(len(cases)) if regular[index] != native[index]}
    assert differences == {}
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(config, regular)
    return regular


@pytest.mark.parametrize("config", [{}, {"AIWAF_REQUIRED_HEADERS": []}, {"AIWAF_HEADER_QUALITY_MIN_SCORE": 7},
                                    {"AIWAF_REQUIRED_HEADERS": {"GET": ["HTTP_USER_AGENT"], "DEFAULT": ["HTTP_USER_AGENT", "HTTP_ACCEPT"]}}])
def test_headers(tmp_path, config):
    alike(tmp_path, HEADER_CASES, **config)


def test_keywords_injection_and_uuids(tmp_path):
    alike(tmp_path, SEQUENCES)


def test_rate_limits(tmp_path):
    alike(tmp_path, [[("get", "/items", {})] * 8, [("get", f"/items/{n}", {}) for n in range(8)]],
          AIWAF_RATE_MAX=3, AIWAF_RATE_FLOOD=6)


def test_honeypot_timing(tmp_path):
    alike(tmp_path, [[("get", "/form", {}), ("post", "/form", {})], [("post", "/form", {})]], AIWAF_MIN_FORM_TIME=5)


def test_exempt_paths(tmp_path):
    alike(tmp_path, [[("get", "/form?q=' OR 1=1--", {})], [("get", "/form/wp-login.php", {})], [("get", "/forms", {})],
                     [("get", "/items/.env", {})], [("get", "/form", {})] * 6, [("get", "/form", {"Accept": None})]],
          AIWAF_EXEMPT_PATHS=["/form"], AIWAF_RATE_MAX=3, AIWAF_RATE_FLOOD=6)


# Probes that look like scans, then requests that reuse their words: what each pipeline learns
# from the probes decides whether the later requests are blocked.
LEARNING = [
    [("get", "/backup-old/config.php.bak", {}), ("get", "/wordpress-setup/install.php", {}), ("get", "/dbadmin/shell.php", {}),
     ("get", "/temp-files/../../etc/passwd", {}), ("get", "/vendor/phpunit/eval-stdin.php", {})],
    [("get", "/items?cmd=whoami", {}), ("get", "/oldsite/exec?cmd=id", {}), ("get", "/console-debug/system.php", {}),
     ("get", "/reports-archive/data?q=UNION SELECT", {}), ("get", "/legacy-filemanager/xmlrpc-relay", {})],
    [("get", "/nothing/backup-old", {}), ("get", "/wordpress-setup", {}), ("get", "/dbadmin", {}), ("get", "/oldsite", {}),
     ("get", "/console-debug", {}), ("get", "/reports-archive", {}), ("get", "/legacy-filemanager", {}), ("get", "/items", {})],
    [("get", "/items/oldsite", {}), ("get", "/items/nothing", {}), ("get", "/form/oldsite.php", {}), ("get", "/userx/oldsite", {})],
]


def learned(tmp_path, native):
    from aiwaf.core.storage_schema import KEYWORDS_CSV

    file = tmp_path / f"data-{native}" / KEYWORDS_CSV
    return sorted(line.split(",")[0] for line in file.read_text().splitlines()[1:]) if file.exists() else []


def test_keyword_learning(tmp_path):
    statuses = alike(tmp_path, LEARNING)
    assert learned(tmp_path, True) == learned(tmp_path, False)
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(statuses, learned(tmp_path, False))


def test_the_anomaly_models_blocks(tmp_path):
    import json

    forest = rust.IsolationForest(n_estimators=50, random_state=7)
    forest.fit([[float(8 + i % 12), 0.0, 0.0, 0.02 + (i % 5) / 100, float(1 + i % 3), 0.0] for i in range(200)])
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"model_type": "aiwaf_rust.IsolationForest", "model_state": forest.to_json()}))
    # A scan: 404s for paths with the anomaly stage's keywords, but none of the keyword stage's.
    scan = [[("get", f"/{word}/area{n}", {}) for n, word in enumerate(
        ["backup", "database", "mysql", "dump", "setup", "install", "installer", "adminer", "cgi-bin", "scripts",
         "base64", "backup", "database", "mysql", "dump", "setup"])] + [("get", "/items", {})],
            [("get", "/items", {}), ("get", "/nothing-here", {})]]
    settings = dict(AIWAF_FORCE_AI=True, AIWAF_ENABLE_KEYWORD_LEARNING=False)
    with_model = alike(tmp_path, scan, AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, **settings)
    without = alike(tmp_path, scan, AIWAF_MODEL_PATH=str(tmp_path / "none.json"), **settings)
    # The model changes what's blocked, so it was exercised.
    assert with_model != without


@pytest.fixture
def countries(monkeypatch):
    """Countries from a stand-in for the GeoIP lookup both pipelines use."""
    from aiwaf.core import geoip

    monkeypatch.setattr(geoip, "lookup_country", lambda *args, **kwargs: GEO["country"])


GEO_CASES = [[("get", "/items", {"_country": country})] for country in ["US", "CN", "cn", "GB", None, "", "JP"]]


@pytest.mark.parametrize("config", [
    {"AIWAF_GEO_BLOCK_ENABLED": True, "AIWAF_GEO_BLOCK_COUNTRIES": ["CN", "jp"]},
    {"AIWAF_GEO_BLOCK_ENABLED": True, "AIWAF_GEO_ALLOW_COUNTRIES": ["US", "GB"]},
    {"AIWAF_GEO_BLOCK_ENABLED": False, "AIWAF_GEO_BLOCK_COUNTRIES": ["CN"]},
    {"AIWAF_GEO_BLOCK_COUNTRIES": ["CN"]},
])
def test_geo_blocking(tmp_path, monkeypatch, countries, config):
    stages = STAGES + ["geo_block"]
    monkeypatch.setattr(sys.modules[__name__], "STAGES", stages)
    alike(tmp_path, GEO_CASES, **config)


def test_countries_blocked_in_the_store(tmp_path, monkeypatch, countries):
    from aiwaf.core.storage_schema import GEO_BLOCKED_COUNTRIES_CSV

    monkeypatch.setattr(sys.modules[__name__], "STAGES", STAGES + ["geo_block"])
    for native in (False, True):
        directory = tmp_path / f"data-{native}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / GEO_BLOCKED_COUNTRIES_CSV).write_text("country,added_date\nGB,2026-11-17T00:00:00\n")
    answers = alike(tmp_path, GEO_CASES, AIWAF_GEO_BLOCK_ENABLED=True)
    assert 403 in [status for statuses in answers for status in statuses]


LOGGED = [[("get", "/items", {})], [("get", "/nothing-here", {})], [("get", "/wp-login.php", {})], [("post", "/form", {})],
          [("get", "/items?q=a", {})]]


def trained_on(tmp_path, native):
    """The requests aiwaf train reads from a pipeline's request log: address, method, path and status."""
    from aiwaf.flask.trainer import FlaskAITrainer

    app = Flask(__name__)
    app.config.update(AIWAF_LOG_DIR=str(tmp_path / f"logs-{native}"))
    trainer = FlaskAITrainer(app)
    with app.app_context():
        records = [trainer._parse(line) for line in trainer._read_all_logs()]
    return sorted((r["path"].split("?")[0], str(r["status"])) for r in records if r)


@pytest.mark.parametrize("log_format", [None, "json", "csv", "combined"])
def test_what_the_trainer_reads_from_the_request_log(tmp_path, monkeypatch, log_format):
    monkeypatch.setattr(sys.modules[__name__], "STAGES", STAGES + ["logging"])
    alike(tmp_path, LOGGED, **({"AIWAF_LOG_FORMAT": log_format} if log_format else {}))
    regular, native = trained_on(tmp_path, False), trained_on(tmp_path, True)
    if os.environ.get("AIWAF_PARITY_PRINT"):
        print(log_format, regular, native)
    assert native == regular
    if log_format in (None, "combined"):
        # In JSON and CSV the regular logger writes both pipelines' logs, in a form this reader
        # doesn't parse, for either.
        assert len(regular) == len(LOGGED)
