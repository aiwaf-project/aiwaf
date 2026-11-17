"""Flask's AIWAF(app) runs as the native pipeline when every setting carries over."""
import logging

import pytest
from flask import Flask

rust = pytest.importorskip("aiwaf_rust")
from aiwaf.core import native_settings  # noqa: E402
from aiwaf.core.native_settings import DEFAULT_FEATURES  # noqa: E402

FULL_ENGINE = all(rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
    if hasattr(rust.NativeCSVPipeline, "supports") else False
needs_full_engine = pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)

    def make(**config):
        app = Flask(__name__)
        app.config.update(AIWAF_DATA_DIR=str(tmp_path / "data"), AIWAF_RATE_MAX=1, AIWAF_RATE_FLOOD=10,
                          AIWAF_LOG_DIR=str(tmp_path / "logs"),
                          AIWAF_PATH_MANIFEST=str(tmp_path / "none.json"), **config)
        app.add_url_rule("/items", "items", lambda: {"ok": True})
        return app
    return make


@needs_full_engine
def test_aiwaf_runs_as_native_when_everything_carries_over(make_app, caplog):
    from aiwaf.flask import AIWAF

    app = make_app()
    with caplog.at_level(logging.INFO, "aiwaf.native"):
        aiwaf = AIWAF(app)
    assert aiwaf.native is not None and aiwaf.middleware_instances == {}
    assert "running the native pipeline for Flask" in caplog.text
    # The stages the regular middleware would register, logging where its trainer reads.
    assert "logging" in aiwaf.native.pipeline.profile["enabled"]
    assert aiwaf.native.pipeline.profile["log_path"] == str(app.config["AIWAF_LOG_DIR"]) + "/aiwaf_native_access.csv"
    client = app.test_client()
    statuses = [client.get("/items", environ_base={"REMOTE_ADDR": "198.51.100.41"}).status_code for _ in range(2)]
    assert statuses == [200, 429]


@needs_full_engine
@pytest.mark.parametrize("how", ["argument", "config", "difference", "database"])
def test_aiwaf_stays_regular_when_asked_or_when_settings_differ(make_app, how):
    from aiwaf.flask import AIWAF

    app = make_app(**({"AIWAF_NATIVE_PIPELINE": False} if how == "config" else {})
                   , **({"AIWAF_SOMETHING_UNKNOWN": 1} if how == "difference" else {}))
    if how == "database":
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    aiwaf = AIWAF(app, native_pipeline=False if how == "argument" else None,
                  use_database=True if how == "database" else None)
    assert aiwaf.native is None and aiwaf.middleware_instances


def test_an_engine_without_every_feature_stays_regular(make_app, monkeypatch):
    from aiwaf.flask import AIWAF

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: False)
    assert AIWAF(make_app()).native is None


@needs_full_engine
def test_the_regular_model_carries_over_only_if_aiwaf_rust_made_it(make_app, tmp_path):
    import json
    from aiwaf.flask import AIWAF
    from aiwaf.flask.flask_integration import AIWAF as Integration

    forest = rust.IsolationForest(n_estimators=5, random_state=1)
    forest.fit([[float(i), 0.0, 0.0, 1.0, 1.0, 0.0] for i in range(16)])
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"model_type": "aiwaf_rust.IsolationForest", "model_state": forest.to_json()}))
    app = make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_FORCE_AI=True)
    assert native_settings.is_rust_model(Integration._regular_model(app))
    assert AIWAF(app).native is not None
    # Held back by the regular middleware until there are enough logs, so it stays regular.
    held = make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_MIN_AI_LOGS=10**9)
    assert AIWAF(held).native is None
    path.write_text(json.dumps({"model": {"kind": "sklearn"}}))
    assert AIWAF(make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_FORCE_AI=True)).native is None
    path.write_text("{")
    assert Integration._regular_model(app) is None
    assert Integration._regular_model(make_app(AIWAF_MODEL_PATH=str(tmp_path / "missing.json"))) is None


@needs_full_engine
def test_a_native_pipeline_that_cant_start_falls_back_to_the_regular_middleware(make_app, monkeypatch, caplog):
    from aiwaf.flask import AIWAF, csv_pipeline

    def fail(*args, **kwargs):
        raise ValueError("bad profile")

    monkeypatch.setattr(csv_pipeline, "CSVPipeline", fail)
    app = make_app()
    with caplog.at_level(logging.WARNING, "aiwaf.native"):
        aiwaf = AIWAF(app)
    assert aiwaf.native is None and "rate_limit" in aiwaf.middleware_instances
    assert "couldn't start for Flask (ValueError: bad profile)" in caplog.text
    assert "aiwaf_csv_pipeline" not in app.extensions
    client = app.test_client()
    statuses = [client.get("/items", environ_base={"REMOTE_ADDR": "198.51.100.44"}).status_code for _ in range(2)]
    assert statuses == [200, 429]


@needs_full_engine
def test_the_reason_an_app_stays_regular_is_logged(make_app, caplog):
    from aiwaf.flask import AIWAF

    native_settings._REPORTED.clear()
    with caplog.at_level(logging.INFO, "aiwaf.native"):
        assert AIWAF(make_app(AIWAF_SOMETHING_UNKNOWN=1)).native is None
    assert "running the regular middleware for Flask" in caplog.text
    assert "AIWAF_SOMETHING_UNKNOWN: not carried" in caplog.text
    native_settings._REPORTED.clear()


@needs_full_engine
def test_settings_native_reads_the_same_way_or_never_reads_dont_keep_the_regular_middleware(make_app, tmp_path):
    from aiwaf.flask import AIWAF

    app = make_app(AIWAF_ACCESS_LOG=str(tmp_path / "access.log"), AIWAF_AI_CONTAMINATION=0.05,
                   AIWAF_GEOIP_DB_PATH=str(tmp_path / "geo.mmdb"), AIWAF_GEO_CACHE_SECONDS=60,
                   AIWAF_GEO_CACHE_PREFIX="geo", AIWAF_RUST_REQUEST_ENGINE=True, AIWAF_TRAINING_STATE_DIR=str(tmp_path))
    assert AIWAF(app).native is not None


@needs_full_engine
@pytest.mark.parametrize("log_format", ["json", "csv", "combined"])
def test_a_chosen_log_format_is_written_by_the_regular_logger_beside_the_native_pipeline(make_app, tmp_path, log_format):
    import json
    from aiwaf.flask import AIWAF
    from aiwaf.flask.logging_middleware import AIWAFLoggingMiddleware

    app = make_app(AIWAF_LOG_FORMAT=log_format)
    aiwaf = AIWAF(app)
    assert aiwaf.native is not None
    assert "logging" not in aiwaf.native.pipeline.profile["enabled"]
    assert list(aiwaf.middleware_instances) == ["logging"]
    assert isinstance(aiwaf.middleware_instances["logging"], AIWAFLoggingMiddleware)
    client = app.test_client()
    statuses = [client.get("/items", environ_base={"REMOTE_ADDR": "198.51.100.45"}).status_code for _ in range(2)]
    assert statuses == [200, 429]
    aiwaf.middleware_instances["logging"].close()
    lines = (tmp_path / "logs" / "access.log").read_text().splitlines()
    assert not (tmp_path / "logs" / "aiwaf_native_access.csv").exists()
    if log_format == "json":
        rows = [json.loads(line) for line in lines]
        assert [(row["status_code"], row["blocked"]) for row in rows] == [(200, False), (429, True)]
    elif log_format == "csv":
        assert len(lines) == 3 and lines[0].startswith("timestamp,")
    else:
        assert len(lines) == 2 and '" 429 ' in lines[1]
    # Blocked requests reach the regular logger's security log too.
    assert "BLOCKED 198.51.100.45" in (tmp_path / "logs" / "aiwaf.log").read_text()


@needs_full_engine
def test_the_regular_keyword_lists_and_block_info_settings_carry_over(make_app):
    from aiwaf.flask import AIWAF

    app = make_app(AIWAF_ALLOWED_PATH_KEYWORDS=["widgets"], AIWAF_EXEMPT_KEYWORDS=["gadgets"], AIWAF_WINDOW_SECONDS=30,
                   AIWAF_CAPTURE_EXTENDED_REQUEST_INFO=True, AIWAF_BLOCKLIST_MAX_HEADERS=10,
                   # Flask's regular middleware reads no header caps, so these change nothing natively either.
                   AIWAF_MAX_USER_AGENT_LENGTH=10)
    aiwaf = AIWAF(app)
    assert aiwaf.native is not None
    profile = aiwaf.native.pipeline.profile
    # The app's routes, its allowed and exempt keywords, and the regular defaults.
    assert {"widgets", "gadgets", "items", "profile"} <= set(profile["legitimate_keywords"])
    assert profile["anomaly_window"] == 30.0 and "header_max_user_agent" not in profile


@needs_full_engine
def test_a_block_records_the_requests_headers_when_the_app_asks(make_app, tmp_path):
    from aiwaf.flask import AIWAF

    app = make_app(AIWAF_CAPTURE_EXTENDED_REQUEST_INFO=True)
    with app.app_context():
        assert AIWAF(app).native is not None
        client = app.test_client()
        response = client.get("/wp-login.php", environ_base={"REMOTE_ADDR": "198.51.100.49"},
                              headers={"User-Agent": "Mozilla/5.0 Chrome/130", "Cookie": "session=secret"})
        assert response.status_code == 403
    entry = next(line for file in (tmp_path / "data").glob("*.csv") for line in file.read_text().splitlines()
                 if "198.51.100.49" in line)
    assert "Mozilla/5.0 Chrome/130" in str(entry) and "secret" not in str(entry)
