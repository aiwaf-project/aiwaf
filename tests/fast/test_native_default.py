"""FastAPI's AIWAF(app) runs as the native pipeline when every setting carries over."""
import logging

import pytest

pytest.importorskip("fastapi")
rust = pytest.importorskip("aiwaf_rust")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from aiwaf.core import native_settings  # noqa: E402
from aiwaf.core.native_settings import DEFAULT_FEATURES  # noqa: E402

FULL_ENGINE = all(rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
    if hasattr(rust.NativeCSVPipeline, "supports") else False


def build(monkeypatch, tmp_path, **kwargs):
    from aiwaf.fast import AIWAF

    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    monkeypatch.setenv("AIWAF_DATA_DIR", str(tmp_path / "data"))
    app = FastAPI()
    app.add_api_route("/items", lambda: {"ok": True})
    aiwaf = AIWAF(app, **kwargs)
    return app, aiwaf


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
def test_aiwaf_runs_as_native_when_everything_carries_over(monkeypatch, tmp_path, caplog):
    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware

    with caplog.at_level(logging.INFO, "aiwaf.native"):
        app, aiwaf = build(monkeypatch, tmp_path)
    assert aiwaf.native is True
    # The native pipeline, and the regular request logger beside it, innermost.
    from aiwaf.fast.middleware.logging_middleware import ASGIRequestLogger as AIWAFLoggingMiddleware

    assert [m.cls for m in app.user_middleware] in ([NativeAIWAFMiddleware], [NativeAIWAFMiddleware, AIWAFLoggingMiddleware])
    assert "running the native pipeline for FastAPI" in caplog.text
    assert TestClient(app).get("/items").status_code == 200


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
def test_aiwaf_stays_regular_when_asked(monkeypatch, tmp_path):
    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware

    app, aiwaf = build(monkeypatch, tmp_path, native_pipeline=False)
    assert aiwaf.native is False and NativeAIWAFMiddleware not in [m.cls for m in app.user_middleware]


def test_an_engine_without_every_feature_stays_regular(monkeypatch, tmp_path):
    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: False)
    app, aiwaf = build(monkeypatch, tmp_path)
    assert aiwaf.native is False and app.user_middleware


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
def test_a_native_pipeline_that_cant_start_falls_back_to_the_regular_middleware(monkeypatch, tmp_path, caplog):
    from aiwaf.fast import csv_pipeline

    def fail(*args, **kwargs):
        raise ValueError("bad profile")

    monkeypatch.setattr(csv_pipeline, "CSVPipeline", fail)
    with caplog.at_level(logging.WARNING, "aiwaf.native"):
        app, aiwaf = build(monkeypatch, tmp_path)
    assert aiwaf.native is False and app.user_middleware
    assert csv_pipeline.NativeAIWAFMiddleware not in [m.cls for m in app.user_middleware]
    assert "couldn't start for FastAPI (ValueError: bad profile)" in caplog.text
    assert TestClient(app).get("/items").status_code == 200


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
def test_the_reason_an_app_stays_regular_is_logged(monkeypatch, tmp_path, caplog):
    native_settings._REPORTED.clear()
    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: feature != "rate_counter")
    with caplog.at_level(logging.INFO, "aiwaf.native"):
        app, aiwaf = build(monkeypatch, tmp_path)
    assert aiwaf.native is False
    assert "running the regular middleware for FastAPI" in caplog.text and "too old" in caplog.text
    native_settings._REPORTED.clear()


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
@pytest.mark.parametrize("logging_on", [False, True])
def test_a_log_directory_of_the_apps_own_doesnt_keep_the_regular_middleware(monkeypatch, tmp_path, logging_on):
    from aiwaf.core.runtime_config import AIWAFConfig

    config = AIWAFConfig(load_from_env=False)
    config.set("logging_middleware.log_dir", str(tmp_path / "logs"))
    config.set("logging_middleware.enabled", logging_on)
    app, aiwaf = build(monkeypatch, tmp_path, config=config)
    assert aiwaf.native is True
    assert TestClient(app).get("/items").status_code == 200
    # The regular logger writes the app's access log beside the native pipeline.
    assert (tmp_path / "logs" / "access.log").exists() is logging_on
    assert not (tmp_path / "logs" / "aiwaf_native_access.csv").exists()


@pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")
@pytest.mark.parametrize("log_format", ["json", "csv"])
def test_a_chosen_log_format_is_written_by_the_regular_logger_beside_the_native_pipeline(monkeypatch, tmp_path, log_format):
    import json
    from aiwaf.core.runtime_config import AIWAFConfig
    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware
    from aiwaf.fast.middleware.logging_middleware import ASGIRequestLogger as AIWAFLoggingMiddleware

    config = AIWAFConfig(load_from_env=False)
    config.set("logging_middleware.enabled", True)
    config.set("logging_middleware.log_dir", str(tmp_path / "logs"))
    config.set("logging_middleware.log_format", log_format)
    app, aiwaf = build(monkeypatch, tmp_path, config=config)
    assert aiwaf.native is True
    # The regular logger innermost, where the regular chain registers it.
    assert [m.cls for m in app.user_middleware] == [NativeAIWAFMiddleware, AIWAFLoggingMiddleware]
    client = TestClient(app)
    assert client.get("/items").status_code == 200
    lines = (tmp_path / "logs" / "access.log").read_text().splitlines()
    assert not (tmp_path / "logs" / "aiwaf_native_access.csv").exists()
    if log_format == "json":
        assert json.loads(lines[-1])["status_code"] == 200
    else:
        assert lines[0].startswith("timestamp") and len(lines) == 2
