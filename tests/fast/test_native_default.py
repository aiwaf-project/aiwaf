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
    assert [m.cls for m in app.user_middleware] == [NativeAIWAFMiddleware]
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
