"""AIWAF_PROFILE=monitor for FastAPI lets through what it would block, and logs it."""
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.core import profiles
from aiwaf.core.runtime_config import AIWAFConfig


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    monkeypatch.delenv("AIWAF_PROFILE", raising=False)
    yield
    profiles.activate("balanced")


def statuses(tmp_path, profile, ip, **settings):
    from aiwaf.fast import AIWAF

    config = AIWAFConfig(load_from_env=False)
    config.set("exemptions.private_ips_exempted", False)
    config.set("logging_middleware.log_dir", str(tmp_path))
    config.set("path_manifest", str(tmp_path / "missing.json"))
    for key, value in settings.items():
        config.set(key, value)
    app = FastAPI()
    app.get("/ok")(lambda: {"ok": True})
    AIWAF(app, config=config, native_pipeline=False, profile=profile)
    with TestClient(app, client=(ip, 5000)) as client:
        client.headers.clear()
        return config, [client.get("/.env").status_code for _ in range(2)]


def test_monitor_lets_a_probe_through_and_logs_it(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
        _config, codes = statuses(tmp_path, "monitor", "203.0.113.31")
    assert codes == [404, 404]
    assert "would block GET /.env from 203.0.113.31 with 403 (IPAndKeywordBlockMiddleware" in caplog.text
    assert "would add 203.0.113.31 to the blocklist" in caplog.text


def test_strict_halves_the_rate_limits_it_hasnt_been_given(tmp_path):
    config, codes = statuses(tmp_path, "strict", "203.0.113.32")
    assert codes == [403, 403]
    assert (config.get("rate_limiting.max_requests"), config.get("rate_limiting.flood_threshold")) == (10, 20)
    assert config.get("AIWAF_UUID_SCORE_ENABLED") is True
    config, _codes = statuses(tmp_path, "strict", "203.0.113.33", **{"rate_limiting.max_requests": 7})
    assert config.get("rate_limiting.max_requests") == 7


def test_responses_replaced_after_the_handler_are_undone_and_native_is_refused(tmp_path, caplog):
    import asyncio

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse

    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware
    from aiwaf.fast.monitor import monitored

    class Replacing(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            await call_next(request)
            return JSONResponse({"error": "uuid"}, status_code=403)

    assert monitored(object) is object
    middleware = monitored(Replacing)(app=None)
    request = Request({"type": "http", "method": "GET", "path": "/u", "headers": [], "query_string": b"",
                       "client": ("203.0.113.34", 1)})
    original = PlainTextResponse("ok")

    async def call_next(_request):
        return original

    with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
        assert asyncio.run(middleware.dispatch(request, call_next)) is original
    assert 'would block GET /u from 203.0.113.34 with 403 (Replacing: {"error":"uuid"})' in caplog.text
    profiles.activate("monitor")
    with pytest.raises(RuntimeError, match="AIWAF_PROFILE=monitor runs the regular middleware"):
        NativeAIWAFMiddleware(FastAPI())
