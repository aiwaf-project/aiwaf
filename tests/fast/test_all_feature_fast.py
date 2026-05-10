from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def _middleware_names(app: FastAPI):
    return [mw.cls.__name__ for mw in app.user_middleware]


def test_all_auto_enables_logging_when_access_log_missing():
    app = FastAPI()
    AIWAF(app, middlewares=["all"], AIWAF_ACCESS_LOG="")
    assert "AIWAFLoggingMiddleware" in _middleware_names(app)


def test_all_auto_disables_logging_when_access_log_configured():
    app = FastAPI()
    AIWAF(app, middlewares=["all"], AIWAF_ACCESS_LOG="/var/log/access.log")
    assert "AIWAFLoggingMiddleware" not in _middleware_names(app)


def test_all_auto_enables_geo_when_block_countries_present(monkeypatch):
    app = FastAPI()

    @app.get("/geo")
    async def geo():
        return {"ok": True}

    monkeypatch.setattr(
        "aiwaf.fast.middleware.geo_block_middleware.should_apply_middleware",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "aiwaf.fast.middleware.geo_block_middleware.is_exempt",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr("aiwaf.fast.middleware.geo_block_middleware.get_country_for_ip", lambda *_a, **_k: "US")
    AIWAF(
        app,
        middlewares=["all"],
        header_validation={"enabled": False},
        rate_limiting={"enabled": False},
        honeypot={"enabled": False},
        geo_block={"enabled": False, "block_countries": ["US"]},
    )
    assert "GeoBlockMiddleware" in _middleware_names(app)


def test_all_auto_disables_uuid_tamper_when_no_uuid_routes():
    app = FastAPI()

    @app.get("/plain")
    async def plain():
        return {"ok": True}

    AIWAF(app, middlewares=["all"], header_validation={"enabled": False}, rate_limiting={"enabled": False})
    assert "UUIDTamperMiddleware" not in _middleware_names(app)


def test_all_auto_enables_uuid_tamper_when_uuid_routes_present():
    app = FastAPI()

    @app.get("/items/{uuid}")
    async def item(uuid: str):
        return {"ok": True}

    AIWAF(app, middlewares=["all"], header_validation={"enabled": False}, rate_limiting={"enabled": False})
    assert "UUIDTamperMiddleware" in _middleware_names(app)
