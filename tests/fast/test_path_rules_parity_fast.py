from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF
from aiwaf.fast.middleware.rate_limit_middleware import _AIWAF_CACHE


def test_path_rules_parity_header_disable_on_api(monkeypatch):
    app = FastAPI()

    @app.get("/api/data")
    async def api_data():
        return {"ok": True}

    @app.get("/ui/data")
    async def ui_data():
        return {"ok": True}

    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.core_header_validation.evaluate_header_policy",
        lambda *args, **kwargs: "forced header failure",
    )

    AIWAF(
        app,
        path_rules=[{"PREFIX": "/api/", "DISABLE": ["header_validation"]}],
        rate_limiting={"enabled": False},
        honeypot={"enabled": False},
    )
    client = TestClient(app)

    assert client.get("/api/data", headers={"user-agent": "", "accept": "*/*"}).status_code == 200
    assert client.get("/ui/data", headers={"user-agent": "", "accept": "*/*"}).status_code == 403


def test_path_rules_parity_rate_override_and_specificity():
    app = FastAPI()

    @app.get("/webhooks/public")
    async def wh_public():
        return {"ok": True}

    @app.get("/webhooks/internal/ping")
    async def wh_internal():
        return {"ok": True}

    _AIWAF_CACHE.clear()
    AIWAF(
        app,
        header_validation={"enabled": False},
        honeypot={"enabled": False},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60, "flood_threshold": 100},
        path_rules=[
            {"PREFIX": "/webhooks/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 2, "FLOOD": 100}},
            {"PREFIX": "/webhooks/internal/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 1, "FLOOD": 100}},
        ],
    )
    client = TestClient(app)

    assert client.get("/webhooks/public").status_code == 200
    assert client.get("/webhooks/public").status_code == 200
    assert client.get("/webhooks/public").status_code == 429

    assert client.get("/webhooks/internal/ping").status_code == 200
    assert client.get("/webhooks/internal/ping").status_code == 429

