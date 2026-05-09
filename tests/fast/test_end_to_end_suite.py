from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch

from aiwaf.fast import AIWAF
from aiwaf.core.honeypot import ACTION_PAGE_EXPIRED, evaluate_form_timing


def test_end_to_end_rate_limit_behavior():
    app = FastAPI()

    @app.get("/path-a")
    async def path_a():
        return {"a": True}

    @app.get("/path-b")
    async def path_b():
        return {"b": True}

    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60},
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    assert client.get("/path-a").status_code == 200
    assert client.get("/path-a").status_code == 429
    # New default behavior is path-scoped rate limiting without soft-limit
    # blacklist escalation, so a different path remains allowed.
    assert client.get("/path-b").status_code == 200
    assert client.get("/path-b").status_code == 429


def test_end_to_end_rate_limit_legacy_ip_mode():
    app = FastAPI()

    @app.get("/path-a")
    async def path_a():
        return {"a": True}

    @app.get("/path-b")
    async def path_b():
        return {"b": True}

    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60, "key_mode": "ip"},
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    assert client.get("/path-a").status_code == 200
    assert client.get("/path-a").status_code == 429
    assert client.get("/path-b").status_code == 429


def test_end_to_end_rate_limit_legacy_soft_blacklist_toggle():
    app = FastAPI()

    @app.get("/path-a")
    async def path_a():
        return {"a": True}

    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={
            "enabled": True,
            "max_requests": 1,
            "window_seconds": 60,
            "soft_block_blacklist": True,
        },
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    assert client.get("/path-a").status_code == 200
    assert client.get("/path-a").status_code == 429
    # Legacy soft blacklist mode should escalate to a persistent block.
    assert client.get("/path-a").status_code == 403


def test_honeypot_policy_page_expired_decision():
    decision = evaluate_form_timing(
        now=500.0,
        get_time=0.0,
        path="/form-expired",
        min_form_time=1.0,
        max_page_time=240.0,
    )
    assert decision.action == ACTION_PAGE_EXPIRED
    assert decision.status_code == 409


def test_honeypot_get_to_obvious_post_only_returns_405():
    app = FastAPI()

    @app.post("/api/create/")
    async def create_only():
        return {"ok": True}

    AIWAF(app, header_validation={"enabled": False}, rate_limiting={"enabled": False}, honeypot={"enabled": True})
    client = TestClient(app)
    assert client.get("/api/create/").status_code == 405


def test_honeypot_post_to_get_only_returns_405():
    app = FastAPI()
    from aiwaf.fast.middleware.honeypot_timing_middleware import _AIWAF_CACHE
    _AIWAF_CACHE.clear()

    @app.get("/read-only")
    async def read_only():
        return {"ok": True}

    AIWAF(app, header_validation={"enabled": False}, rate_limiting={"enabled": False}, honeypot={"enabled": True})
    client = TestClient(app)
    assert client.post("/read-only").status_code == 405
