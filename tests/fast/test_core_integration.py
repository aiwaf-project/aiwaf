"""Comprehensive tests for AIWAF core integration and FastAPI runtime behavior."""

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.core.runtime_blacklist import BlacklistManager
from aiwaf.core import AIWAF


BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "connection": "keep-alive",
}


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/data")
    async def data():
        return {"ok": True}

    return app


def test_aiwaf_integrates_with_fastapi_and_allows_valid_request(monkeypatch):
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_available",
        lambda: False,
    )

    app = _build_app()
    AIWAF(app)
    client = TestClient(app)

    response = client.get("/api/data", headers=BROWSER_HEADERS)
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_rate_limiting_blocks_and_blacklists_ip():
    app = _build_app()
    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60},
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    first = client.get("/api/data")
    second = client.get("/api/data")

    assert first.status_code == 200
    assert second.status_code == 429
    assert BlacklistManager.is_blocked("testclient")


def test_lifespan_registration_skips_if_already_present():
    app = _build_app()

    @asynccontextmanager
    async def existing_lifespan(_app):
        yield

    app.router.lifespan_context = existing_lifespan
    AIWAF(app)

    assert app.router.lifespan_context is existing_lifespan


def test_core_exemption_and_block_wrappers_work():
    app = _build_app()
    aiwaf = AIWAF(app, rate_limiting={"enabled": False})

    aiwaf.add_exemption("203.0.113.10", "trusted")
    assert aiwaf.is_exempted("203.0.113.10")

    blocked = aiwaf.block_ip("203.0.113.10", "should not block exempt")
    assert blocked is False

    assert aiwaf.remove_exemption("203.0.113.10") is True
    assert aiwaf.block_ip("203.0.113.10", "manual block") is True
    assert aiwaf.is_blocked("203.0.113.10") is True
    assert aiwaf.unblock_ip("203.0.113.10") is True


def test_core_statistics_activity_and_export_have_expected_shape():
    app = _build_app()
    aiwaf = AIWAF(app, rate_limiting={"enabled": False})
    aiwaf.block_ip("198.51.100.8", "test reason", duration=60)

    stats = aiwaf.get_statistics()
    assert "aiwaf" in stats
    assert "blacklist" in stats
    assert "configuration" in stats

    activity = aiwaf.get_recent_activity(hours=1)
    assert "recent_blocks" in activity
    assert "summary" in activity

    exported = aiwaf.export_data()
    assert "configuration" in exported
    assert "statistics" in exported
    assert "recent_activity" in exported
    assert "health" in exported


def test_core_health_check_reports_healthy_components():
    app = _build_app()
    aiwaf = AIWAF(app, rate_limiting={"enabled": False})

    health = aiwaf.health_check()
    assert health["status"] in {"healthy", "degraded"}
    assert health["components"]["storage"] == "healthy"
    assert health["components"]["blacklist"] == "healthy"
    assert health["components"]["exemptions"] == "healthy"

    cleanup = aiwaf.cleanup()
    assert "cleaned_blocks" in cleanup


def test_update_config_validates_and_rejects_invalid_values():
    aiwaf = AIWAF(_build_app(), rate_limiting={"enabled": False})

    aiwaf.update_config({"rate_limiting": {"max_requests": 150}})
    assert aiwaf.config.get("rate_limiting.max_requests") == 150

    with pytest.raises(ValueError):
        aiwaf.update_config({"rate_limiting": {"max_requests": -1}})


def test_core_repr_mentions_enabled_features_and_storage():
    aiwaf = AIWAF(
        _build_app(),
        header_validation={"enabled": True},
        rate_limiting={"enabled": False},
    )

    rep = repr(aiwaf)
    assert "AIWAF(" in rep
    assert "header_validation" in rep
    assert "storage=memory" in rep
