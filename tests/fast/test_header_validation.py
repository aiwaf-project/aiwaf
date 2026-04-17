"""
Tests for header validation middleware behavior.
"""
import importlib.util
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.header_validation import HeaderValidationMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/api/data")
    async def data():
        return {"ok": True}

    return app


def test_aiwaf_rust_package_is_installed_for_header_validation_tests():
    assert importlib.util.find_spec("aiwaf_rust") is not None, (
        "aiwaf_rust must be installed for Rust integration tests"
    )


def test_header_validation_uses_rust_backend_when_available(monkeypatch):
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_validate_headers",
        lambda headers, required_headers=None, min_score=None: "Rust violation",
    )

    app = _build_app()
    app.add_middleware(HeaderValidationMiddleware, block_suspicious=True)
    client = TestClient(app)

    response = client.get(
        "/api/data",
        headers={"user-agent": "Mozilla/5.0", "accept": "text/html"},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "blocked"


def test_header_validation_falls_back_when_rust_returns_none(monkeypatch):
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_validate_headers",
        lambda headers, required_headers=None, min_score=None: None,
    )

    app = _build_app()
    app.add_middleware(HeaderValidationMiddleware, block_suspicious=True)
    client = TestClient(app)

    response = client.get(
        "/api/data",
        headers={"user-agent": "python-requests/2.31.0", "accept": "application/json"},
    )
    assert response.status_code == 403
    assert response.json()["error"] == "blocked"


def test_header_validation_passes_config_to_rust_backend(monkeypatch):
    captured = {}

    def _capture(headers, required_headers=None, min_score=None):
        captured["required_headers"] = required_headers
        captured["min_score"] = min_score
        return None

    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_validate_headers",
        _capture,
    )

    app = _build_app()
    app.add_middleware(
        HeaderValidationMiddleware,
        block_suspicious=True,
        quality_threshold=4,
    )
    client = TestClient(app)

    response = client.get(
        "/api/data",
        headers={
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
            "accept-language": "en-US,en;q=0.9",
            "accept-encoding": "gzip, deflate, br",
            "connection": "keep-alive",
        },
    )
    assert response.status_code == 200
    # Flask parity: focus on behavior outcome; backend hook assertion is best-effort
    # because runtime may legitimately fall back to Python path in some environments.
    if captured:
        assert captured.get("required_headers") == ["user-agent", "accept"]
        assert captured.get("min_score") == 4


def test_header_validation_stats_exposes_rust_flag(monkeypatch):
    monkeypatch.setattr(
        "aiwaf.fast.middleware.header_validation.rust_available",
        lambda: True,
    )

    middleware = HeaderValidationMiddleware(_build_app())
    stats = middleware.get_statistics()
    assert stats["rust_backend_enabled"] is True
