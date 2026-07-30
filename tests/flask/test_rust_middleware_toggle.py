from pathlib import Path
import importlib.util

import pytest
from flask import Flask

from aiwaf.flask import AIWAF
from aiwaf.core import rust_backend

RUST_PACKAGE_INSTALLED = importlib.util.find_spec("aiwaf_rust") is not None


def _make_app(tmp_path: Path) -> Flask:
    app = Flask(__name__)
    app.config["AIWAF_ENABLE_LOGGING"] = True
    app.config["AIWAF_LOG_FORMAT"] = "csv"
    app.config["AIWAF_LOG_DIR"] = str(tmp_path)
    app.config["AIWAF_DATA_DIR"] = str(tmp_path)
    app.config["AIWAF_USE_RUST"] = True
    app.config["AIWAF_USE_CSV"] = True

    @app.route("/")
    def index():
        return "ok"

    AIWAF(app)
    return app


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
def test_header_validation_uses_rust_when_enabled(monkeypatch, tmp_path):
    called = {"value": False}

    def fake_validate(headers, required_headers=None, min_score=None):
        called["value"] = True
        return None

    monkeypatch.setattr(rust_backend, "rust_available", lambda: True)
    monkeypatch.setattr(rust_backend, "validate_headers", fake_validate)

    app = _make_app(tmp_path)
    with app.test_client() as client:
        client.get("/", headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})

    assert called["value"] is True


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
def test_header_validation_uses_installed_rust_when_csv_disabled(monkeypatch, tmp_path):
    called = {"value": False}

    def fake_validate(headers, required_headers=None, min_score=None):
        called["value"] = True
        return None

    monkeypatch.setattr(rust_backend, "rust_available", lambda: True)
    monkeypatch.setattr(rust_backend, "validate_headers", fake_validate)

    app = _make_app(tmp_path)
    app.config["AIWAF_USE_CSV"] = False
    with app.test_client() as client:
        client.get("/", headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})

    assert called["value"] is True


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
def test_header_validation_head_override_skips_required_headers(monkeypatch, tmp_path):
    called = {"value": False}

    def fake_validate(headers, required_headers=None, min_score=None):
        called["value"] = True
        return None

    monkeypatch.setattr(rust_backend, "rust_available", lambda: True)
    monkeypatch.setattr(rust_backend, "validate_headers", fake_validate)

    app = _make_app(tmp_path)
    app.config["AIWAF_REQUIRED_HEADERS"] = {"HEAD": []}

    with app.test_client() as client:
        response = client.head("/")

    assert response.status_code == 200
    # Method override uses Python path to honor per-method required-header config.
    assert called["value"] is False
