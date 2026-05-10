from flask import Flask

from aiwaf.flask import AIWAF
from aiwaf.flask.rate_limit_middleware import _aiwaf_cache


def _create_app(tmp_path, middlewares=None, config=None):
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "AIWAF_USE_CSV": True,
            "AIWAF_DATA_DIR": str(tmp_path),
            "AIWAF_EXEMPT_PATHS": set(),
            "AIWAF_ENABLE_LOGGING": False,
            "AIWAF_RATE_WINDOW": 60,
            "AIWAF_RATE_MAX": 1,
            "AIWAF_RATE_FLOOD": 100,
        }
    )
    if config:
        app.config.update(config)
    AIWAF(app, middlewares=middlewares)
    return app


def test_path_rules_parity_header_disable_on_api(tmp_path, monkeypatch):
    app = _create_app(
        tmp_path,
        middlewares=["header_validation"],
        config={"AIWAF_PATH_RULES": [{"PREFIX": "/api/", "DISABLE": ["HeaderValidationMiddleware"]}]},
    )

    @app.route("/api/data")
    def api_data():
        return "ok"

    @app.route("/ui/data")
    def ui_data():
        return "ok"

    monkeypatch.setattr(
        "aiwaf.flask.header_validation_middleware.header_validation.evaluate_header_policy",
        lambda *args, **kwargs: "forced header failure",
    )

    client = app.test_client()
    assert client.get("/api/data", headers={"User-Agent": ""}).status_code == 200
    assert client.get("/ui/data", headers={"User-Agent": ""}).status_code == 403


def test_path_rules_parity_rate_override_and_specificity(tmp_path):
    app = _create_app(
        tmp_path,
        middlewares=["rate_limit"],
        config={
            "AIWAF_PATH_RULES": [
                {"PREFIX": "/webhooks/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 2, "FLOOD": 100}},
                {"PREFIX": "/webhooks/internal/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 1, "FLOOD": 100}},
            ]
        },
    )

    @app.route("/webhooks/public")
    def wh_public():
        return "ok"

    @app.route("/webhooks/internal/ping")
    def wh_internal():
        return "ok"

    _aiwaf_cache.clear()
    client = app.test_client()
    headers = {"User-Agent": "Test Browser 1.0"}

    assert client.get("/webhooks/public", headers=headers).status_code == 200
    assert client.get("/webhooks/public", headers=headers).status_code == 200
    assert client.get("/webhooks/public", headers=headers).status_code == 429

    assert client.get("/webhooks/internal/ping", headers=headers).status_code == 200
    assert client.get("/webhooks/internal/ping", headers=headers).status_code == 429

