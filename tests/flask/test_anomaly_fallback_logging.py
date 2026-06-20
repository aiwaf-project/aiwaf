from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask

from aiwaf.flask.anomaly_middleware import AIAnomalyMiddleware
from aiwaf.flask.middleware_logger import AIWAFLoggerMiddleware


def _outcome():
    return SimpleNamespace(updated_history=[], learned_keywords=[], block=False, reason=None)


def test_flask_anomaly_writes_fallback_training_csv(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        AIWAF_LOG_DIR=str(tmp_path),
        AIWAF_MIDDLEWARE_LOGGING=False,
    )

    @app.route("/ok")
    def ok():
        return "ok", 200

    AIAnomalyMiddleware(app)

    with patch("aiwaf.flask.anomaly_middleware.core_evaluate_anomaly", return_value=_outcome()):
        client = app.test_client()
        resp = client.get("/ok", headers={"User-Agent": "pytest-agent"})
        assert resp.status_code == 200

    csv_path = tmp_path / "aiwaf_requests.csv"
    assert csv_path.exists()
    content = csv_path.read_text(encoding="utf-8")
    assert "pytest-agent" in content
    assert "/ok" in content


def test_flask_anomaly_skips_fallback_when_logger_enabled(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        AIWAF_LOG_DIR=str(tmp_path),
        AIWAF_MIDDLEWARE_LOGGING=True,
    )

    @app.route("/ok")
    def ok():
        return "ok", 200

    AIWAFLoggerMiddleware(app)
    AIAnomalyMiddleware(app)

    with patch("aiwaf.flask.anomaly_middleware.core_evaluate_anomaly", return_value=_outcome()):
        client = app.test_client()
        resp = client.get("/ok")
        assert resp.status_code == 200

    # Fallback file should not be created by anomaly middleware when logger is active.
    assert not (tmp_path / "aiwaf_requests.csv").exists()
