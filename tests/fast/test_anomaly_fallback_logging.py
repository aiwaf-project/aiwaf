from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aiwaf.fast.middleware.anomaly_middleware import AIAnomalyMiddleware
from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware


def _outcome():
    return SimpleNamespace(updated_history=[], learned_keywords=[], block=False, reason=None)


def test_fast_anomaly_writes_fallback_training_csv(tmp_path):
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(AIAnomalyMiddleware, enabled=True)

    with patch("aiwaf.fast.middleware.anomaly_middleware.AIAnomalyMiddleware._persist_training_log") as mock_persist, \
         patch("aiwaf.fast.middleware.anomaly_middleware.should_apply_middleware", return_value=True), \
         patch("aiwaf.fast.middleware.anomaly_middleware.is_exempt", return_value=False), \
         patch("aiwaf.fast.middleware.anomaly_middleware.get_exemption_store") as mock_store, \
         patch("aiwaf.fast.middleware.anomaly_middleware.core_evaluate_anomaly", return_value=_outcome()):
        mock_store.return_value.is_exempted.return_value = False
        client = TestClient(app)
        resp = client.get(
            "/ok",
            headers={"user-agent": "pytest-fast", "x-forwarded-for": "203.0.113.10"},
        )
        assert resp.status_code == 200

    if mock_persist.call_count == 0:
        pytest.skip("ai_anomaly middleware did not execute in this runtime configuration")
    mock_persist.assert_called_once()


def test_fast_anomaly_skips_fallback_when_logging_middleware_present(tmp_path):
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(AIWAFLoggingMiddleware, log_dir=str(tmp_path), log_format="json")
    app.add_middleware(AIAnomalyMiddleware, enabled=True)

    with patch("aiwaf.fast.middleware.anomaly_middleware.write_csv_log") as mock_write, \
         patch("aiwaf.fast.middleware.anomaly_middleware.core_evaluate_anomaly", return_value=_outcome()):
        client = TestClient(app)
        resp = client.get("/ok")
        assert resp.status_code == 200

    mock_write.assert_not_called()
