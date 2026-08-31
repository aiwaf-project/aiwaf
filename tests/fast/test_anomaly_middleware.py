from aiwaf.fast.middleware.anomaly_middleware import AIAnomalyMiddleware
from types import SimpleNamespace


def test_ai_anomaly_middleware_has_enabled_flag():
    middleware = AIAnomalyMiddleware(app=lambda scope, receive, send: None, enabled=True)
    assert middleware.enabled is True
def test_path_exists_uses_known_route_templates():
    from aiwaf.fast.middleware.anomaly_middleware import AIAnomalyMiddleware

    middleware = AIAnomalyMiddleware.__new__(AIAnomalyMiddleware)
    middleware._route_paths = {"/health"}
    assert middleware._path_exists("/health?ready=1")
    assert not middleware._path_exists("/missing")


def test_segment_malicious_context_rules():
    from aiwaf.fast.middleware.anomaly_middleware import _segment_has_malicious_context

    assert not _segment_has_malicious_context("/safe", "", "")
    assert _segment_has_malicious_context("/../secret", "", "secret")
    assert _segment_has_malicious_context("/search", "q=union", "union")
    assert _segment_has_malicious_context("/.env", "", ".env")


def test_persist_training_log_writes_csv_when_logging_is_absent(tmp_path):
    middleware = AIAnomalyMiddleware.__new__(AIAnomalyMiddleware)
    config = {"logging_middleware.log_dir": str(tmp_path)}
    app = SimpleNamespace(user_middleware=[], state=SimpleNamespace(aiwaf_config=config))
    request = SimpleNamespace(
        app=app,
        method="GET",
        url=SimpleNamespace(path="/health"),
        headers={"user-agent": "pytest"},
    )
    response = SimpleNamespace(status_code=200, headers={})
    middleware._persist_training_log(request, response, "203.0.113.160", 0.01)
    assert "203.0.113.160" in (tmp_path / "aiwaf_requests.csv").read_text(encoding="utf-8")
