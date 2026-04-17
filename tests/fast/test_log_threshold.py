from aiwaf.fast.middleware.anomaly_middleware import AIAnomalyMiddleware


def test_ai_anomaly_middleware_accepts_enabled_flag():
    middleware = AIAnomalyMiddleware(app=lambda scope, receive, send: None, enabled=False)
    assert middleware.enabled is False
