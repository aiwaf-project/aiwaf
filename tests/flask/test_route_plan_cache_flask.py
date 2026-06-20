from flask import Flask

from aiwaf.flask import exemption_decorators


def test_route_plan_is_built_once_per_flask_request(monkeypatch):
    app = Flask(__name__)
    calls = 0
    original = exemption_decorators.get_route_execution_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(exemption_decorators, "get_route_execution_plan", counted)

    @app.before_request
    def check_multiple_middlewares():
        assert exemption_decorators.should_apply_middleware("rate_limit") is True
        assert exemption_decorators.should_apply_middleware("header_validation") is True

    @app.get("/health")
    def health():
        return {"status": "ok"}

    with app.test_client() as client:
        assert client.get("/health").status_code == 200

    assert calls == 1
