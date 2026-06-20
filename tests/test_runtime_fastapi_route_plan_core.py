from types import SimpleNamespace

from aiwaf.core import runtime_fastapi_decorators


class FakeRequest:
    def __init__(self):
        self.url = SimpleNamespace(path="/health")
        self.scope = {}
        self.state = SimpleNamespace()


def test_route_plan_is_built_once_per_fastapi_request(monkeypatch):
    request = FakeRequest()
    calls = 0
    original = runtime_fastapi_decorators.get_route_execution_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_fastapi_decorators, "get_route_execution_plan", counted)

    assert runtime_fastapi_decorators.should_apply_middleware(request, "rate_limit") is True
    assert runtime_fastapi_decorators.should_apply_middleware(request, "header_validation") is True
    assert calls == 1
