from types import SimpleNamespace

from aiwaf.core import runtime_fastapi_decorators
import asyncio


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


def test_all_route_decorators_execute_sync_and_async_endpoints():
    async def async_endpoint(value):
        return value

    def sync_endpoint(value):
        return value

    selective = runtime_fastapi_decorators.aiwaf_exempt_from("rate_limit")(sync_endpoint)
    assert asyncio.run(selective("ok")) == "ok"
    assert selective._aiwaf_exempt_middlewares == {"rate_limit"}

    only = runtime_fastapi_decorators.aiwaf_only("logging")(sync_endpoint)
    assert asyncio.run(only("ok")) == "ok"
    assert "logging" not in only._aiwaf_exempt_middlewares

    required = runtime_fastapi_decorators.aiwaf_require_protection("geo_block")(async_endpoint)
    assert asyncio.run(required("ok")) == "ok"
    assert required._aiwaf_required_middlewares == {"geo_block"}

    request = FakeRequest()
    assert not runtime_fastapi_decorators._is_path_rule_disabled(request, "logging", None)
    assert runtime_fastapi_decorators._is_path_rule_disabled(
        request,
        "logging",
        [{"PREFIX": "/health", "DISABLE": ["logging"]}],
    )
