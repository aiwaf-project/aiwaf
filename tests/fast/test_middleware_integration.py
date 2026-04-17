from types import SimpleNamespace

from aiwaf.fast.decorators import should_apply_middleware


def _request(path: str):
    return SimpleNamespace(url=SimpleNamespace(path=path), scope={})


def test_should_apply_middleware_path_rule_disable():
    request = _request("/api/health")
    rules = [{"PREFIX": "/api/", "DISABLE": ["header_validation"]}]
    assert should_apply_middleware(request, "header_validation", rules) is False
    assert should_apply_middleware(request, "rate_limit", rules) is True

