from types import SimpleNamespace

from aiwaf.fast.decorators import get_path_rule_overrides, should_apply_middleware


def _request(path: str):
    return SimpleNamespace(url=SimpleNamespace(path=path), scope={})


def test_path_rules_disable_and_override_reading():
    rules = [
        {
            "PREFIX": "/myapp/api/",
            "DISABLE": ["rate_limit"],
            "HEADER_VALIDATION": {"quality_threshold": 1},
        }
    ]
    request = _request("/myapp/api/data")

    assert should_apply_middleware(request, "rate_limit", rules) is False
    assert get_path_rule_overrides(request, "HEADER_VALIDATION", rules)["quality_threshold"] == 1

