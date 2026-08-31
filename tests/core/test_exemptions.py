from aiwaf.core.exemptions import (
    get_path_rule_overrides_for_path,
    is_middleware_disabled_for_path,
    should_apply_middleware_for_path,
)


def test_path_rules_longest_prefix_and_disable():
    rules = [
        {"PREFIX": "/api/", "DISABLE": ["header_validation"]},
        {"PREFIX": "/api/v1/", "DISABLE": ["rate_limit"]},
    ]
    assert is_middleware_disabled_for_path("/api/v1/users", rules, "rate_limit") is True
    assert is_middleware_disabled_for_path("/api/v1/users", rules, "header_validation") is False


def test_path_rules_disable_accepts_class_names():
    rules = [{"PREFIX": "/v1/", "DISABLE": ["HeaderValidationMiddleware"]}]
    assert is_middleware_disabled_for_path("/v1/ping", rules, "header_validation") is True


def test_path_rules_overrides_case_fallback():
    rules = [{"PREFIX": "/api/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 10}}]
    overrides = get_path_rule_overrides_for_path("/api/data", rules, "RATE_LIMIT")
    assert overrides["WINDOW"] == 60
    assert overrides["MAX"] == 10


def test_should_apply_precedence_required_over_exempt_and_rules():
    rules = [{"PREFIX": "/api/", "DISABLE": ["rate_limit"]}]
    assert (
        should_apply_middleware_for_path(
            "/api/data",
            rules,
            "rate_limit",
            fully_exempt=True,
            exempt_middlewares={"rate_limit"},
            required_middlewares={"rate_limit"},
        )
        is True
    )


def test_should_apply_respects_path_rule_then_exemptions():
    rules = [{"PREFIX": "/api/", "DISABLE": ["HeaderValidationMiddleware"]}]
    assert should_apply_middleware_for_path("/api/data", rules, "header_validation") is False
    assert should_apply_middleware_for_path("/web/data", [], "header_validation", fully_exempt=True) is False
    assert (
        should_apply_middleware_for_path(
            "/web/data", [], "header_validation", exempt_middlewares={"header_validation"}
        )
        is False
    )
