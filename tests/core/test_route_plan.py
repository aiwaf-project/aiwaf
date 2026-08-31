from aiwaf.core.route_plan import RoutePlanCache, RoutePolicyCache, get_route_execution_plan


def test_route_plan_is_reused_for_identical_policy_inputs():
    rules = [{"PREFIX": "/api/", "DISABLE": ["header_validation"]}]

    first = get_route_execution_plan("/api/items", rules)
    second = get_route_execution_plan("/api/items", rules)

    assert first is second
    assert first.should_apply("header_validation") is False
    assert first.should_apply("rate_limit") is True


def test_route_plan_changes_when_rules_change():
    first = get_route_execution_plan(
        "/api/items",
        [{"PREFIX": "/api/", "DISABLE": ["header_validation"]}],
    )
    second = get_route_execution_plan(
        "/api/items",
        [{"PREFIX": "/api/", "DISABLE": ["rate_limit"]}],
    )

    assert first is not second
    assert first.should_apply("header_validation") is False
    assert second.should_apply("header_validation") is True
    assert second.should_apply("rate_limit") is False


def test_route_plan_preserves_required_middleware_precedence():
    plan = get_route_execution_plan(
        "/api/items",
        [{"PREFIX": "/api/", "DISABLE": ["rate_limit"]}],
        fully_exempt=True,
        exempt_middlewares={"rate_limit"},
        required_middlewares={"rate_limit"},
    )

    assert plan.should_apply("rate_limit") is True
    assert plan.should_apply("header_validation") is False


def test_route_plan_caches_rate_limit_overrides_without_exposing_mutable_state():
    plan = get_route_execution_plan(
        "/api/items",
        [{"PREFIX": "/api/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 10}}],
    )

    first = plan.get_rate_limit_overrides()
    first["MAX"] = 999

    assert plan.get_rate_limit_overrides() == {"WINDOW": 60, "MAX": 10}


def test_route_plan_cache_is_bounded():
    cache = RoutePlanCache(maxsize=2)
    plans = [get_route_execution_plan(f"/item/{index}", []) for index in range(3)]

    for index, plan in enumerate(plans):
        cache.get_or_create(index, lambda plan=plan: plan)

    assert len(cache) == 2


def test_route_policy_compiles_once_for_same_rules_and_version():
    cache = RoutePolicyCache()
    rules = [{"PREFIX": "/api/", "DISABLE": ["header_validation"]}]

    first = cache.get_or_compile(rules, version=1)
    second = cache.get_or_compile(rules, version=1)
    changed_version = cache.get_or_compile(rules, version=2)

    assert first is second
    assert changed_version is not first


def test_route_plan_version_invalidates_in_place_rule_changes():
    rules = [{"PREFIX": "/api/", "DISABLE": ["header_validation"]}]
    first = get_route_execution_plan("/api/items", rules, policy_version=1)

    rules[0]["DISABLE"] = ["rate_limit"]
    unchanged_version = get_route_execution_plan("/api/items", rules, policy_version=1)
    changed_version = get_route_execution_plan("/api/items", rules, policy_version=2)

    assert unchanged_version is first
    assert changed_version is not first
    assert changed_version.should_apply("header_validation") is True
    assert changed_version.should_apply("rate_limit") is False
def test_clear_route_plan_cache_clears_both_layers():
    from aiwaf.core.route_plan import (
        RoutePlanCache,
        RoutePolicyCache,
        clear_route_plan_cache,
    )

    policy = RoutePolicyCache()
    policy._policies["x"] = object()
    policy.clear()
    assert policy._policies == {}
    plans = RoutePlanCache()
    plans._plans["x"] = object()
    plans.clear()
    assert plans._plans == {}
    clear_route_plan_cache()
