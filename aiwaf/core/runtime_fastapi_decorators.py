"""FastAPI decorators and middleware gating helpers for route-level exemptions."""

import inspect
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Set

from aiwaf.core.exemptions import (
    get_path_rule_overrides_for_path as core_get_path_rule_overrides_for_path,
    is_middleware_disabled_for_path as core_is_middleware_disabled_for_path,
)
from aiwaf.core.route_plan import get_route_execution_plan

ALL_MIDDLEWARES = {
    "ip_keyword_block",
    "rate_limit",
    "honeypot",
    "header_validation",
    "geo_block",
    "ai_anomaly",
    "uuid_tamper",
    "logging",
}


def _mark_endpoint(func, *, fully_exempt: bool, exempt_middlewares: Set[str], required_middlewares: Optional[Set[str]] = None):
    func.aiwaf_exempt = bool(fully_exempt)
    func._aiwaf_exempt = bool(fully_exempt)
    func._aiwaf_exempt_middlewares = set(exempt_middlewares)
    func._aiwaf_required_middlewares = set(required_middlewares or set())
    return func


def aiwaf_exempt(endpoint_func):
    """
    Decorator to exempt a FastAPI endpoint from AIWAF middleware protection.

    This mirrors Django/Flask behavior by marking the wrapped endpoint with
    both ``aiwaf_exempt`` and ``_aiwaf_exempt`` attributes for compatibility.
    """

    @wraps(endpoint_func)
    async def wrapped_endpoint(*args, **kwargs):
        if inspect.iscoroutinefunction(endpoint_func):
            return await endpoint_func(*args, **kwargs)
        return endpoint_func(*args, **kwargs)

    return _mark_endpoint(wrapped_endpoint, fully_exempt=True, exempt_middlewares=set())


def aiwaf_exempt_from(*middleware_names):
    """Exempt a route from selected middleware names."""
    selected = {str(name).strip().lower() for name in middleware_names if name}

    def decorator(endpoint_func):
        @wraps(endpoint_func)
        async def wrapped_endpoint(*args, **kwargs):
            if inspect.iscoroutinefunction(endpoint_func):
                return await endpoint_func(*args, **kwargs)
            return endpoint_func(*args, **kwargs)

        return _mark_endpoint(wrapped_endpoint, fully_exempt=False, exempt_middlewares=selected)

    return decorator


def aiwaf_only(*middleware_names):
    """Apply only selected middlewares to a route (exempt all others)."""
    selected = {str(name).strip().lower() for name in middleware_names if name}
    exempt_middlewares = ALL_MIDDLEWARES - selected
    return aiwaf_exempt_from(*exempt_middlewares)


def aiwaf_require_protection(*middleware_names):
    """Require specific middlewares for a route even if exemptions would skip them."""
    required = {str(name).strip().lower() for name in middleware_names if name}

    def decorator(endpoint_func):
        @wraps(endpoint_func)
        async def wrapped_endpoint(*args, **kwargs):
            if inspect.iscoroutinefunction(endpoint_func):
                return await endpoint_func(*args, **kwargs)
            return endpoint_func(*args, **kwargs)

        return _mark_endpoint(
            wrapped_endpoint,
            fully_exempt=False,
            exempt_middlewares=set(),
            required_middlewares=required,
        )

    return decorator


def _endpoint_from_request(request) -> Any:
    if not hasattr(request, "scope"):
        return None
    return request.scope.get("endpoint")


def _is_path_rule_disabled(request, middleware_name: str, path_rules: Optional[Iterable[Dict[str, Any]]]) -> bool:
    if not path_rules:
        return False
    path = getattr(request.url, "path", "")
    return core_is_middleware_disabled_for_path(path, path_rules, middleware_name)


def get_path_rule_overrides(request, key: str, path_rules: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Fetch PATH_RULES override block for the request path."""
    if not path_rules:
        return {}
    if str(key).upper() == "RATE_LIMIT":
        return _get_request_route_plan(request, path_rules).get_rate_limit_overrides()
    path = getattr(request.url, "path", "")
    return core_get_path_rule_overrides_for_path(path, path_rules, key)


def should_apply_middleware(request, middleware_name: str, path_rules: Optional[Iterable[Dict[str, Any]]] = None) -> bool:
    """Decide whether middleware should run for this request."""
    return _get_request_route_plan(request, path_rules).should_apply(middleware_name)


def _get_request_route_plan(request, path_rules: Optional[Iterable[Dict[str, Any]]] = None):
    endpoint = _endpoint_from_request(request)
    path = getattr(request.url, "path", "")
    rules = path_rules or []
    app = (getattr(request, "scope", {}) or {}).get("app")
    app_state = getattr(app, "state", None)
    policy_version = getattr(app_state, "aiwaf_route_plan_version", 0)

    required = set()
    fully_exempt = False
    exempt_middlewares = set()
    if endpoint is not None:
        required = getattr(endpoint, "_aiwaf_required_middlewares", set()) or set()
        fully_exempt = bool(getattr(endpoint, "aiwaf_exempt", False) or getattr(endpoint, "_aiwaf_exempt", False))
        exempt_middlewares = getattr(endpoint, "_aiwaf_exempt_middlewares", set()) or set()

    request_key = (
        path,
        id(rules),
        repr(policy_version),
        fully_exempt,
        frozenset(exempt_middlewares),
        frozenset(required),
    )
    state = getattr(request, "state", None)
    if state is not None and getattr(state, "_aiwaf_route_plan_key", None) == request_key:
        return state._aiwaf_route_plan

    plan = get_route_execution_plan(
        path,
        rules,
        policy_version=policy_version,
        fully_exempt=fully_exempt,
        exempt_middlewares=exempt_middlewares,
        required_middlewares=required,
    )
    if state is not None:
        state._aiwaf_route_plan_key = request_key
        state._aiwaf_route_plan = plan
    return plan
