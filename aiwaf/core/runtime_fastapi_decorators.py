"""FastAPI decorators and middleware gating helpers for route-level exemptions."""

import inspect
from functools import wraps
from typing import Any, Dict, Iterable, Optional, Set

from aiwaf.core.exemptions import get_path_rule_for_path as core_get_path_rule_for_path

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
    rule = core_get_path_rule_for_path(path, path_rules)
    if not rule:
        return False
    disabled = rule.get("DISABLE", []) or []
    if not isinstance(disabled, (list, tuple, set)):
        return False
    target = middleware_name.lower()
    return any(str(item).strip().lower() == target for item in disabled if item)


def get_path_rule_overrides(request, key: str, path_rules: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Fetch PATH_RULES override block for the request path."""
    if not path_rules:
        return {}
    path = getattr(request.url, "path", "")
    rule = core_get_path_rule_for_path(path, path_rules)
    if not rule:
        return {}
    value = rule.get(key, {}) or rule.get(key.lower(), {}) or {}
    return value if isinstance(value, dict) else {}


def should_apply_middleware(request, middleware_name: str, path_rules: Optional[Iterable[Dict[str, Any]]] = None) -> bool:
    """Decide whether middleware should run for this request."""
    middleware_name = (middleware_name or "").strip().lower()
    endpoint = _endpoint_from_request(request)

    if endpoint is not None:
        required = getattr(endpoint, "_aiwaf_required_middlewares", set()) or set()
        if middleware_name in {str(item).strip().lower() for item in required if item}:
            return True

    if _is_path_rule_disabled(request, middleware_name, path_rules):
        return False

    if endpoint is not None:
        if getattr(endpoint, "aiwaf_exempt", False) or getattr(endpoint, "_aiwaf_exempt", False):
            return False
        exempt_middlewares = getattr(endpoint, "_aiwaf_exempt_middlewares", set()) or set()
        normalized = {str(item).strip().lower() for item in exempt_middlewares if item}
        if middleware_name in normalized:
            return False

    return True
