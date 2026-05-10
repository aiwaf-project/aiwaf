"""Shared request-method validation policy."""

from dataclasses import dataclass
from typing import Optional

from .honeypot import should_block_get_to_post_only_endpoint


ACTION_ALLOW = "allow"
ACTION_BLOCK = "block"


@dataclass(frozen=True)
class MethodDecision:
    action: str
    reason: Optional[str] = None
    status_code: int = 405
    message: Optional[str] = None


def fastapi_route_accepts_method(request, method: str) -> bool:
    """
    Conservative FastAPI route/method detection.
    Returns True when uncertain to avoid false positives.
    """
    try:
        from starlette.routing import Match  # lazy import
    except Exception:
        return True

    try:
        method_u = method.upper()
        route = request.scope.get("route")
        if route is not None:
            methods = {m.upper() for m in getattr(route, "methods", set())}
            if not methods:
                return True
            return method_u in methods

        app = request.scope.get("app")
        router = getattr(app, "router", None)
        routes = getattr(router, "routes", []) if router is not None else []
        path_matched = False

        for candidate in routes:
            try:
                match, _ = candidate.matches(request.scope)
            except Exception:
                continue
            if match == Match.FULL:
                methods = {m.upper() for m in getattr(candidate, "methods", set())}
                if not methods or method_u in methods:
                    return True
                path_matched = True
            elif match == Match.PARTIAL:
                path_matched = True
                methods = {m.upper() for m in getattr(candidate, "methods", set())}
                if methods and method_u in methods:
                    return True
        if path_matched:
            return False
        return True
    except Exception:
        return True


def flask_route_accepts_method(app, path: str, method: str) -> bool:
    """
    Conservative Flask route/method detection with path-aware mismatch handling.
    Returns True when uncertain to avoid false positives.
    """
    try:
        from werkzeug.exceptions import MethodNotAllowed, NotFound  # lazy import
    except Exception:
        return True

    try:
        method_u = method.upper()
        adapter = app.url_map.bind("")
        try:
            adapter.match(path, method=method_u)
            return True
        except MethodNotAllowed:
            return False
        except NotFound:
            return True
    except Exception:
        return True


def evaluate_method_policy(
    *,
    method: str,
    path: str,
    accepts_get: bool,
    accepts_post: bool,
    accepts_method: bool,
) -> MethodDecision:
    method_u = (method or "").upper()
    if method_u == "GET":
        if not accepts_get and should_block_get_to_post_only_endpoint(path, accepts_get=False):
            return MethodDecision(
                action=ACTION_BLOCK,
                reason=f"GET to obvious POST-only endpoint: {path}",
                message=f"GET not allowed for {path}",
            )
        return MethodDecision(action=ACTION_ALLOW)

    if method_u == "POST":
        if not accepts_post:
            return MethodDecision(
                action=ACTION_BLOCK,
                reason=f"POST to GET-only view: {path}",
                message=f"POST not allowed for {path}",
            )
        return MethodDecision(action=ACTION_ALLOW)

    if method_u in {"HEAD", "OPTIONS"}:
        return MethodDecision(action=ACTION_ALLOW)

    if not accepts_method:
        return MethodDecision(
            action=ACTION_BLOCK,
            reason=f"{method_u} to view that doesn't support it: {path}",
            message=f"{method_u} not allowed for {path}",
        )

    return MethodDecision(action=ACTION_ALLOW)
