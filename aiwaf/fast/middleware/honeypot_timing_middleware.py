"""FastAPI honeypot timing middleware."""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import should_apply_middleware
from ..utils import get_blacklist_extended_info, get_ip
from ...core.honeypot import (
    honeypot_get_key,
    evaluate_form_timing,
    ACTION_BLOCK,
    ACTION_PAGE_EXPIRED,
)
from ...core.method_validation import evaluate_method_policy, ACTION_BLOCK as METHOD_BLOCK

_AIWAF_CACHE = {}


class HoneypotTimingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, min_form_time=1.0, max_page_time=240, path_rules=None):
        super().__init__(app)
        self.min_form_time = float(min_form_time)
        self.max_page_time = float(max_page_time)
        self.path_rules = path_rules or []

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "honeypot", self.path_rules):
            return await call_next(request)

        ip = get_ip(request)
        now = time.time()

        def _view_accepts_method(method: str) -> bool:
            try:
                route = request.scope.get("route")
                method_u = method.upper()
                if route is not None:
                    methods = {m.upper() for m in getattr(route, "methods", set())}
                    if not methods:
                        return True
                    return method_u in methods

                # Fallback: detect path match even when routing did not resolve a route
                # for this method (common for method-mismatch 405 scenarios).
                app = request.scope.get("app")
                router = getattr(app, "router", None)
                routes = getattr(router, "routes", []) if router is not None else []
                path_matched = False
                for candidate in routes:
                    try:
                        match, _child_scope = candidate.matches(request.scope)
                    except Exception:
                        continue
                    if match == Match.FULL:
                        methods = {m.upper() for m in getattr(candidate, "methods", set())}
                        if not methods or method_u in methods:
                            return True
                        path_matched = True
                    elif match == Match.PARTIAL:
                        # Path matched but method likely not allowed.
                        path_matched = True
                        methods = {m.upper() for m in getattr(candidate, "methods", set())}
                        if methods and method_u in methods:
                            return True
                if path_matched:
                    return False
                return True
            except Exception:
                return True

        method_decision = evaluate_method_policy(
            method=request.method,
            path=request.url.path,
            accepts_get=_view_accepts_method("GET"),
            accepts_post=_view_accepts_method("POST"),
            accepts_method=_view_accepts_method(request.method),
        )
        if method_decision.action == METHOD_BLOCK:
            BlacklistManager.block(
                ip,
                method_decision.reason,
                extended_request_info=get_blacklist_extended_info(request),
            )
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = method_decision.message or "method not allowed"
            return JSONResponse({"error": "blocked", "message": method_decision.message}, status_code=method_decision.status_code)

        if request.method == "GET":
            _AIWAF_CACHE[honeypot_get_key(ip)] = now
        elif request.method == "POST":
            decision = evaluate_form_timing(
                now=now,
                get_time=_AIWAF_CACHE.get(honeypot_get_key(ip)),
                path=request.url.path,
                min_form_time=self.min_form_time,
                max_page_time=self.max_page_time,
            )
            if decision.action == ACTION_PAGE_EXPIRED:
                _AIWAF_CACHE.pop(honeypot_get_key(ip), None)
                return JSONResponse(
                    {
                        "error": "page_expired",
                        "message": decision.message or "Page has expired. Please reload and try again.",
                        "reload_required": True,
                    },
                    status_code=decision.status_code or 409,
                )
            if decision.action == ACTION_BLOCK:
                BlacklistManager.block(
                    ip,
                    decision.reason or "Form submitted too quickly",
                    extended_request_info=get_blacklist_extended_info(request),
                )
                request.state.aiwaf_blocked = True
                request.state.aiwaf_block_reason = "honeypot"
                return JSONResponse({"error": "blocked"}, status_code=decision.status_code or 403)

        return await call_next(request)
