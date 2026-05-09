# Flask-adapted HoneypotTimingMiddleware
import time
from flask import request, jsonify, current_app
from .utils import get_ip
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware
from aiwaf.core.honeypot import (
    honeypot_get_key,
    evaluate_form_timing,
    ACTION_BLOCK,
    ACTION_PAGE_EXPIRED,
)
from aiwaf.core.method_validation import evaluate_method_policy, ACTION_BLOCK as METHOD_BLOCK

_aiwaf_cache = {}


def _is_authenticated_request() -> bool:
    """Best-effort auth detection without introducing hard dependencies."""
    try:
        from flask_login import current_user  # type: ignore

        return bool(getattr(current_user, "is_authenticated", False))
    except Exception:
        return False


class HoneypotTimingMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        def _view_accepts_method(method: str) -> bool:
            try:
                rule = request.url_rule
                if not rule:
                    return True
                methods = {m.upper() for m in getattr(rule, "methods", set())}
                return method.upper() in methods
            except Exception:
                return True

        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from honeypot detection
            if not should_apply_middleware('honeypot'):
                return None  # Allow request to proceed without honeypot timing checks

            if app.config.get("AIWAF_HONEYPOT_SKIP_AUTHENTICATED", True) and _is_authenticated_request():
                return None
            
            ip = get_ip()
            now = time.time()
            decision = evaluate_method_policy(
                method=request.method,
                path=request.path,
                accepts_get=_view_accepts_method("GET"),
                accepts_post=_view_accepts_method("POST"),
                accepts_method=_view_accepts_method(request.method),
            )
            if decision.action == METHOD_BLOCK:
                BlacklistManager.block(ip, decision.reason)
                return jsonify({"error": "blocked", "message": decision.message}), decision.status_code

            if request.method == "GET":
                _aiwaf_cache[honeypot_get_key(ip)] = now
            elif request.method == "POST":
                decision = evaluate_form_timing(
                    now=now,
                    get_time=_aiwaf_cache.get(honeypot_get_key(ip)),
                    path=request.path,
                    min_form_time=app.config.get("AIWAF_MIN_FORM_TIME", 1.0),
                    max_page_time=app.config.get("AIWAF_MAX_PAGE_TIME", 240),
                )
                if decision.action == ACTION_PAGE_EXPIRED:
                    _aiwaf_cache.pop(honeypot_get_key(ip), None)
                    return jsonify(
                        {
                            "error": "page_expired",
                            "message": decision.message or "Page has expired. Please reload and try again.",
                            "reload_required": True,
                        }
                    ), (decision.status_code or 409)
                if decision.action == ACTION_BLOCK:
                    BlacklistManager.block(ip, decision.reason or "Form submitted too quickly")
                    return jsonify({"error": "blocked"}), (decision.status_code or 403)
