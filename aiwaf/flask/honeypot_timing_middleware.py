# Flask-adapted HoneypotTimingMiddleware
import time
from flask import request, jsonify, current_app
from .utils import get_ip
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware

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
        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from honeypot detection
            if not should_apply_middleware('honeypot'):
                return None  # Allow request to proceed without honeypot timing checks

            if app.config.get("AIWAF_HONEYPOT_SKIP_AUTHENTICATED", True) and _is_authenticated_request():
                return None
            
            ip = get_ip()
            now = time.time()
            if request.method == "POST":
                get_time = _aiwaf_cache.get(f"honeypot_get:{ip}")
                if get_time is not None:
                    time_diff = now - get_time
                    min_time = app.config.get("AIWAF_MIN_FORM_TIME", 1.0)
                    if time_diff < min_time:
                        BlacklistManager.block(ip, f"Form submitted too quickly ({time_diff:.2f}s)")
                        return jsonify({"error": "blocked"}), 403
            elif request.method == "GET":
                _aiwaf_cache[f"honeypot_get:{ip}"] = now
