# Flask-adapted RateLimitMiddleware
import time
from flask import request, jsonify, current_app
from .utils import get_ip, is_exempt
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware, get_path_rule_overrides
from aiwaf.core.exemptions import get_path_rule_for_path as core_get_path_rule_for_path

_aiwaf_cache = {}

class RateLimitMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        if not getattr(app, "_aiwaf_rate_cache_key", None):
            app._aiwaf_rate_cache_key = f"{id(app)}:{time.time_ns()}"

        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from rate limiting
            if not should_apply_middleware('rate_limit'):
                return None  # Allow request to proceed without rate limiting

            # Legacy exemption check for backward compatibility
            if is_exempt(request):
                return None  # Allow request to proceed

            if request.environ.get("aiwaf_rate_limit_checked"):
                return None
            request.environ["aiwaf_rate_limit_checked"] = True
            
            ip = get_ip()
            app_key = app._aiwaf_rate_cache_key
            path = request.path or "unknown"
            key = f"ratelimit:{app_key}:{ip}:{path}"
            now = time.time()
            timestamps = _aiwaf_cache.get(key, [])
            window = app.config.get("AIWAF_RATE_WINDOW", 10)
            max_req = app.config.get("AIWAF_RATE_MAX", 20)
            flood = app.config.get("AIWAF_RATE_FLOOD", 40)
            overrides = get_path_rule_overrides("RATE_LIMIT")
            if not overrides:
                overrides = _resolve_rate_limit_overrides(app, request.path)
            if overrides:
                window = overrides.get("WINDOW", window)
                max_req = overrides.get("MAX", max_req)
                flood = overrides.get("FLOOD", flood)
            
            timestamps = [t for t in timestamps if now - t < window]
            timestamps.append(now)
            _aiwaf_cache[key] = timestamps
            if len(timestamps) > flood:
                BlacklistManager.block(ip, "Flood pattern")
                return jsonify({"error": "blocked"}), 403
            if len(timestamps) > max_req:
                return jsonify({"error": "too_many_requests"}), 429


def _resolve_rate_limit_overrides(app, path):
    try:
        rules = app.config.get("AIWAF_PATH_RULES")
        if rules is None:
            settings = app.config.get("AIWAF_SETTINGS", {})
            rules = settings.get("PATH_RULES")
        rules = rules or []
        best = core_get_path_rule_for_path(path, rules)
        if not best:
            return {}
        return best.get("RATE_LIMIT") or best.get("rate_limit") or {}
    except Exception:
        return {}
