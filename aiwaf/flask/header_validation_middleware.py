# Flask-adapted HeaderValidationMiddleware
import re
from flask import request, jsonify, current_app
from .utils import get_ip, is_exempt
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware
from aiwaf.core import rust_backend
from aiwaf.core import header_validation

def _get_min_quality_score(config_required_headers=None, method=None):
    default_min = current_app.config.get("AIWAF_HEADER_QUALITY_MIN_SCORE", 3)
    required_headers = header_validation.resolve_required_headers(config_required_headers, method)
    if not required_headers:
        return 0
    return default_min

class HeaderValidationMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from header validation
            if not should_apply_middleware('header_validation'):
                return None  # Allow request to proceed without header validation
            
            # Legacy exemption check for backward compatibility
            if is_exempt(request):
                return None  # Allow request to proceed
            
            ip = get_ip()
            req_method = (request.method or "GET").upper()
            configured_required = current_app.config.get("AIWAF_REQUIRED_HEADERS")

            use_rust = (
                current_app.config.get("AIWAF_USE_RUST", False)
                and current_app.config.get("AIWAF_USE_CSV", True)
                and rust_backend.rust_available()
            )
            effective_required_headers = header_validation.resolve_required_headers(configured_required, req_method)
            default_required_headers = header_validation.resolve_required_headers(None, req_method)
            has_method_override = effective_required_headers != default_required_headers
            min_score = _get_min_quality_score(configured_required, req_method)

            if use_rust and not has_method_override:
                reason = rust_backend.validate_headers(
                    request.environ, effective_required_headers, min_score
                )
            else:
                reason = header_validation.validate_headers_python(
                    request.environ,
                    method=req_method,
                    config_required_headers=configured_required,
                    min_score=min_score,
                )

            if reason:
                BlacklistManager.block(ip, reason)
                logger = getattr(current_app, "aiwaf_logger", None)
                if logger is not None:
                    logger.mark_request_blocked(reason)
                return jsonify({"error": "blocked"}), 403
