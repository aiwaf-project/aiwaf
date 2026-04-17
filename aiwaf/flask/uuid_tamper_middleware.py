# Flask-adapted UUIDTamperMiddleware (stub)
from flask import request, jsonify
from .utils import get_ip
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware
import re

class UUIDTamperMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from UUID tampering detection
            if not should_apply_middleware('uuid_tamper'):
                return None  # Allow request to proceed without UUID checking
            
            ip = get_ip()
            uuid_val = request.args.get("uuid")
            if uuid_val and not re.match(r"^[a-f0-9\-]{36}$", uuid_val):
                BlacklistManager.block(ip, "UUID tampering")
                return jsonify({"error": "blocked"}), 403
