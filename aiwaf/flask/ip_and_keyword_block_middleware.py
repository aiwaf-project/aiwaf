# Flask-adapted IPAndKeywordBlockMiddleware
import re
from flask import request, jsonify
from .utils import get_ip, is_exempt
from .blacklist_manager import BlacklistManager
from .storage import get_keyword_store
from .exemption_decorators import should_apply_middleware

class IPAndKeywordBlockMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        @app.before_request
        def before_request():
            # Check exemption status first - skip if exempt from this middleware
            if not should_apply_middleware('ip_keyword_block'):
                return None  # Allow request to proceed without IP/keyword checking
            
            # Legacy exemption check for backward compatibility
            if is_exempt(request):
                return None  # Allow request to proceed
            
            ip = get_ip()
            path = request.path.lower()
            
            # Get logger if available
            logger = getattr(app, 'aiwaf_logger', None)
            
            # Check if IP is blacklisted first
            if BlacklistManager.is_blocked(ip):
                if logger:
                    logger.mark_request_blocked(f"IP blacklisted: {ip}")
                return jsonify({"error": "blocked"}), 403
            
            keyword_store = get_keyword_store()
            malicious_keywords = [".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "shell", "filemanager"]
            segments = [seg for seg in re.split(r"\W+", path) if len(seg) > 3]
            
            for kw in malicious_keywords:
                if kw in path:
                    keyword_store.add_keyword(kw)
                    BlacklistManager.block(ip, f"Keyword block: {kw}")
                    
                    if logger:
                        logger.mark_request_blocked(f"Malicious keyword: {kw}")
                    
                    return jsonify({"error": "blocked"}), 403
            
            # Block if segment matches learned keyword
            for seg in segments:
                if seg in keyword_store.get_top_keywords():
                    BlacklistManager.block(ip, f"Learned keyword block: {seg}")
                    
                    if logger:
                        logger.mark_request_blocked(f"Learned keyword: {seg}")
                    
                    return jsonify({"error": "blocked"}), 403
