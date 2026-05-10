# Flask-adapted IPAndKeywordBlockMiddleware
import re
from flask import request, jsonify
from .utils import get_blacklist_extended_info, get_ip, is_exempt
from .blacklist_manager import BlacklistManager
from .storage import get_keyword_store
from .exemption_decorators import should_apply_middleware
from aiwaf.core.ip_keyword import evaluate_keyword_policy, extract_path_segments
from aiwaf.core.request_context import extract_query_keys_from_flask_request

class IPAndKeywordBlockMiddleware:
    def __init__(self, app=None):
        self.app = app
        self.keyword_learning_enabled = True
        self.dynamic_top_n = 10
        self.exempt_keywords = set()
        self.legitimate_path_keywords = set()
        self.safe_prefixes = set()
        self.malicious_keywords = {".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "shell", "filemanager"}
        if app is not None:
            self.keyword_learning_enabled = bool(app.config.get("AIWAF_ENABLE_KEYWORD_LEARNING", True))
            self.dynamic_top_n = int(app.config.get("AIWAF_DYNAMIC_TOP_N", 10))
            self.exempt_keywords = self._get_exempt_keywords(app)
            self.legitimate_path_keywords = self._get_legitimate_path_keywords(app)
            self.safe_prefixes = self._collect_safe_prefixes(app)
            self.init_app(app)

    def _get_exempt_keywords(self, app):
        tokens = set()
        for path in app.config.get("AIWAF_EXEMPT_PATHS", set()) or set():
            for seg in re.split(r"\W+", str(path).strip("/").lower()):
                if len(seg) > 3:
                    tokens.add(seg)
        tokens.update(app.config.get("AIWAF_EXEMPT_KEYWORDS", []) or [])
        return tokens

    def _get_legitimate_path_keywords(self, app):
        default_legitimate = {
            "profile", "user", "users", "account", "accounts", "settings", "dashboard",
            "home", "about", "contact", "help", "search", "list", "lists", "view",
            "views", "edit", "create", "update", "delete", "detail", "details", "api",
            "auth", "login", "logout", "register", "signup", "signin", "reset",
            "confirm", "activate", "verify", "page", "pages", "category", "categories",
            "tag", "tags", "post", "posts", "article", "articles", "blog", "blogs",
            "news", "item", "items", "admin", "administration", "manage", "manager",
            "control", "panel", "config", "configuration", "option", "options",
            "preference", "preferences",
        }
        route_keywords = set()
        for rule in app.url_map.iter_rules():
            for seg in re.findall(r"([a-zA-Z]\w{2,})", str(rule.rule).lower()):
                route_keywords.add(seg)
        route_keywords.update(app.config.get("AIWAF_ALLOWED_PATH_KEYWORDS", []) or [])
        route_keywords.update(app.config.get("AIWAF_EXEMPT_KEYWORDS", []) or [])
        return default_legitimate | route_keywords

    def _collect_safe_prefixes(self, app):
        prefixes = set()
        for rule in app.url_map.iter_rules():
            path = str(rule.rule).strip("/")
            if not path:
                continue
            first = path.split("/")[0]
            if first:
                prefixes.add(first.lower())
        return prefixes

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
            path_exists = bool(getattr(request, "endpoint", None))
            
            # Get logger if available
            logger = getattr(app, 'aiwaf_logger', None)
            
            # Check if IP is blacklisted first
            if BlacklistManager.is_blocked(ip):
                if logger:
                    logger.mark_request_blocked(f"IP blacklisted: {ip}")
                return jsonify({"error": "blocked"}), 403
            
            keyword_store = get_keyword_store()
            dynamic_top = keyword_store.get_top_keywords(self.dynamic_top_n) if self.keyword_learning_enabled else []

            def _is_malicious_context(seg: str) -> bool:
                query = request.query_string.decode("utf-8", errors="ignore").lower() if request.query_string else ""
                if seg in query and any(token in query for token in ("union", "select", "drop", "insert", "script", "alert", "eval")):
                    return True
                if "../" in path or "..\\" in path:
                    return True
                if seg.startswith(".") and not path_exists:
                    return True
                if (not path_exists) and any(ext in seg for ext in (".php", ".asp", ".jsp", ".cgi")):
                    return True
                return False

            decision = evaluate_keyword_policy(
                path=request.path,
                query_keys=list(extract_query_keys_from_flask_request(request)),
                path_exists=path_exists,
                keyword_learning_enabled=self.keyword_learning_enabled,
                static_keywords=self.malicious_keywords,
                dynamic_keywords=dynamic_top,
                legitimate_keywords=set(self.legitimate_path_keywords),
                exempt_keywords=set(self.exempt_keywords),
                safe_prefixes=self.safe_prefixes,
                malicious_keywords=set(self.malicious_keywords),
                is_malicious_context=_is_malicious_context,
            )
            for seg in decision.learned_keywords:
                keyword_store.add_keyword(seg)
            if decision.block_reason:
                BlacklistManager.block(
                    ip,
                    decision.block_reason,
                    extended_request_info=get_blacklist_extended_info(request),
                )
                if logger:
                    logger.mark_request_blocked(decision.block_reason)
                return jsonify({"error": "blocked"}), 403
