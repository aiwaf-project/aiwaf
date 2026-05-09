"""FastAPI IP/keyword blocking middleware."""

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..storage import get_keyword_store
from ..utils import get_blacklist_extended_info, get_ip, is_exempt
from ..decorators import should_apply_middleware
from ...core.ip_keyword import evaluate_keyword_policy, extract_path_segments
from ...core.request_context import extract_query_keys_from_fastapi_request


class IPAndKeywordBlockMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, path_rules=None, malicious_keywords=None):
        super().__init__(app)
        self.path_rules = path_rules or []
        self.malicious_keywords = set(malicious_keywords or [
            ".php",
            "xmlrpc",
            "wp-",
            ".env",
            ".git",
            ".bak",
            "shell",
            "filemanager",
        ])
        self.keyword_learning_enabled = True
        self.dynamic_top_n = 10
        self.exempt_keywords = set()
        self.legitimate_path_keywords = set()
        self.safe_prefixes = set()
        self._init_rich_policy(app)

    def _init_rich_policy(self, app):
        try:
            app_state = getattr(app, "app", app)
            state_obj = getattr(app_state, "state", None)
            cfg = getattr(state_obj, "aiwaf_config", None)
            if cfg is not None:
                self.keyword_learning_enabled = bool(cfg.get("ai_anomaly.enabled", True) if hasattr(cfg, "get") else True)
                self.dynamic_top_n = int(cfg.get("AIWAF_DYNAMIC_TOP_N", 10) if hasattr(cfg, "get") else 10)
        except Exception:
            pass
        # Conservative defaults mirroring Django baseline + route extraction.
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
        safe_prefixes = set()
        app_state = getattr(app, "app", app)
        routes = getattr(getattr(app_state, "router", None), "routes", [])
        for route in routes:
            path = str(getattr(route, "path", "") or "")
            for seg in re.findall(r"([a-zA-Z]\w{2,})", path.lower()):
                route_keywords.add(seg)
            cleaned = path.strip("/")
            if cleaned:
                first = cleaned.split("/")[0]
                if first:
                    safe_prefixes.add(first.lower())
        self.legitimate_path_keywords = default_legitimate | route_keywords
        self.safe_prefixes = safe_prefixes

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "ip_keyword_block", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        path = request.url.path.lower()
        path_exists = request.scope.get("route") is not None

        if BlacklistManager.is_blocked(ip):
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = f"IP blacklisted: {ip}"
            return JSONResponse({"error": "blocked"}, status_code=403)

        keyword_store = get_keyword_store()
        dynamic_top = keyword_store.get_top_keywords(self.dynamic_top_n) if self.keyword_learning_enabled else []

        def _is_malicious_context(seg: str) -> bool:
            query = str(request.url.query).lower()
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
            path=request.url.path,
            query_keys=list(extract_query_keys_from_fastapi_request(request)),
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
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = decision.block_reason
            return JSONResponse({"error": "blocked"}, status_code=403)

        return await call_next(request)
