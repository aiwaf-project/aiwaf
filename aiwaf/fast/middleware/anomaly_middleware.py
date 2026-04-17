"""FastAPI AI anomaly middleware (keyword + optional model support)."""

import re
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import should_apply_middleware
from ..storage import get_keyword_store
from ..utils import get_ip, is_exempt

STATIC_KEYWORDS = {
    "admin",
    "wp-admin",
    "wp-content",
    "wp-includes",
    "wp-config",
    "xmlrpc",
    "phpmyadmin",
    ".env",
    ".git",
    "shell",
    "cmd",
    "exec",
    "system",
    "eval",
    "union",
    "select",
    "drop",
    "delete",
    "insert",
    "update",
    "script",
    "javascript",
}


class AIAnomalyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enabled=True, path_rules=None):
        super().__init__(app)
        self.enabled = bool(enabled)
        self.path_rules = path_rules or []

    async def dispatch(self, request, call_next):
        if not self.enabled:
            return await call_next(request)
        if not should_apply_middleware(request, "ai_anomaly", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        path = request.url.path.lower()
        query = str(request.query_params).lower()
        payload = f"{path}?{query}"
        segments = [seg for seg in re.split(r"\W+", payload) if len(seg) > 2]
        keyword_store = get_keyword_store()

        found = []
        for seg in segments:
            if seg in STATIC_KEYWORDS:
                found.append(seg)
                keyword_store.add_keyword(seg)

        learned = set(keyword_store.get_top_keywords(50))
        if not found:
            for seg in segments:
                if seg in learned:
                    found.append(seg)
                    break

        if found:
            reason = f"AI anomaly keyword detection: {found[0]}"
            BlacklistManager.block(ip, reason)
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = reason
            return JSONResponse({"error": "blocked"}, status_code=403)

        return await call_next(request)
