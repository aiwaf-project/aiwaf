"""FastAPI IP/keyword blocking middleware."""

import re
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..storage import get_keyword_store
from ..utils import get_ip, is_exempt
from ..decorators import should_apply_middleware


class IPAndKeywordBlockMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, path_rules=None, malicious_keywords=None):
        super().__init__(app)
        self.path_rules = path_rules or []
        self.malicious_keywords = malicious_keywords or [
            ".php",
            "xmlrpc",
            "wp-",
            ".env",
            ".git",
            ".bak",
            "shell",
            "filemanager",
        ]

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "ip_keyword_block", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        path = request.url.path.lower()

        if BlacklistManager.is_blocked(ip):
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = f"IP blacklisted: {ip}"
            return JSONResponse({"error": "blocked"}, status_code=403)

        keyword_store = get_keyword_store()
        segments = [seg for seg in re.split(r"\W+", path) if len(seg) > 3]

        for kw in self.malicious_keywords:
            if kw in path:
                keyword_store.add_keyword(kw)
                BlacklistManager.block(ip, f"Keyword block: {kw}")
                request.state.aiwaf_blocked = True
                request.state.aiwaf_block_reason = f"Malicious keyword: {kw}"
                return JSONResponse({"error": "blocked"}, status_code=403)

        learned = set(keyword_store.get_top_keywords(100))
        for seg in segments:
            if seg in learned:
                BlacklistManager.block(ip, f"Learned keyword block: {seg}")
                request.state.aiwaf_blocked = True
                request.state.aiwaf_block_reason = f"Learned keyword: {seg}"
                return JSONResponse({"error": "blocked"}, status_code=403)

        return await call_next(request)
