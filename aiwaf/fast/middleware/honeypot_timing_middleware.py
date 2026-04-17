"""FastAPI honeypot timing middleware."""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import should_apply_middleware
from ..utils import get_ip

_AIWAF_CACHE = {}


class HoneypotTimingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, min_form_time=1.0, path_rules=None):
        super().__init__(app)
        self.min_form_time = float(min_form_time)
        self.path_rules = path_rules or []

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "honeypot", self.path_rules):
            return await call_next(request)

        ip = get_ip(request)
        now = time.time()
        if request.method == "POST":
            get_time = _AIWAF_CACHE.get(f"honeypot_get:{ip}")
            if get_time is not None:
                time_diff = now - get_time
                if time_diff < self.min_form_time:
                    BlacklistManager.block(ip, f"Form submitted too quickly ({time_diff:.2f}s)")
                    request.state.aiwaf_blocked = True
                    request.state.aiwaf_block_reason = "honeypot"
                    return JSONResponse({"error": "blocked"}, status_code=403)
        elif request.method == "GET":
            _AIWAF_CACHE[f"honeypot_get:{ip}"] = now

        return await call_next(request)
