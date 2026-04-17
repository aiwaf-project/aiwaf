"""FastAPI rate-limit middleware."""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import get_path_rule_overrides, should_apply_middleware
from ..utils import get_ip, is_exempt

_AIWAF_CACHE = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=20, window_seconds=10, flood_threshold=40, path_rules=None):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.flood_threshold = flood_threshold
        self.path_rules = path_rules or []
        self.app_key = f"{id(app)}:{time.time_ns()}"

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "rate_limit", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        path = request.url.path or "unknown"
        key = f"ratelimit:{self.app_key}:{ip}:{path}"
        now = time.time()
        timestamps = _AIWAF_CACHE.get(key, [])

        window = self.window_seconds
        max_req = self.max_requests
        flood = self.flood_threshold
        overrides = get_path_rule_overrides(request, "RATE_LIMIT", self.path_rules)
        if overrides:
            window = int(overrides.get("WINDOW", window))
            max_req = int(overrides.get("MAX", max_req))
            flood = int(overrides.get("FLOOD", flood))

        timestamps = [t for t in timestamps if now - t < window]
        timestamps.append(now)
        _AIWAF_CACHE[key] = timestamps

        if len(timestamps) > flood:
            BlacklistManager.block(ip, "Flood pattern")
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "Flood pattern"
            return JSONResponse({"error": "blocked"}, status_code=403)
        if len(timestamps) > max_req:
            BlacklistManager.block(ip, "Rate limit exceeded")
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "Rate limit exceeded"
            return JSONResponse({"error": "too_many_requests"}, status_code=429)

        return await call_next(request)
