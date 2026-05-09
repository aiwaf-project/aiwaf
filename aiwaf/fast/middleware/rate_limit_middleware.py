"""FastAPI rate-limit middleware."""

import time
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import get_path_rule_overrides, should_apply_middleware
from ..utils import get_blacklist_extended_info, get_ip, is_exempt
from ...core.rate_limit import (
    THROTTLE,
    FLOOD_BLOCK,
    build_rate_limit_key,
    evaluate_rate_limit,
    normalize_rate_key_mode,
)
from ...core.block_responses import blocked_response, throttle_response

_AIWAF_CACHE = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        max_requests=20,
        window_seconds=10,
        flood_threshold=40,
        path_rules=None,
        key_mode="ip_path",
        soft_block_blacklist=False,
    ):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.flood_threshold = flood_threshold
        self.path_rules = path_rules or []
        self.app_key = f"{id(app)}:{time.time_ns()}"
        self.key_mode = normalize_rate_key_mode(key_mode)
        self.soft_block_blacklist = bool(soft_block_blacklist)

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "rate_limit", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        path = request.url.path or "unknown"
        key = build_rate_limit_key("ratelimit", ip, path, key_mode=self.key_mode, app_key=self.app_key)
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

        decision = evaluate_rate_limit(
            timestamps=timestamps,
            now=now,
            window_seconds=window,
            max_requests=max_req,
            flood_threshold=flood,
        )
        _AIWAF_CACHE[key] = decision.timestamps

        if decision.action == FLOOD_BLOCK:
            BlacklistManager.block(ip, "Flood pattern", extended_request_info=get_blacklist_extended_info(request))
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "Flood pattern"
            payload, status = blocked_response()
            return JSONResponse(payload, status_code=status)
        if decision.action == THROTTLE:
            if self.soft_block_blacklist:
                BlacklistManager.block(
                    ip,
                    "Rate limit exceeded",
                    extended_request_info=get_blacklist_extended_info(request),
                )
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "Rate limit exceeded"
            payload, status = throttle_response()
            return JSONResponse(payload, status_code=status)

        return await call_next(request)
