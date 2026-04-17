"""FastAPI UUID tamper middleware."""

import re
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from ..blacklist import BlacklistManager
from ..decorators import should_apply_middleware
from ..utils import get_ip


UUID_RE = re.compile(r"^[a-f0-9\-]{36}$")


class UUIDTamperMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, path_rules=None):
        super().__init__(app)
        self.path_rules = path_rules or []

    async def dispatch(self, request, call_next):
        if not should_apply_middleware(request, "uuid_tamper", self.path_rules):
            return await call_next(request)

        ip = get_ip(request)
        uuid_val = request.query_params.get("uuid")
        if uuid_val and not UUID_RE.match(uuid_val):
            BlacklistManager.block(ip, "UUID tampering")
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "UUID tampering"
            return JSONResponse({"error": "blocked"}, status_code=403)

        return await call_next(request)
