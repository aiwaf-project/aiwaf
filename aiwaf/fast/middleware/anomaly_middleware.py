"""FastAPI AI anomaly middleware.

Uses the shared core anomaly engine (same decision flow as Django/Flask):
- maintain per-IP request history within a sliding window
- optionally score with an IsolationForest model (sklearn or aiwaf-rust)
- perform conservative "scanning behavior" analysis before blocking
- learn suspicious keywords from 404s on non-existent paths
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse

from aiwaf.core.anomaly import HistoryEntry, evaluate_anomaly as core_evaluate_anomaly

from ..blacklist import BlacklistManager
from ..decorators import should_apply_middleware
from ..storage import get_exemption_store, get_keyword_store
from ..utils import get_blacklist_extended_info, get_ip, is_exempt


STATIC_KW = [
    ".php",
    "xmlrpc",
    "wp-",
    ".env",
    ".git",
    ".bak",
    "conflg",
    "shell",
    "filemanager",
]


class AIAnomalyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enabled: bool = True, path_rules=None):
        super().__init__(app)
        self.enabled = bool(enabled)
        self.path_rules = path_rules or []
        self.window_seconds = 60
        self.keyword_learning_enabled = True
        self.model: Any = None
        self.request_cache: Dict[str, List[HistoryEntry]] = {}
        self._route_paths = set()
        self._init_from_app(app)

    def _init_from_app(self, app):
        # Pull settings from attached AIWAF config when available.
        try:
            state = getattr(app, "state", None)
            cfg = getattr(state, "aiwaf_config", None)
            if cfg is not None and hasattr(cfg, "get"):
                self.window_seconds = int(cfg.get("AIWAF_WINDOW_SECONDS", self.window_seconds))
                self.window_seconds = int(cfg.get("ai_anomaly.window_seconds", self.window_seconds))
                self.keyword_learning_enabled = bool(cfg.get("AIWAF_ENABLE_KEYWORD_LEARNING", True))
        except Exception:
            pass

        # Collect route paths for best-effort "path exists" decisions on history entries.
        try:
            routes = getattr(getattr(app, "router", None), "routes", []) or []
            for route in routes:
                p = str(getattr(route, "path", "") or "")
                if p:
                    self._route_paths.add(p)
        except Exception:
            pass

    def _path_exists(self, path: str) -> bool:
        # Best-effort: exact match against known route templates.
        try:
            candidate = (path or "").split("?", 1)[0]
            return candidate in self._route_paths
        except Exception:
            return False

    async def dispatch(self, request, call_next):
        if not self.enabled:
            return await call_next(request)
        if not should_apply_middleware(request, "ai_anomaly", self.path_rules):
            return await call_next(request)
        if is_exempt(request):
            return await call_next(request)

        ip = get_ip(request)
        if get_exemption_store().is_exempted(ip):
            return await call_next(request)

        if BlacklistManager.is_blocked(ip):
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = "IP already blacklisted"
            return JSONResponse({"error": "blocked"}, status_code=403)

        start = time.time()
        response = await call_next(request)
        now = time.time()
        resp_time = now - start

        key = f"aiwaf:{ip}"
        history = self.request_cache.get(key, [])
        path_lower = (request.url.path or "").lower()
        query_lower = str(getattr(request.url, "query", "") or "").lower()

        def _path_exists(candidate: str) -> bool:
            try:
                if candidate == request.url.path:
                    return request.scope.get("route") is not None
            except Exception:
                pass
            return self._path_exists(candidate)

        def _is_malicious_context(seg: str) -> bool:
            if not seg:
                return False
            if "../" in path_lower or "..\\" in path_lower:
                return True
            if seg in query_lower and any(token in query_lower for token in ("union", "select", "drop", "insert", "script", "alert", "eval")):
                return True
            if seg.startswith("."):
                return True
            return False

        outcome = core_evaluate_anomaly(
            ip=ip,
            path=request.url.path,
            status_code=int(getattr(response, "status_code", 0) or 0),
            response_time=float(resp_time),
            now=float(now),
            history=history,
            window_seconds=float(self.window_seconds),
            model=self.model,
            static_keywords=STATIC_KW,
            malicious_keywords=STATIC_KW,
            keyword_learning_enabled=bool(self.keyword_learning_enabled),
            path_exists=_path_exists,
            is_exempt_path=lambda _p: False,
            is_malicious_context=_is_malicious_context,
        )

        self.request_cache[key] = outcome.updated_history

        if outcome.learned_keywords:
            keyword_store = get_keyword_store()
            for seg in outcome.learned_keywords:
                keyword_store.add_keyword(seg)

        if outcome.block and outcome.reason:
            BlacklistManager.block(
                ip,
                outcome.reason,
                extended_request_info=get_blacklist_extended_info(request),
            )
            request.state.aiwaf_blocked = True
            request.state.aiwaf_block_reason = outcome.reason
            return JSONResponse({"error": "blocked"}, status_code=403)

        return response
