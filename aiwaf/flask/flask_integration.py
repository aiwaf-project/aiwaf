from __future__ import annotations

from typing import Iterable

from .ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .honeypot_timing_middleware import HoneypotTimingMiddleware
from .header_validation_middleware import HeaderValidationMiddleware
from .anomaly_middleware import AIAnomalyMiddleware
from .uuid_tamper_middleware import UUIDTamperMiddleware
from .logging_middleware import AIWAFLoggingMiddleware
from .geo_block_middleware import GeoBlockMiddleware


class AIWAF:
    AVAILABLE_MIDDLEWARES = {
        "ip_keyword_block": IPAndKeywordBlockMiddleware,
        "rate_limit": RateLimitMiddleware,
        "honeypot": HoneypotTimingMiddleware,
        "header_validation": HeaderValidationMiddleware,
        "geo_block": GeoBlockMiddleware,
        "ai_anomaly": AIAnomalyMiddleware,
        "uuid_tamper": UUIDTamperMiddleware,
        "logging": AIWAFLoggingMiddleware,
    }

    def __init__(self, app=None, middlewares: Iterable[str] | None = None, disable_middlewares: Iterable[str] | None = None, use_database=None):
        self.app = None
        self.middleware_instances = {}
        self.enabled_middlewares = set()
        if app is not None:
            self.init_app(app, middlewares=middlewares, disable_middlewares=disable_middlewares, use_database=use_database)

    @classmethod
    def list_available_middlewares(cls):
        return list(cls.AVAILABLE_MIDDLEWARES.keys())

    def get_enabled_middlewares(self):
        return list(self.enabled_middlewares)

    def is_middleware_enabled(self, name: str) -> bool:
        return name in self.enabled_middlewares

    def _resolve_middlewares(self, middlewares, disable_middlewares):
        all_keys = list(self.AVAILABLE_MIDDLEWARES.keys())
        if middlewares is None:
            enabled = set(all_keys)
        else:
            enabled = {m for m in middlewares if m in self.AVAILABLE_MIDDLEWARES}
        if disable_middlewares:
            enabled -= {m for m in disable_middlewares if m in self.AVAILABLE_MIDDLEWARES}
        return enabled

    def init_app(self, app, middlewares: Iterable[str] | None = None, disable_middlewares: Iterable[str] | None = None, use_database=None):
        self.app = app
        enabled = self._resolve_middlewares(middlewares, disable_middlewares)
        self.enabled_middlewares = set(enabled)

        # ensure database initialized when requested
        if use_database is True:
            from .middleware import _init_database
            _init_database(app)

        for name in self.AVAILABLE_MIDDLEWARES:
            if name not in enabled:
                continue
            middleware_cls = self.AVAILABLE_MIDDLEWARES[name]
            instance = middleware_cls(app)
            self.middleware_instances[name] = instance

        app.extensions = getattr(app, "extensions", {})
        app.extensions["aiwaf"] = self
        return self
