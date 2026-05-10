"""
AIWAF FastAPI middleware entrypoint.

Usage:
    import aiwaf.fast.middleware as middleware
"""

from importlib import import_module

__all__ = [
    "all",
    "IPAndKeywordBlockMiddleware",
    "RateLimitMiddleware",
    "HoneypotTimingMiddleware",
    "HeaderValidationMiddleware",
    "GeoBlockMiddleware",
    "AIAnomalyMiddleware",
    "UUIDTamperMiddleware",
    "AIWAFLoggingMiddleware",
]
all = "all"


def __getattr__(name):
    if name == "all":
        return all
    module_map = {
        "IPAndKeywordBlockMiddleware": ".ip_and_keyword_block_middleware",
        "RateLimitMiddleware": ".rate_limit_middleware",
        "HoneypotTimingMiddleware": ".honeypot_timing_middleware",
        "HeaderValidationMiddleware": ".header_validation",
        "GeoBlockMiddleware": ".geo_block_middleware",
        "AIAnomalyMiddleware": ".anomaly_middleware",
        "UUIDTamperMiddleware": ".uuid_tamper_middleware",
        "AIWAFLoggingMiddleware": ".logging_middleware",
    }
    if name in module_map:
        module = import_module(module_map[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
