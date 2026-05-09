"""
AIWAF Django entrypoint.

Usage:
    import aiwaf.django as aiwaf
"""

from importlib import import_module

__all__ = [
    "JsonExceptionMiddleware",
    "IPAndKeywordBlockMiddleware",
    "RateLimitMiddleware",
    "GeoBlockMiddleware",
    "AIAnomalyMiddleware",
    "HoneypotTimingMiddleware",
    "UUIDTamperMiddleware",
    "HeaderValidationMiddleware",
    "aiwaf_exempt",
    "aiwaf_exempt_from",
    "aiwaf_only",
    "aiwaf_require_protection",
]


def __getattr__(name):
    if name in {
        "aiwaf_exempt",
        "aiwaf_exempt_from",
        "aiwaf_only",
        "aiwaf_require_protection",
    }:
        module = import_module(".decorators", __name__)
        return getattr(module, name)
    if name in __all__:
        module = import_module(".middleware", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
