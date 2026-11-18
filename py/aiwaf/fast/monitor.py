"""AIWAF_PROFILE=monitor for FastAPI: middleware logs what it would block and lets the request through."""

from __future__ import annotations

from aiwaf.core import profiles


def _reason(response) -> str:
    body = getattr(response, "body", b"") or b""
    try:
        return body[:200].decode("utf-8", "replace") or "blocked"
    except AttributeError:
        return "blocked"


def monitored(cls):
    """A subclass of a BaseHTTPMiddleware that passes on each request it would block."""
    from starlette.middleware.base import BaseHTTPMiddleware

    if not isinstance(cls, type) or not issubclass(cls, BaseHTTPMiddleware):
        return cls

    class Monitored(cls):
        async def dispatch(self, request, call_next):
            from aiwaf.core.runtime_utils import get_ip

            passed = {}

            async def passing(inner):
                passed["response"] = await call_next(inner)
                return passed["response"]

            response = await super().dispatch(request, passing)
            if "response" not in passed:
                profiles.would_block(request.method, request.url.path, get_ip(request),
                                     getattr(response, "status_code", "-"), f"{cls.__name__}: {_reason(response)}")
                return await call_next(request)
            original = passed["response"]
            if response is not original and getattr(response, "status_code", 0) >= 400 > original.status_code:
                profiles.would_block(request.method, request.url.path, get_ip(request), response.status_code,
                                     f"{cls.__name__}: {_reason(response)}")
                return original
            return response

    Monitored.__name__ = cls.__name__
    Monitored.__qualname__ = cls.__qualname__
    return Monitored
