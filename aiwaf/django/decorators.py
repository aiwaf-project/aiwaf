from functools import wraps
from django.utils.decorators import method_decorator


ALL_MIDDLEWARES = {
    "ip_keyword_block",
    "rate_limit",
    "honeypot",
    "header_validation",
    "geo_block",
    "ai_anomaly",
    "uuid_tamper",
    "logging",
}


def _normalize(names):
    return {str(name).strip().lower() for name in (names or []) if name}


def _mark_view(view_func, *, fully_exempt=False, exempt_middlewares=None, required_middlewares=None):
    prev_fully_exempt = bool(getattr(view_func, "aiwaf_exempt", False) or getattr(view_func, "_aiwaf_exempt", False))
    prev_exempt = set(getattr(view_func, "_aiwaf_exempt_middlewares", set()) or set())
    prev_required = set(getattr(view_func, "_aiwaf_required_middlewares", set()) or set())
    next_exempt = prev_exempt | set(exempt_middlewares or set())
    next_required = prev_required | set(required_middlewares or set())
    next_fully_exempt = bool(fully_exempt or prev_fully_exempt)

    view_func.aiwaf_exempt = next_fully_exempt  # legacy flag
    view_func._aiwaf_exempt = next_fully_exempt
    view_func._aiwaf_exempt_middlewares = next_exempt
    view_func._aiwaf_required_middlewares = next_required
    return view_func


def aiwaf_exempt(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        return view_func(*args, **kwargs)

    return _mark_view(wrapped_view, fully_exempt=True, exempt_middlewares=set())


def aiwaf_exempt_from(*middleware_names):
    selected = _normalize(middleware_names)

    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            return view_func(*args, **kwargs)

        return _mark_view(wrapped_view, fully_exempt=False, exempt_middlewares=selected)

    return decorator


def aiwaf_only(*middleware_names):
    selected = _normalize(middleware_names)
    exempt = ALL_MIDDLEWARES - selected
    return aiwaf_exempt_from(*exempt)


def aiwaf_require_protection(*middleware_names):
    required = _normalize(middleware_names)

    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(*args, **kwargs):
            return view_func(*args, **kwargs)

        return _mark_view(
            wrapped_view,
            fully_exempt=False,
            exempt_middlewares=set(),
            required_middlewares=required,
        )

    return decorator


# For class-based views
aiwaf_exempt_view = method_decorator(aiwaf_exempt, name="dispatch")
