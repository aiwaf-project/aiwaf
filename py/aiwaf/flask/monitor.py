"""AIWAF_PROFILE=monitor for Flask: hooks log what they would block and let the request through."""

from __future__ import annotations

import functools

from aiwaf.core import profiles


def _describe(result):
    response = result[0] if isinstance(result, tuple) and result else result
    status = result[1] if isinstance(result, tuple) and len(result) > 1 else getattr(response, "status_code", "-")
    try:
        body = response.get_data(as_text=True) if hasattr(response, "get_data") else str(response)
    except Exception:  # noqa: BLE001 - only used for a log line
        body = "blocked"
    return status, (body or "blocked")[:200]


def _log(name, status, reason):
    from flask import request

    from .utils import get_ip

    profiles.would_block(request.method, request.path, get_ip(), status, f"{name}: {reason}")


def _monitor_before(function, name):
    @functools.wraps(function)
    def monitored(*args, **kwargs):
        from werkzeug.exceptions import HTTPException

        try:
            result = function(*args, **kwargs)
        except HTTPException as error:
            if (error.code or 0) < 400:
                raise
            _log(name, error.code, error.description or error.name)
            return None
        if result is not None:
            _log(name, *_describe(result))
        return None

    return monitored


def _monitor_after(function, name):
    @functools.wraps(function)
    def monitored(response):
        changed = function(response)
        if changed is not response and getattr(changed, "status_code", 0) >= 400 > getattr(response, "status_code", 0):
            _log(name, *_describe(changed))
            return response
        return changed

    return monitored


def monitored_init(init_app):
    """Wraps the before_request and after_request hooks an init_app adds, in monitor mode."""

    @functools.wraps(init_app)
    def wrapper(self, app, *args, **kwargs):
        before = list(app.before_request_funcs.get(None, []))
        after = list(app.after_request_funcs.get(None, []))
        result = init_app(self, app, *args, **kwargs)
        if profiles.monitoring():
            name = type(self).__name__
            app.before_request_funcs[None] = [
                function if function in before else _monitor_before(function, name)
                for function in app.before_request_funcs.get(None, [])
            ]
            app.after_request_funcs[None] = [
                function if function in after else _monitor_after(function, name)
                for function in app.after_request_funcs.get(None, [])
            ]
        return result

    return wrapper
