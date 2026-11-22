"""Flask-SocketIO events, checked like request bodies.

Flask isn't ASGI, so ``aiwaf.core.websocket.WebSocketGuard`` can't see Socket.IO events. Here,
each event a client sends is checked with the payload check: its name, and each argument as
text, bytes or JSON. A finding disconnects the client and the handler doesn't run.

- ``guard_socketio(socketio)`` checks every event of a ``SocketIO`` instance.
- ``@aiwaf_message()`` checks one handler's events, under ``@socketio.on(...)``.

Both follow ``AIWAF_SQL_INJECTION_MODE``: monitor logs the rule and runs the handler, off checks
nothing. Log events carry the rule and namespace only, never event content.
"""

from __future__ import annotations

import functools
import json
import logging
from typing import Any, Iterable, Optional

from aiwaf.core.sql_injection import Finding, SQLInjectionPolicy
from aiwaf.core.websocket import inspect_message

LOGGER = logging.getLogger("aiwaf.websocket")
# Socket.IO's own events, which carry no client message.
_LIFECYCLE = {"connect", "disconnect"}


def _event_finding(message: Any, args: Iterable[Any], policy: SQLInjectionPolicy) -> Optional[Finding]:
    for value in (message, *args):
        if value is None or callable(value):
            continue
        if not isinstance(value, (str, bytes, bytearray)):
            try:
                value = json.dumps(value, default=str)
            except (TypeError, ValueError, RecursionError):
                return Finding("payload_inspection_limit", 413)
        finding = inspect_message(value, policy)
        if finding is not None:
            return finding
    return None


def _report(policy: SQLInjectionPolicy, namespace: Any, finding: Finding) -> None:
    LOGGER.warning("%s", json.dumps({"event": "aiwaf.websocket", "mode": policy.mode, "path": str(namespace or "/"),
                                     "rule": finding.rule}, sort_keys=True))


def guard_socketio(socketio, mode: Optional[str] = None, max_bytes: Optional[int] = None):
    """Checks every event a Flask-SocketIO ``SocketIO`` instance receives. Returns it."""
    policy = SQLInjectionPolicy(mode, max_bytes)
    if policy.mode == "off":
        return socketio
    dispatch = getattr(socketio, "_handle_event", None)
    if dispatch is None:
        raise RuntimeError("This Flask-SocketIO version can't be guarded as a whole; "
                           "decorate handlers with @aiwaf_message() instead")

    def checked(handler, message, namespace, sid, *args):
        if message not in _LIFECYCLE:
            finding = _event_finding(message, args, policy)
            if finding is not None:
                _report(policy, namespace, finding)
                if policy.mode == "block":
                    socketio.server.disconnect(sid, namespace=namespace)
                    return None
        return dispatch(handler, message, namespace, sid, *args)

    socketio._handle_event = checked
    return socketio


def aiwaf_message(mode: Optional[str] = None, max_bytes: Optional[int] = None):
    """A decorator for one Flask-SocketIO handler: checks its arguments before it runs."""
    policy = SQLInjectionPolicy(mode, max_bytes)

    def decorate(handler):
        @functools.wraps(handler)
        def checked(*args, **kwargs):
            if policy.mode != "off":
                finding = _event_finding(None, [*args, *kwargs.values()], policy)
                if finding is not None:
                    import flask
                    import flask_socketio

                    _report(policy, getattr(flask.request, "namespace", "/"), finding)
                    if policy.mode == "block":
                        flask_socketio.disconnect()
                        return None
            return handler(*args, **kwargs)

        return checked

    return decorate
