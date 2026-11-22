"""WebSocket messages, checked like request bodies.

HTTP checks never see what a WebSocket carries after its handshake, so this checks each message
a client sends with the payload check: SQL injection, remote code execution and their disguises,
and JSON messages by their values. ``WebSocketGuard`` is ASGI middleware, for FastAPI and
Starlette (``app.add_middleware(WebSocketGuard)``) and Django Channels (wrap the application).
``inspect_message`` checks one message, for other servers such as Flask-SocketIO handlers.

A message with a finding closes the socket with 1008 (policy violation), or 1009 when it's too
large to check, and the application sees a disconnect instead of the message. It follows
``AIWAF_SQL_INJECTION_MODE``: monitor logs the rule and passes the message, off checks nothing.
Log events carry the rule and path only, never message content.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .sql_injection import Finding, SQLInjectionPolicy

LOGGER = logging.getLogger("aiwaf.websocket")


def inspect_message(message: Any, policy: Optional[SQLInjectionPolicy] = None) -> Optional[Finding]:
    """The finding for one WebSocket message (text or bytes), or None. Ignores the policy's mode."""
    policy = policy or SQLInjectionPolicy("block")
    if message is None:
        return None
    data = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    return SQLInjectionPolicy("block", policy.max_bytes).inspect(body=data)


class WebSocketGuard:
    """ASGI middleware that checks each message a WebSocket client sends."""

    def __init__(self, app, mode: Optional[str] = None, max_bytes: Optional[int] = None):
        self.app = app
        self.policy = SQLInjectionPolicy(mode, max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket" or self.policy.mode == "off":
            return await self.app(scope, receive, send)
        path = str(scope.get("path", ""))
        closed = False

        async def checked_receive():
            nonlocal closed
            if closed:
                return {"type": "websocket.disconnect", "code": 1008}
            message = await receive()
            if message.get("type") != "websocket.receive":
                return message
            data = message.get("text")
            if data is None:
                data = message.get("bytes")
            finding = inspect_message(data, self.policy)
            if finding is None:
                return message
            LOGGER.warning("%s", json.dumps({"event": "aiwaf.websocket", "mode": self.policy.mode, "path": path,
                                             "rule": finding.rule}, sort_keys=True))
            if self.policy.mode == "monitor":
                return message
            closed = True
            code = 1009 if finding.status == 413 else 1008
            await send({"type": "websocket.close", "code": code})
            return {"type": "websocket.disconnect", "code": code}

        return await self.app(scope, checked_receive, send)
