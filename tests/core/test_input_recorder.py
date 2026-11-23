"""Recording which query and body names views read, in Flask, FastAPI and on its own."""

import asyncio
import json
from pathlib import Path

import pytest

from aiwaf.core import input_recorder
from aiwaf.core.input_recorder import AsgiInputRecorder, Recording, destination, record_flask_inputs


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_destination_follows_the_environment(monkeypatch):
    monkeypatch.delenv(input_recorder.ENVIRONMENT, raising=False)
    assert destination() is None
    assert destination("off") is None and destination("") is None
    assert destination("1") == Path(input_recorder.DEFAULT_PATH)
    assert destination("logs/seen.jsonl") == Path("logs/seen.jsonl")
    monkeypatch.setenv(input_recorder.ENVIRONMENT, "true")
    assert destination() == Path(input_recorder.DEFAULT_PATH)


def test_a_recording_notes_only_its_own_containers(tmp_path):
    mine, other = {}, {}
    recording = Recording(lambda container: container is mine)
    recording.note(mine, "q")
    recording.note(other, "x")
    recording.note(mine, "csrf_token")
    body = recording.wrap({"a": 1, "b": 2})
    assert recording.wrap([1]) == [1]
    token = input_recorder._current.set(recording)
    try:
        assert body["a"] == 1 and body.get("b") == 2
        assert input_recorder.current() is recording
        list(body.items())
        list(iter(body))
        recording.note(body, "after whole")
    finally:
        input_recorder._current.reset(token)
    assert recording.keys == {"q", "a", "b", "*"}
    broken = Recording(lambda container: 1 / 0)
    assert broken.mine({}) is False
    recording.save("/x", "get", tmp_path / "out" / "seen.jsonl")
    recording.save(None, "get", tmp_path / "skipped.jsonl")
    assert [line["key"] for line in lines(tmp_path / "out" / "seen.jsonl")] == ["*", "a", "b", "q"]
    assert not (tmp_path / "skipped.jsonl").exists()


def test_flask_records_args_form_and_json(tmp_path):
    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/search/<int:page>", methods=["GET", "POST"])
    def search(page):
        request.args.get("q")
        if request.is_json:
            data = request.get_json()
            return data["name"]
        return request.form["title"] if request.method == "POST" else ""

    target = tmp_path / "seen.jsonl"
    record_flask_inputs(app, str(target))
    client = app.test_client()
    client.get("/search/1?q=a&extra=1")
    client.post("/search/2", data={"title": "t", "csrf_token": "x"})
    client.post("/search/3", json={"name": "n", "unread": 1})
    found = sorted({(line["method"], line["key"]) for line in lines(target)})
    assert found == [("GET", "q"), ("POST", "name"), ("POST", "q"), ("POST", "title")]
    assert {line["route"] for line in lines(target)} == {"/search/<int:page>"}
    assert record_flask_inputs(Flask("off"), "0").name == "off"


def test_asgi_records_fastapi_query_form_and_json(tmp_path):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: int, request: Request):
        request.query_params.get("q")
        return {}

    @app.post("/json")
    async def body(request: Request):
        data = await request.json()
        return {"name": data["name"]}

    @app.post("/form")
    async def form(request: Request):
        values = await request.form()
        return {"title": values.get("title")}

    target = tmp_path / "seen.jsonl"
    app.add_middleware(AsgiInputRecorder, path=str(target))
    client = TestClient(app)
    client.get("/items/1?q=a&extra=2")
    client.post("/json", json={"name": "n", "other": 1})
    client.post("/form", data={"title": "t"})
    found = sorted({(line["route"], line["key"]) for line in lines(target)})
    assert found == [("/form", "title"), ("/items/{item_id}", "q"), ("/json", "name")]


def test_asgi_passes_through_when_off_or_not_http():
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    asyncio.run(AsgiInputRecorder(app, path="off")({"type": "http"}, None, None))
    asyncio.run(AsgiInputRecorder(app, path="unused.jsonl")({"type": "lifespan"}, None, None))
    assert seen == ["http", "lifespan"]
    assert not Path("unused.jsonl").exists()


@pytest.mark.parametrize("label", ["django", "werkzeug", "starlette"])
def test_install_patches_once(label):
    input_recorder.install()
    input_recorder.install()
    assert label in input_recorder._installed


def test_aiwaf_itself_records_when_asked(tmp_path, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from flask import Flask, request

    from aiwaf.fast import AIWAF as FastAIWAF
    from aiwaf.flask import AIWAF as FlaskAIWAF

    monkeypatch.delenv("AIWAF_RECORD_INPUTS", raising=False)
    flask_target = tmp_path / "flask.jsonl"
    app = Flask(__name__)
    app.config.update(AIWAF_RECORD_INPUTS=str(flask_target), AIWAF_USE_CSV=False)

    @app.route("/search")
    def search():
        return request.args.get("q", "")

    FlaskAIWAF(app, middlewares=["rate_limit"])
    app.test_client().get("/search?q=a&extra=1", headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    assert {(line["route"], line["key"]) for line in lines(flask_target)} == {("/search", "q")}
    plain = Flask("plain")
    plain.config.update(AIWAF_USE_CSV=False)
    FlaskAIWAF(plain, middlewares=["rate_limit"])
    assert not plain.before_request_funcs.get(None) or all(
        "recording" not in hook.__name__ for hook in plain.before_request_funcs[None])

    fast_target = tmp_path / "fast.jsonl"
    api = FastAPI()

    @api.get("/items")
    async def items(request: Request):
        return {"q": request.query_params.get("q")}

    FastAIWAF(api, middlewares=["rate_limit"], AIWAF_RECORD_INPUTS=str(fast_target))
    assert "AsgiInputRecorder" in [mw.cls.__name__ for mw in api.user_middleware]
    TestClient(api).get("/items?q=a&extra=2", headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"})
    assert {(line["route"], line["key"]) for line in lines(fast_target)} == {("/items", "q")}
    quiet = FastAPI()
    FastAIWAF(quiet, middlewares=["rate_limit"])
    assert "AsgiInputRecorder" not in [mw.cls.__name__ for mw in quiet.user_middleware]
