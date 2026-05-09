import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware


def test_fast_logging_json_includes_normalized_context(tmp_path):
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(AIWAFLoggingMiddleware, log_dir=str(tmp_path), log_format="json")
    client = TestClient(app)
    resp = client.get(
        "/ok?a=1",
        headers={"user-agent": "UA-fast", "referer": "https://ref.fast/", "accept": "text/html"},
    )
    assert resp.status_code == 200

    access_log = tmp_path / "access.log"
    lines = access_log.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["path"] == "/ok"
    assert payload["query_string"] == "a=1"
    assert payload["referer"] == "https://ref.fast/"
    assert payload["user_agent"] == "UA-fast"
