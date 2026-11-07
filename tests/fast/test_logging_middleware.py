import time
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware


def test_logging_middleware_writes_access_log(tmp_path):
    app = FastAPI()

    @app.get("/ok")
    async def ok():
        return {"ok": True}

    app.add_middleware(
        AIWAFLoggingMiddleware,
        log_dir=str(tmp_path),
        log_format="combined",
    )
    client = TestClient(app)
    resp = client.get("/ok", headers={"user-agent": "Mozilla/5.0", "accept": "text/html"})
    assert resp.status_code == 200

    access_log = tmp_path / "access.log"
    assert access_log.exists()
    assert "GET /ok" in access_log.read_text(encoding="utf-8")
import time
from types import SimpleNamespace


def test_csv_and_aiwaf_event_log_writers(tmp_path):
    from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware

    middleware = AIWAFLoggingMiddleware.__new__(AIWAFLoggingMiddleware)
    middleware.access_log_file = tmp_path / "access.csv"
    middleware.aiwaf_log_file = tmp_path / "aiwaf.log"
    request = SimpleNamespace(
        method="GET",
        url=SimpleNamespace(path="/admin", query=""),
        scope={"http_version": "1.1"},
        headers={"user-agent": "pytest"},
        client=SimpleNamespace(host="203.0.113.60"),
        state=SimpleNamespace(aiwaf_blocked=True, aiwaf_block_reason="test"),
    )
    response = SimpleNamespace(status_code=403, headers={})
    middleware._log_access_csv(request, response, time.time())
    middleware._log_aiwaf_event(request)
    assert (tmp_path / "access.csv").exists()
    assert "BLOCKED" in (tmp_path / "aiwaf.log").read_text(encoding="utf-8")


def test_combined_log_timestamp_is_utc_on_non_utc_server(tmp_path, monkeypatch):
    monkeypatch.setenv("TZ", "Etc/GMT+5")  # POSIX sign: UTC-5
    time.tzset()
    try:
        app = FastAPI()

        @app.get("/ok")
        async def ok():
            return {"ok": True}

        app.add_middleware(AIWAFLoggingMiddleware, log_dir=str(tmp_path), log_format="combined")
        client = TestClient(app)
        before = datetime.now(timezone.utc).replace(microsecond=0)
        client.get("/ok", headers={"user-agent": "Mozilla/5.0", "accept": "text/html"})
        after = datetime.now(timezone.utc)

        line = (tmp_path / "access.log").read_text(encoding="utf-8")
        stamp = line.split("[", 1)[1].split("]", 1)[0]
        logged = datetime.strptime(stamp, "%d/%b/%Y:%H:%M:%S %z")
        assert before <= logged <= after
    finally:
        monkeypatch.undo()
        time.tzset()
