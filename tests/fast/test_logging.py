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

