from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def test_logging_enabled_through_aiwaf_core(tmp_path):
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    AIWAF(
        app,
        logging_middleware={"enabled": True, "log_dir": str(tmp_path), "log_format": "combined"},
    )
    client = TestClient(app)
    resp = client.get("/ping", headers={"user-agent": "Mozilla/5.0", "accept": "text/html"})
    assert resp.status_code == 200
    assert (tmp_path / "access.log").exists()

