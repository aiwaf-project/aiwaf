from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def test_path_rules_can_disable_header_validation_for_prefix():
    app = FastAPI()

    @app.get("/api/data")
    async def data():
        return {"ok": True}

    AIWAF(
        app,
        path_rules=[{"PREFIX": "/api/", "DISABLE": ["header_validation"]}],
    )
    client = TestClient(app)
    resp = client.get("/api/data", headers={"user-agent": "", "accept": "text/html"})
    assert resp.status_code == 200

