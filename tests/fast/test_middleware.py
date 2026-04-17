from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def test_keyword_path_blocking_in_fast_middleware():
    app = FastAPI()

    @app.get("/safe")
    async def safe():
        return {"ok": True}

    AIWAF(app, header_validation={"enabled": False}, rate_limiting={"enabled": False})
    client = TestClient(app)
    response = client.get("/admin.php")
    assert response.status_code == 403

