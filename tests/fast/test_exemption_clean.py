from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def test_private_ip_exemption_allows_fast_requests():
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    AIWAF(app, exemptions={"private_ips_exempted": True, "auto_exempt_patterns": ["127.0.0.1"]})
    client = TestClient(app)
    response = client.get("/health", headers={"user-agent": "Mozilla/5.0", "accept": "text/html"})
    assert response.status_code == 200

