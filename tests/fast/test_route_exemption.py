"""FastAPI route-level exemption behavior parity tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF, aiwaf_exempt


def test_aiwaf_exempt_route_bypasses_header_validation_and_rate_limiting():
    app = FastAPI()

    @app.get("/protected")
    async def protected():
        return {"ok": True}

    @app.get("/health")
    @aiwaf_exempt
    async def health():
        return {"status": "ok"}

    AIWAF(
        app,
        header_validation={"enabled": True, "block_suspicious": True, "quality_threshold": 3},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60},
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    # Exempt endpoint should pass even with suspicious headers.
    exempt_first = client.get("/health", headers={"user-agent": "curl/8.0.1", "accept": "*/*"})
    exempt_second = client.get("/health", headers={"user-agent": "curl/8.0.1", "accept": "*/*"})
    assert exempt_first.status_code == 200
    assert exempt_second.status_code == 200

    # Non-exempt endpoint should be blocked by header validation first.
    protected_resp = client.get("/protected", headers={"user-agent": "curl/8.0.1", "accept": "*/*"})
    assert protected_resp.status_code == 403
