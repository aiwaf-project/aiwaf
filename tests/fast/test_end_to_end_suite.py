from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF


def test_end_to_end_rate_limit_behavior():
    app = FastAPI()

    @app.get("/path-a")
    async def path_a():
        return {"a": True}

    @app.get("/path-b")
    async def path_b():
        return {"b": True}

    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={"enabled": True, "max_requests": 1, "window_seconds": 60},
        exemptions={"private_ips_exempted": False, "auto_exempt_patterns": []},
    )
    client = TestClient(app)

    assert client.get("/path-a").status_code == 200
    assert client.get("/path-a").status_code == 429
    # Current runtime behavior blocks the client IP after limit breach, so
    # subsequent paths from same client are denied.
    assert client.get("/path-b").status_code == 403
    assert client.get("/path-b").status_code == 403
