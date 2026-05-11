from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF
from aiwaf.fast.middleware.rate_limit_middleware import _AIWAF_CACHE


def test_fastapi_rate_limit_redis_backend_missing_url_falls_back():
    app = FastAPI()

    @app.get("/rl")
    async def rl():
        return {"ok": True}

    _AIWAF_CACHE.clear()
    AIWAF(
        app,
        header_validation={"enabled": False},
        honeypot={"enabled": False},
        rate_limiting={
            "enabled": True,
            "window_seconds": 60,
            "max_requests": 1,
            "flood_threshold": 100,
            "cache_backend": "redis",  # no redis_url configured, should fall back
        },
    )

    client = TestClient(app)
    assert client.get("/rl").status_code == 200
    assert client.get("/rl").status_code == 429

