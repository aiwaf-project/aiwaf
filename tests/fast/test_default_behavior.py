from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF

BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "accept": "text/html,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip, deflate",
    "connection": "keep-alive",
}


def test_default_aiwaf_allows_valid_request():
    app = FastAPI()

    @app.get("/test")
    async def test():
        return {"ok": True}

    AIWAF(app)
    client = TestClient(app)
    resp = client.get("/test", headers=BROWSER_HEADERS)
    assert resp.status_code == 200
