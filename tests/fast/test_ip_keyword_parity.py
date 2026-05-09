from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.fast import AIWAF
from aiwaf.fast.storage import get_keyword_store


def _app():
    app = FastAPI()

    @app.get("/safe")
    async def safe():
        return {"ok": True}

    AIWAF(app, header_validation={"enabled": False}, rate_limiting={"enabled": False}, honeypot={"enabled": False})
    return app


def test_fast_keyword_policy_parity_allow_safe_route():
    app = _app()
    client = TestClient(app)
    assert client.get("/safe").status_code == 200


def test_fast_keyword_policy_parity_block_static_malicious_path():
    app = _app()
    client = TestClient(app)
    assert client.get("/admin.php").status_code == 403


def test_fast_keyword_policy_parity_block_learned_keyword():
    app = _app()
    store = get_keyword_store()
    store.add_keyword("shellupload")
    client = TestClient(app)
    assert client.get("/shellupload").status_code == 403
