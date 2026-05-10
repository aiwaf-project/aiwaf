from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

from aiwaf.fast import AIWAF
from aiwaf.core.uuid_tamper import clear_uuid_score_state


def test_uuid_404_scoring_blocks_after_threshold():
    clear_uuid_score_state()
    app = FastAPI()

    @app.get("/missing")
    async def missing():
        return JSONResponse({"ok": False}, status_code=404)

    AIWAF(
        app,
        header_validation={"enabled": False},
        rate_limiting={"enabled": False},
        honeypot={"enabled": False},
        ip_keyword_block={"enabled": False},
        ai_anomaly={"enabled": False},
        AIWAF_UUID_SCORE_WINDOW_SECONDS=60,
        AIWAF_UUID_SCORE_BLOCK_THRESHOLD=6,
        AIWAF_UUID_SCORE_NOT_FOUND_WEIGHT=1,
    )
    client = TestClient(app)
    uid = "550e8400-e29b-41d4-a716-446655440000"
    for _ in range(5):
        resp = client.get(f"/missing?uuid={uid}")
        assert resp.status_code == 404
    resp = client.get(f"/missing?uuid={uid}")
    assert resp.status_code == 403
