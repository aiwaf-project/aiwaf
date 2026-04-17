from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware


def test_logging_client_records_error_log_for_4xx(tmp_path):
    app = FastAPI()

    @app.get("/missing")
    async def missing():
        return Response(status_code=404)

    app.add_middleware(AIWAFLoggingMiddleware, log_dir=str(tmp_path), log_format="json")
    client = TestClient(app)
    resp = client.get("/missing", headers={"user-agent": "Mozilla/5.0", "accept": "text/html"})
    assert resp.status_code == 404
    assert (tmp_path / "error.log").exists()

