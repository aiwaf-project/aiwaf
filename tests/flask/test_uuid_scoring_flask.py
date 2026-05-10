from flask import Flask

from aiwaf.flask.uuid_tamper_middleware import UUIDTamperMiddleware
from aiwaf.core.uuid_tamper import clear_uuid_score_state


def test_uuid_404_scoring_blocks_after_threshold():
    clear_uuid_score_state()
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["AIWAF_UUID_SCORE_WINDOW_SECONDS"] = 60
    app.config["AIWAF_UUID_SCORE_BLOCK_THRESHOLD"] = 6
    app.config["AIWAF_UUID_SCORE_NOT_FOUND_WEIGHT"] = 1
    app.config["AIWAF_EXEMPT_PATHS"] = set()
    UUIDTamperMiddleware(app)

    @app.route("/missing")
    def missing():
        return "not found", 404

    uid = "550e8400-e29b-41d4-a716-446655440000"
    with app.test_client() as client:
        for _ in range(5):
            resp = client.get(f"/missing?uuid={uid}")
            assert resp.status_code == 404
        resp = client.get(f"/missing?uuid={uid}")
        assert resp.status_code == 403
