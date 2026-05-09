from flask import Flask

from aiwaf.flask import AIWAF
from aiwaf.flask.storage import get_keyword_store


def _app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        {
            "TESTING": True,
            "AIWAF_USE_CSV": True,
            "AIWAF_DATA_DIR": str(tmp_path),
            "AIWAF_EXEMPT_PATHS": set(),
            "AIWAF_ENABLE_LOGGING": False,
            "AIWAF_RATE_WINDOW": 60,
            "AIWAF_RATE_MAX": 100,
            "AIWAF_RATE_FLOOD": 200,
        }
    )

    @app.route("/safe")
    def safe():
        return "OK"

    AIWAF(app, middlewares=["ip_keyword_block"])
    return app


def test_flask_keyword_policy_parity_allow_safe_route(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    assert client.get("/safe", headers={"User-Agent": "Test Browser 1.0"}).status_code == 200


def test_flask_keyword_policy_parity_block_static_malicious_path(tmp_path):
    app = _app(tmp_path)
    client = app.test_client()
    assert client.get("/admin.php", headers={"User-Agent": "Test Browser 1.0"}).status_code == 403


def test_flask_keyword_policy_parity_block_learned_keyword(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        store = get_keyword_store()
        store.add_keyword("shellupload")
    client = app.test_client()
    assert client.get("/shellupload", headers={"User-Agent": "Test Browser 1.0"}).status_code == 403
