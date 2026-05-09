import json
from flask import Flask

from aiwaf.flask.logging_middleware import AIWAFLoggingMiddleware


def test_flask_logging_json_includes_normalized_context(tmp_path):
    app = Flask(__name__)
    app.config["AIWAF_LOG_DIR"] = str(tmp_path)
    app.config["AIWAF_LOG_FORMAT"] = "json"

    @app.route("/ok")
    def ok():
        return "ok"

    AIWAFLoggingMiddleware(app)

    with app.test_client() as client:
        resp = client.get("/ok?a=1", headers={"User-Agent": "UA-flask", "Referer": "https://ref.flask/"})
        assert resp.status_code == 200

    access_log = tmp_path / "access.log"
    lines = access_log.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["path"] == "/ok"
    assert payload["query_string"] == "a=1"
    assert payload["referer"] == "https://ref.flask/"
    assert payload["user_agent"] == "UA-flask"
