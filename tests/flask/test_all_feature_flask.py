from flask import Flask

from aiwaf.flask import AIWAF


def _make_app():
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["AIWAF_USE_CSV"] = True
    app.config["AIWAF_DATA_DIR"] = "test_data"
    return app


def test_all_auto_enables_logging_when_access_log_missing():
    app = _make_app()
    app.config["AIWAF_ACCESS_LOG"] = ""
    aiwaf = AIWAF(app, middlewares=["all"])
    assert aiwaf.is_middleware_enabled("logging")


def test_all_auto_disables_logging_when_access_log_configured():
    app = _make_app()
    app.config["AIWAF_ACCESS_LOG"] = "/var/log/access.log"
    aiwaf = AIWAF(app, middlewares=["all"])
    assert not aiwaf.is_middleware_enabled("logging")


def test_all_auto_enables_geo_when_block_countries_present(monkeypatch):
    app = _make_app()
    app.config["AIWAF_GEO_BLOCK_COUNTRIES"] = ["US"]

    @app.route("/geo")
    def geo():
        return "ok"

    monkeypatch.setattr("aiwaf.flask.geo_block_middleware.get_country_for_ip", lambda *_a, **_k: "US")

    aiwaf = AIWAF(app, middlewares=["all"])
    assert aiwaf.is_middleware_enabled("geo_block")
    assert app.config.get("AIWAF_GEO_BLOCK_ENABLED") is True

    with app.test_client() as client:
        resp = client.get("/geo")
        assert resp.status_code == 403


def test_all_auto_disables_uuid_tamper_when_no_uuid_routes():
    app = _make_app()

    @app.route("/plain")
    def plain():
        return "ok"

    aiwaf = AIWAF(app, middlewares=["all"])
    assert not aiwaf.is_middleware_enabled("uuid_tamper")


def test_all_auto_enables_uuid_tamper_when_uuid_routes_present():
    app = _make_app()

    @app.route("/items/<uuid:item_id>")
    def item(item_id):
        return str(item_id)

    aiwaf = AIWAF(app, middlewares=["all"])
    assert aiwaf.is_middleware_enabled("uuid_tamper")
