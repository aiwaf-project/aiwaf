"""AIWAF_PROFILE=monitor for Flask lets through what it would block, and logs it."""
import logging

import pytest
from flask import Flask
from flask.testing import FlaskClient

from aiwaf.core import profiles


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    monkeypatch.delenv("AIWAF_PROFILE", raising=False)
    yield
    profiles.activate("balanced")


def app_for(tmp_path, profile, **config):
    from aiwaf.flask import AIWAF

    app = Flask(__name__)
    app.config.update(AIWAF_PROFILE=profile, AIWAF_DATA_DIR=str(tmp_path), AIWAF_EXEMPT_IPS=[],
                      AIWAF_LOG_DIR=str(tmp_path), **config)

    @app.route("/ok")
    def ok():
        return "ok"

    AIWAF(app, native_pipeline=False)
    return app


def get(app, path, ip):
    return FlaskClient.open(app.test_client(), path, environ_base={"REMOTE_ADDR": ip}).status_code


def test_monitor_lets_a_probe_through_and_logs_it(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
        app = app_for(tmp_path, "monitor")
        assert [get(app, "/.env", "203.0.113.21") for _ in range(2)] == [404, 404]
    assert "would block GET /.env from 203.0.113.21 with 403 (IPAndKeywordBlockMiddleware" in caplog.text
    assert "would add 203.0.113.21 to the blocklist" in caplog.text


def test_strict_halves_flasks_rate_limits_and_still_blocks(tmp_path):
    app = app_for(tmp_path, "strict")
    assert (app.config["AIWAF_RATE_MAX"], app.config["AIWAF_RATE_FLOOD"]) == (10, 20)
    assert get(app, "/.env", "203.0.113.22") == 403
    assert app_for(tmp_path, "strict", AIWAF_RATE_MAX=7).config["AIWAF_RATE_MAX"] == 7


def test_after_request_blocks_are_undone_and_explicit_native_is_refused(tmp_path, caplog):
    from aiwaf.flask.csv_pipeline import NativeAIWAF
    from aiwaf.flask.monitor import _monitor_after, _monitor_before

    app = Flask(__name__)
    with app.test_request_context("/x"):
        original = app.response_class("fine", 200)
        blocked = app.response_class("no", 403)
        with caplog.at_level(logging.WARNING, "aiwaf.monitor"):
            assert _monitor_after(lambda response: blocked, "UUID")(original) is original
            assert _monitor_after(lambda response: response, "UUID")(original) is original
            assert _monitor_before(lambda: ("blocked", 429), "Rate")() is None
            assert _monitor_before(lambda: None, "Rate")() is None
        assert "with 403 (UUID: no)" in caplog.text and "with 429 (Rate: blocked)" in caplog.text

        def redirect():
            from werkzeug.exceptions import HTTPException
            error = HTTPException()
            error.code = 302
            raise error
        with pytest.raises(Exception):
            _monitor_before(redirect, "Rate")()
    app.config["AIWAF_PROFILE"] = "monitor"
    with pytest.raises(RuntimeError, match="AIWAF_PROFILE=monitor runs the regular middleware"):
        NativeAIWAF(app)
