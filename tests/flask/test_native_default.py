"""Flask's AIWAF(app) runs as the native pipeline when every setting carries over."""
import logging

import pytest
from flask import Flask

rust = pytest.importorskip("aiwaf_rust")
from aiwaf.core import native_settings  # noqa: E402
from aiwaf.core.native_settings import DEFAULT_FEATURES  # noqa: E402

FULL_ENGINE = all(rust.NativeCSVPipeline.supports(f) for f in DEFAULT_FEATURES) \
    if hasattr(rust.NativeCSVPipeline, "supports") else False
needs_full_engine = pytest.mark.skipif(not FULL_ENGINE, reason="aiwaf-rust with every native feature required")


@pytest.fixture
def make_app(tmp_path, monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)

    def make(**config):
        app = Flask(__name__)
        app.config.update(AIWAF_DATA_DIR=str(tmp_path / "data"), AIWAF_RATE_MAX=1, AIWAF_RATE_FLOOD=10,
                          AIWAF_LOG_DIR=str(tmp_path / "logs"),
                          AIWAF_PATH_MANIFEST=str(tmp_path / "none.json"), **config)
        app.add_url_rule("/items", "items", lambda: {"ok": True})
        return app
    return make


@needs_full_engine
def test_aiwaf_runs_as_native_when_everything_carries_over(make_app, caplog):
    from aiwaf.flask import AIWAF

    app = make_app()
    with caplog.at_level(logging.INFO, "aiwaf.native"):
        aiwaf = AIWAF(app)
    assert aiwaf.native is not None and aiwaf.middleware_instances == {}
    assert "running the native pipeline for Flask" in caplog.text
    # The stages the regular middleware would register, logging where its trainer reads.
    assert "logging" in aiwaf.native.pipeline.profile["enabled"]
    assert aiwaf.native.pipeline.profile["log_path"] == str(app.config["AIWAF_LOG_DIR"]) + "/aiwaf_native_access.csv"
    client = app.test_client()
    statuses = [client.get("/items", environ_base={"REMOTE_ADDR": "198.51.100.41"}).status_code for _ in range(2)]
    assert statuses == [200, 429]


@needs_full_engine
@pytest.mark.parametrize("how", ["argument", "config", "difference", "database"])
def test_aiwaf_stays_regular_when_asked_or_when_settings_differ(make_app, how):
    from aiwaf.flask import AIWAF

    app = make_app(**({"AIWAF_NATIVE_PIPELINE": False} if how == "config" else {})
                   , **({"AIWAF_SOMETHING_UNKNOWN": 1} if how == "difference" else {}))
    if how == "database":
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
    aiwaf = AIWAF(app, native_pipeline=False if how == "argument" else None,
                  use_database=True if how == "database" else None)
    assert aiwaf.native is None and aiwaf.middleware_instances


def test_an_engine_without_every_feature_stays_regular(make_app, monkeypatch):
    from aiwaf.flask import AIWAF

    monkeypatch.setattr(native_settings, "_engine_supports", lambda feature: False)
    assert AIWAF(make_app()).native is None


@needs_full_engine
def test_the_regular_model_carries_over_only_if_aiwaf_rust_made_it(make_app, tmp_path):
    import json
    from aiwaf.flask import AIWAF
    from aiwaf.flask.flask_integration import AIWAF as Integration

    forest = rust.IsolationForest(n_estimators=5, random_state=1)
    forest.fit([[float(i), 0.0, 0.0, 1.0, 1.0, 0.0] for i in range(16)])
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"model_type": "aiwaf_rust.IsolationForest", "model_state": forest.to_json()}))
    app = make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_FORCE_AI=True)
    assert native_settings.is_rust_model(Integration._regular_model(app))
    assert AIWAF(app).native is not None
    # Held back by the regular middleware until there are enough logs, so it stays regular.
    held = make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_MIN_AI_LOGS=10**9)
    assert AIWAF(held).native is None
    path.write_text(json.dumps({"model": {"kind": "sklearn"}}))
    assert AIWAF(make_app(AIWAF_MODEL_PATH=str(path), AIWAF_ALLOW_CUSTOM_MODEL_PATH=True, AIWAF_FORCE_AI=True)).native is None
    path.write_text("{")
    assert Integration._regular_model(app) is None
    assert Integration._regular_model(make_app(AIWAF_MODEL_PATH=str(tmp_path / "missing.json"))) is None
