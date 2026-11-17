"""Correct behavior, in addition to regular/native agreement."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwaf.core.runtime_config import AIWAFConfig
from aiwaf.fast import AIWAF
from aiwaf.fast.middleware.anomaly_middleware import AIAnomalyMiddleware
from aiwaf.fast.middleware.ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware


def installed(app, kind):
    layer = app.middleware_stack
    while layer is not None:
        if isinstance(layer, kind):
            return layer
        layer = getattr(layer, "app", None)
    raise AssertionError(f"{kind.__name__} not installed")


def config_for(tmp_path):
    config = AIWAFConfig(load_from_env=False)
    config.set("exemptions.private_ips_exempted", False)
    config.set("logging_middleware.log_dir", str(tmp_path / "logs"))
    config.set("path_manifest", str(tmp_path / "missing.json"))
    return config


@pytest.mark.parametrize("native", [False, True])
def test_route_words_and_parameterized_mounts(tmp_path, monkeypatch, native):
    if native:
        pytest.importorskip("aiwaf_rust")
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    config = config_for(tmp_path)
    app = FastAPI()
    waf = AIWAF(app, config=config, middlewares=["ip_keyword_block", "ai_anomaly"], native_pipeline=native)
    # Routes registered after AIWAF still exist when the middleware stack is built.
    child = FastAPI()
    child.add_api_route("/catalogue/{item_id}", lambda item_id: {"ok": True})
    app.mount("/storefront", child)
    with TestClient(app, client=("198.51.100.18", 5000)) as client:
        assert client.get("/storefront/catalogue/42").status_code == 200
        # A dynamic keyword corresponding to an app route must not block legitimate traffic.
        from aiwaf.core.runtime_storage import get_keyword_store
        for _ in range(15):
            get_keyword_store().add_keyword("catalogue")
        assert client.get("/storefront/catalogue/43").status_code == 200
        assert client.get("/unknown/.env").status_code == 403
    assert bool(waf.native) == native
    if native:
        profile = installed(app, NativeAIWAFMiddleware).pipeline.profile
        assert "catalogue" in profile["legitimate_keywords"]
        assert "storefront" in profile["safe_prefixes"]
    else:
        keyword = installed(app, IPAndKeywordBlockMiddleware)
        assert "catalogue" in keyword.legitimate_path_keywords
        anomaly = installed(app, AIAnomalyMiddleware)
        assert anomaly._path_exists("/storefront/catalogue/42")
        assert not anomaly._path_exists("/storefront/missing")


@pytest.mark.parametrize("native", [False, True, "from_settings"])
@pytest.mark.parametrize("setting", ["ai_anomaly.model_path", "AIWAF_MODEL_PATH"])
def test_saved_model_is_loaded_and_used(tmp_path, monkeypatch, native, setting):
    rust = pytest.importorskip("aiwaf_rust")
    from aiwaf.core.model_artifacts import rust_model_artifact
    from aiwaf.core.model_serialization import dump_model_artifact
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    model = rust.IsolationForest(n_estimators=10, random_state=42)
    model.fit([[float(i), 0., .01, 0., 0., 0.] for i in range(20)])
    path = tmp_path / "model.json"
    dump_model_artifact(rust_model_artifact(model, list(range(6)), 20, "fastapi"), path)
    config = config_for(tmp_path)
    config.set(setting, str(path))
    app = FastAPI()
    app.add_api_route("/healthy", lambda: {"ok": True})
    if native == "from_settings":
        app.add_middleware(NativeAIWAFMiddleware, config={"from_settings": True, "enabled": ["ai_anomaly"]},
                           aiwaf_config=config)
    else:
        waf = AIWAF(app, config=config, middlewares=["ai_anomaly"], native_pipeline=native)
    with TestClient(app, client=("198.51.100.19", 5000)) as client:
        assert client.get("/healthy").status_code == 200
    if native != "from_settings":
        assert bool(waf.native) == native
    if native:
        pipeline = installed(app, NativeAIWAFMiddleware).pipeline
        assert "healthy" in pipeline.profile["legitimate_keywords"]
        engine = pipeline.native
        assert engine.diagnostics()[0] is True
        assert engine.diagnostics()[3] == 1
    else:
        middleware = installed(app, AIAnomalyMiddleware)
        assert middleware.model.to_json() == model.to_json()
        assert len(next(iter(middleware.request_cache.values()))) == 1


@pytest.mark.parametrize("contents", [None, "not json", '{"model_type": "unknown"}',
                                    '{"model_type": "aiwaf_rust.IsolationForest", "model_state": "bad"}'])
def test_bad_model_fails_open_with_warning(tmp_path, caplog, contents):
    from aiwaf.fast.runtime_policy import load_trained_model
    path = tmp_path / "model.json"
    if contents is not None:
        path.write_text(contents)
    config = config_for(tmp_path)
    config.set("ai_anomaly.model_path", str(path))
    assert load_trained_model(config) is None
    assert "Could not load FastAPI AI model" in caplog.text


@pytest.mark.parametrize("damage", ["feature_count", "missing_key", "missing_tree_key"])
def test_unusable_model_state_is_rejected(tmp_path, caplog, damage):
    rust = pytest.importorskip("aiwaf_rust")
    from aiwaf.core.model_artifacts import rust_model_artifact
    from aiwaf.core.model_serialization import dump_model_artifact
    from aiwaf.fast.runtime_policy import load_trained_model
    model = rust.IsolationForest(n_estimators=2, random_state=42)
    model.fit([[float(i)] * 6 for i in range(10)])
    artifact = rust_model_artifact(model, list(range(6)), 10, "fastapi")
    state = artifact["model_state"]
    if damage == "feature_count":
        state["nFeaturesIn_"] = 2
    elif damage == "missing_key":
        del state["verbose"]
    else:
        del state["trees"][0]["splitAttr"]
    path = tmp_path / "model.json"
    dump_model_artifact(artifact, path)
    config = config_for(tmp_path)
    config.set("ai_anomaly.model_path", str(path))
    assert load_trained_model(config) is None
    assert "Could not load FastAPI AI model" in caplog.text


def test_loaded_model_changes_scanning_verdicts_in_both_runtimes(tmp_path, monkeypatch):
    monkeypatch.delenv("AIWAF_NATIVE_PIPELINE", raising=False)
    rust = pytest.importorskip("aiwaf_rust")
    from aiwaf.core.model_artifacts import rust_model_artifact
    from aiwaf.core.model_serialization import dump_model_artifact
    from tests.fast.test_native_parity import alike
    model = rust.IsolationForest(n_estimators=20, random_state=42)
    model.fit([[float(i), 0., .01, 0., 0., 0.] for i in range(20)])
    path = tmp_path / "model.json"
    dump_model_artifact(rust_model_artifact(model, list(range(6)), 20, "fastapi"), path)
    cases = [[("get", f"/missing-probe-{i}.php", {}) for i in range(10)]]
    settings = {"ip_keyword_block.enabled": False}
    without_model = alike(tmp_path, monkeypatch, cases, **settings)
    with_model = alike(tmp_path, monkeypatch, cases, **settings, **{"ai_anomaly.model_path": str(path)})
    assert without_model == [[404] * 10]
    assert 403 in with_model[0]
