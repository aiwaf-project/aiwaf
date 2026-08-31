import os
import tempfile
import importlib.util
from pathlib import Path

import pytest

from aiwaf.flask import rust_backend
from aiwaf.core import header_validation

RUST_PACKAGE_INSTALLED = importlib.util.find_spec("aiwaf_rust") is not None


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
def test_real_rust_isolation_forest_surface_is_discoverable():
    assert rust_backend.rust_available() is True
    rust_class = rust_backend.get_isolation_forest_class()
    assert rust_class is not None

    model = rust_class(
        n_estimators=4,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=1,
        warm_start=False,
    )
    model.fit([[0.0], [0.1], [1.0], [1.1]])
    restored = rust_backend.load_isolation_forest(model.to_json())
    assert restored is not None


def test_python_header_validation_blocks_missing():
    environ = {"HTTP_USER_AGENT": "Mozilla/5.0"}
    reason = header_validation.validate_headers_python(environ)
    assert reason and "Missing required headers" in reason


def test_python_header_validation_allows_head_with_override():
    environ = {}
    reason = header_validation.validate_headers_python(
        environ,
        method="HEAD",
        config_required_headers={"HEAD": []},
    )
    assert reason is None


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
@pytest.mark.skipif(not rust_backend.rust_available(), reason="Rust extension not available")
def test_rust_validate_headers_blocks_missing():
    environ = {"HTTP_USER_AGENT": "Mozilla/5.0"}
    reason = rust_backend.validate_headers(environ)
    assert reason and "Missing required headers" in reason


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
@pytest.mark.skipif(not rust_backend.rust_available(), reason="Rust extension not available")
def test_rust_extract_features_basic():
    records = [
        {
            "ip": "1.2.3.4",
            "path_lower": "/wp-admin",
            "path_len": 9,
            "timestamp": 1000.0,
            "response_time": 0.1,
            "status_idx": 0,
            "kw_check": True,
            "total_404": 2,
        },
        {
            "ip": "1.2.3.4",
            "path_lower": "/home",
            "path_len": 5,
            "timestamp": 1005.0,
            "response_time": 0.2,
            "status_idx": 1,
            "kw_check": False,
            "total_404": 2,
        },
    ]
    features = rust_backend.extract_features(records, ["wp-admin", "sql"])
    assert features is not None
    assert len(features) == 2

    first = features[0]
    assert first["ip"] == "1.2.3.4"
    assert first["path_len"] == 9
    assert first["kw_hits"] >= 1
    assert first["burst_count"] >= 1
    assert first["total_404"] == 2

    second = features[1]
    assert second["path_len"] == 5
    assert second["kw_hits"] == 0


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
@pytest.mark.skipif(not rust_backend.rust_available(), reason="Rust extension not available")
def test_rust_analyze_recent_behavior_basic():
    entries = [
        {"path_lower": "/wp-admin", "timestamp": 1000.0, "status": 404, "kw_check": True},
        {"path_lower": "/wp-content", "timestamp": 1003.0, "status": 404, "kw_check": True},
        {"path_lower": "/home", "timestamp": 1006.0, "status": 200, "kw_check": False},
    ]
    result = rust_backend.analyze_recent_behavior(entries, ["wp-admin", "wp-content", "sql"])
    assert result is not None
    assert result["total_requests"] == 3
    assert result["max_404s"] == 2
    assert result["scanning_404s"] >= 1
    assert isinstance(result["should_block"], bool)


@pytest.mark.skipif(not RUST_PACKAGE_INSTALLED, reason="aiwaf_rust package not installed")
@pytest.mark.skipif(not rust_backend.rust_available(), reason="Rust extension not available")
def test_rust_write_csv_log_deprecated():
    temp_dir = Path(tempfile.mkdtemp(prefix="aiwaf_rust_test_"))
    path = temp_dir / "access.csv"
    headers = ["timestamp", "ip"]
    row = {"timestamp": "t", "ip": "127.0.0.1"}

    ok = rust_backend.write_csv_log(str(path), headers, row)
    assert not ok
    assert not path.exists()

    lock_path = path.with_suffix(".csv.lock")
    if lock_path.exists():
        os.remove(lock_path)
    if path.exists():
        os.remove(path)
    os.rmdir(temp_dir)


def test_supports_chunked_feature_extraction_with_fake_module(monkeypatch):
    class FakeRust:
        def extract_features_batch_with_state(self, records, static_keywords, state):
            return {"features": [], "state": state}

        def finalize_feature_state(self, static_keywords, state):
            return {"features": []}

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    assert rust_backend.supports_chunked_feature_extraction() is True


def test_extract_features_batch_and_finalize_with_fake_module(monkeypatch):
    calls = {"batch": [], "finalize": []}

    class FakeRust:
        def extract_features_batch_with_state(self, records, static_keywords, state):
            calls["batch"].append((records, static_keywords, state))
            return {
                "features": [{"ip": r["ip"], "path_len": r["path_len"]} for r in records],
                "state": {"seen": len(records) if state is None else state["seen"] + len(records)},
            }

        def finalize_feature_state(self, static_keywords, state):
            calls["finalize"].append((static_keywords, state))
            return {"features": [{"ip": "final", "path_len": 0}]}

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())

    features, state = rust_backend.extract_features_batch(
        [{"ip": "1.2.3.4", "path_len": 9}],
        ["wp-admin"],
        None,
    )
    assert features == [{"ip": "1.2.3.4", "path_len": 9}]
    assert state == {"seen": 1}

    final = rust_backend.finalize_feature_state(["wp-admin"], state)
    assert final == [{"ip": "final", "path_len": 0}]
    assert len(calls["batch"]) == 1
    assert len(calls["finalize"]) == 1


def test_validate_headers_prefers_config_api_when_available(monkeypatch):
    calls = {"with_config": 0, "plain": 0}

    class FakeRust:
        def validate_headers_with_config(self, headers, required_headers, min_score):
            calls["with_config"] += 1
            assert headers == {"HTTP_USER_AGENT": "ua"}
            assert required_headers == {"GET": ["User-Agent"]}
            assert min_score == 70
            return "ok"

        def validate_headers(self, headers):
            calls["plain"] += 1
            return "plain"

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    out = rust_backend.validate_headers(
        {"HTTP_USER_AGENT": "ua"},
        required_headers={"GET": ["User-Agent"]},
        min_score=70,
    )
    assert out == "ok"
    assert calls["with_config"] == 1
    assert calls["plain"] == 0


def test_validate_headers_falls_back_to_none_on_exception(monkeypatch):
    class FakeRust:
        def validate_headers(self, headers):
            raise RuntimeError("boom")

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    assert rust_backend.validate_headers({"HTTP_USER_AGENT": "ua"}) is None


def test_extract_features_batch_supports_tuple_return(monkeypatch):
    class FakeRust:
        def extract_features_batch_with_state(self, records, static_keywords, state):
            return [{"ip": "1.2.3.4", "path_len": 9}], {"seen": 1}

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    features, state = rust_backend.extract_features_batch([{"ip": "1.2.3.4", "path_len": 9}], ["wp-admin"])
    assert features == [{"ip": "1.2.3.4", "path_len": 9}]
    assert state == {"seen": 1}


def test_extract_features_batch_returns_none_for_invalid_contract(monkeypatch):
    class FakeRust:
        def extract_features_batch_with_state(self, records, static_keywords, state):
            return "not-a-valid-contract"

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    features, state = rust_backend.extract_features_batch([{"ip": "1.2.3.4", "path_len": 9}], ["wp-admin"], {"seen": 0})
    assert features is None
    assert state == {"seen": 0}


def test_extract_features_batch_returns_none_on_exception(monkeypatch):
    class FakeRust:
        def extract_features_batch_with_state(self, records, static_keywords, state):
            raise RuntimeError("boom")

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    features, state = rust_backend.extract_features_batch([{"ip": "1.2.3.4", "path_len": 9}], ["wp-admin"], {"seen": 0})
    assert features is None
    assert state == {"seen": 0}


def test_finalize_feature_state_returns_none_on_exception(monkeypatch):
    class FakeRust:
        def finalize_feature_state(self, static_keywords, state):
            raise RuntimeError("boom")

    monkeypatch.setattr(rust_backend, "aiwaf_rust", FakeRust())
    assert rust_backend.finalize_feature_state(["wp-admin"], {"seen": 1}) is None
