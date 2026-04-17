"""
Tests for Rust backend wrapper helpers.
"""
import importlib.util
from types import SimpleNamespace

import aiwaf.core.rust_backend as rust_backend


def test_aiwaf_rust_package_is_installed():
    assert importlib.util.find_spec("aiwaf_rust") is not None, (
        "aiwaf_rust must be installed for Rust backend tests"
    )
    assert rust_backend.rust_available() is True


def test_validate_headers_uses_config_api_when_available(monkeypatch):
    backend = SimpleNamespace(
        validate_headers_with_config=lambda headers, required_headers, min_score: "bad headers",
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    reason = rust_backend.validate_headers(
        {"user-agent": "x"},
        required_headers=["user-agent", "accept"],
        min_score=3,
    )
    assert reason == "bad headers"


def test_validate_headers_falls_back_to_legacy_api(monkeypatch):
    backend = SimpleNamespace(
        validate_headers=lambda headers: "legacy bad headers",
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    reason = rust_backend.validate_headers({"user-agent": "x"})
    assert reason == "legacy bad headers"


def test_validate_headers_returns_none_on_backend_error(monkeypatch):
    def _boom(headers):
        raise RuntimeError("backend failure")

    backend = SimpleNamespace(validate_headers=_boom)
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.validate_headers({"user-agent": "x"}) is None


def test_validate_headers_returns_none_when_backend_missing(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", None)
    assert rust_backend.validate_headers({"user-agent": "x"}) is None


def test_validate_headers_prefers_config_api_over_legacy(monkeypatch):
    calls = {"config": 0, "legacy": 0}

    def _config(headers, required_headers, min_score):
        calls["config"] += 1
        return None

    def _legacy(headers):
        calls["legacy"] += 1
        return "legacy"

    backend = SimpleNamespace(
        validate_headers_with_config=_config,
        validate_headers=_legacy,
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    result = rust_backend.validate_headers({"accept": "*/*"}, ["accept"], 2)
    assert result is None
    assert calls == {"config": 1, "legacy": 0}


def test_extract_features_returns_none_when_backend_missing(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", None)
    assert rust_backend.extract_features([{"a": 1}], ["k"]) is None


def test_extract_features_returns_payload(monkeypatch):
    backend = SimpleNamespace(
        extract_features=lambda records, static_keywords: [[1, 0, 1]],
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.extract_features([{"a": 1}], ["k"]) == [[1, 0, 1]]


def test_extract_features_returns_none_on_backend_error(monkeypatch):
    def _boom(records, static_keywords):
        raise RuntimeError("boom")

    backend = SimpleNamespace(extract_features=_boom)
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.extract_features([{"a": 1}], ["k"]) is None


def test_extract_features_batch_accepts_dict_result(monkeypatch):
    backend = SimpleNamespace(
        extract_features_batch_with_state=lambda records, keywords, state: {
            "features": [[1, 2, 3]],
            "state": {"pending": 1},
        },
        finalize_feature_state=lambda keywords, state: {"features": [[9, 9, 9]]},
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)

    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state={"x": 1})
    assert features == [[1, 2, 3]]
    assert state == {"pending": 1}
    assert rust_backend.finalize_feature_state(["k"], state) == [[9, 9, 9]]


def test_extract_features_batch_accepts_tuple_result(monkeypatch):
    backend = SimpleNamespace(
        extract_features_batch_with_state=lambda records, keywords, state: ([[4, 5]], {"y": 2}),
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)

    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state=None)
    assert features == [[4, 5]]
    assert state == {"y": 2}


def test_extract_features_batch_returns_none_when_backend_missing(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", None)
    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state={"s": 1})
    assert features is None
    assert state == {"s": 1}


def test_extract_features_batch_returns_none_when_api_not_supported(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", SimpleNamespace())
    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state={"s": 1})
    assert features is None
    assert state == {"s": 1}


def test_extract_features_batch_returns_none_for_unexpected_shape(monkeypatch):
    backend = SimpleNamespace(
        extract_features_batch_with_state=lambda records, keywords, state: {"invalid": True},
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state={"s": 1})
    assert features is None
    assert state is None


def test_extract_features_batch_returns_none_on_backend_error(monkeypatch):
    def _boom(records, keywords, state):
        raise RuntimeError("boom")

    backend = SimpleNamespace(extract_features_batch_with_state=_boom)
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    features, state = rust_backend.extract_features_batch([{"a": 1}], ["k"], state={"s": 1})
    assert features is None
    assert state == {"s": 1}


def test_chunked_support_flags(monkeypatch):
    backend = SimpleNamespace()
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.supports_chunked_feature_extraction() is False

    backend.extract_features_batch_with_state = lambda records, keywords, state: ([], state)
    backend.finalize_feature_state = lambda keywords, state: []
    assert rust_backend.supports_chunked_feature_extraction() is True


def test_finalize_feature_state_returns_none_when_backend_missing(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", None)
    assert rust_backend.finalize_feature_state(["k"], state={"s": 1}) is None


def test_finalize_feature_state_returns_none_when_api_not_supported(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", SimpleNamespace())
    assert rust_backend.finalize_feature_state(["k"], state={"s": 1}) is None


def test_finalize_feature_state_returns_raw_result(monkeypatch):
    backend = SimpleNamespace(finalize_feature_state=lambda keywords, state: [[7, 8, 9]])
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.finalize_feature_state(["k"], state={"s": 1}) == [[7, 8, 9]]


def test_finalize_feature_state_returns_none_on_backend_error(monkeypatch):
    def _boom(keywords, state):
        raise RuntimeError("boom")

    backend = SimpleNamespace(finalize_feature_state=_boom)
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.finalize_feature_state(["k"], state={"s": 1}) is None


def test_analyze_recent_behavior_returns_none_when_backend_missing(monkeypatch):
    monkeypatch.setattr(rust_backend, "aiwaf_rust", None)
    assert rust_backend.analyze_recent_behavior([{"a": 1}], ["k"]) is None


def test_analyze_recent_behavior_returns_payload(monkeypatch):
    backend = SimpleNamespace(
        analyze_recent_behavior=lambda entries, static_keywords: {"risk": 0.1},
    )
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.analyze_recent_behavior([{"a": 1}], ["k"]) == {"risk": 0.1}


def test_analyze_recent_behavior_returns_none_on_backend_error(monkeypatch):
    def _boom(entries, static_keywords):
        raise RuntimeError("boom")

    backend = SimpleNamespace(analyze_recent_behavior=_boom)
    monkeypatch.setattr(rust_backend, "aiwaf_rust", backend)
    assert rust_backend.analyze_recent_behavior([{"a": 1}], ["k"]) is None
