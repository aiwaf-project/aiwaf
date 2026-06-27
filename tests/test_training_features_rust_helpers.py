from datetime import datetime, timezone
from types import SimpleNamespace

from aiwaf.core import training_features


def test_build_records_uses_rust_helper_when_available(monkeypatch):
    calls = {}

    def rust_build_records(parsed, ip_404, path_exists_fn, path_exempt_fn, status_idx_list):
        calls["args"] = (parsed, ip_404, path_exists_fn, path_exempt_fn, status_idx_list)
        return [{"ip": "1.1.1.1", "path_lower": "/rust"}]

    monkeypatch.setattr(
        training_features,
        "rust_backend",
        SimpleNamespace(build_records=rust_build_records),
    )

    result = training_features.build_records(
        [{"ip": "1.1.1.1", "path": "/x", "response_time": 1.0, "status": 404, "timestamp": datetime.now(timezone.utc)}],
        {"1.1.1.1": 1},
        lambda path: False,
        lambda path: False,
        [200, 404],
    )

    assert result == [{"ip": "1.1.1.1", "path_lower": "/rust"}]
    assert calls["args"][1] == {"1.1.1.1": 1}


def test_build_records_falls_back_when_rust_helper_returns_none(monkeypatch):
    monkeypatch.setattr(
        training_features,
        "rust_backend",
        SimpleNamespace(build_records=lambda *args: None),
    )
    timestamp = datetime.fromtimestamp(10, timezone.utc)

    result = training_features.build_records(
        [{"ip": "1.1.1.1", "path": "/wp-login.php", "response_time": 0.2, "status": 404, "timestamp": timestamp}],
        {"1.1.1.1": 3},
        lambda path: False,
        lambda path: False,
        [200, 404],
    )

    assert result[0]["path_lower"] == "/wp-login.php"
    assert result[0]["timestamp_epoch"] == 10
    assert result[0]["kw_check"] is True
    assert result[0]["status_idx"] == 1


def test_rust_payload_from_records_uses_rust_helper(monkeypatch):
    monkeypatch.setattr(
        training_features,
        "rust_backend",
        SimpleNamespace(rust_payload_from_records=lambda records: [{"from": "rust"}]),
    )

    assert training_features.rust_payload_from_records([{"ip": "1.1.1.1"}]) == [{"from": "rust"}]


def test_python_feature_from_record_uses_rust_helper(monkeypatch):
    monkeypatch.setattr(
        training_features,
        "rust_backend",
        SimpleNamespace(python_feature_from_record=lambda record, ip_times, static_kw: {"from": "rust"}),
    )

    assert training_features.python_feature_from_record({}, {}, []) == {"from": "rust"}


def test_python_features_batched_uses_rust_helper(monkeypatch):
    monkeypatch.setattr(
        training_features,
        "rust_backend",
        SimpleNamespace(python_features_batched=lambda *args: [{"from": "rust"}]),
    )

    result = training_features.python_features_batched([{"ip": "1.1.1.1"}], {}, [], lambda rows, size: [rows], 1, False, 1, 1)

    assert result == [{"from": "rust"}]
