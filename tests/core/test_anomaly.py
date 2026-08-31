from aiwaf.core import anomaly
from aiwaf.core.anomaly import (
    analyze_recent_behavior,
    analyze_recent_behavior_python,
    build_feature_vector,
    compute_kw_hits,
    evaluate_anomaly,
    extract_segments,
    predict_anomaly,
    trim_history,
)


def test_compute_kw_hits_counts_each_matching_keyword_once():
    assert compute_kw_hits("/wp-admin/login.php", ["wp-", "admin", ".php", ""]) == 3


def test_trim_history_applies_window_and_minimum_window():
    history = [(8.5, "/old", 200, 0.1), (9.5, "/new", 200, 0.1)]
    assert trim_history(history, now=10.0, window_seconds=1.0) == [history[1]]
    assert trim_history(None, now=10.0, window_seconds=0.0) == []


def test_extract_segments_normalizes_and_filters_short_segments():
    assert extract_segments("/API/Evil-Shell/x") == ["evil", "shell"]


def test_analyze_recent_behavior_python_blocks_scanning_keywords():
    now = 1000.0
    recent = [
        (now - 1, "/wp-admin", 404, 0.01),
        (now - 2, "/xmlrpc.php", 404, 0.01),
        (now - 3, "/.env", 404, 0.01),
        (now - 4, "/phpmyadmin", 404, 0.01),
        (now - 5, "/wp-includes", 404, 0.01),
    ]

    stats = analyze_recent_behavior_python(
        recent,
        static_keywords=["wp-", "xmlrpc", ".env", "phpmyadmin"],
        path_exists=lambda _p: False,
        is_exempt_path=lambda _p: False,
    )
    assert stats.max_404s == 5
    assert stats.should_block is True


def test_analyze_recent_behavior_uses_python_for_empty_rust_result(monkeypatch):
    monkeypatch.setattr(anomaly, "rust_available", lambda: True)
    monkeypatch.setattr(anomaly, "rust_analyze_recent_behavior", lambda *_args: None)
    recent = [(999.0, "/safe", 200, 0.01)]

    stats = analyze_recent_behavior(
        recent,
        static_keywords=[],
        path_exists=lambda _path: True,
        is_exempt_path=lambda _path: False,
    )

    assert stats is not None
    assert stats.total_requests == 1
    assert stats.should_block is False


def test_analyze_recent_behavior_maps_rust_result(monkeypatch):
    monkeypatch.setattr(anomaly, "rust_available", lambda: True)
    monkeypatch.setattr(
        anomaly,
        "rust_analyze_recent_behavior",
        lambda *_args: {
            "avg_kw_hits": 4,
            "max_404s": 6,
            "avg_burst": 7,
            "total_requests": 8,
            "scanning_404s": 5,
            "legitimate_404s": 1,
            "should_block": True,
        },
    )

    stats = analyze_recent_behavior(
        [(999.0, "/wp-admin", 404, 0.01)],
        static_keywords=["wp-"],
        path_exists=lambda _path: False,
        is_exempt_path=lambda _path: False,
    )

    assert stats is not None
    assert stats.should_block is True
    assert stats.scanning_404s == 5


def test_analyze_recent_behavior_returns_none_without_history():
    assert analyze_recent_behavior(
        [],
        static_keywords=[],
        path_exists=lambda _path: False,
        is_exempt_path=lambda _path: False,
    ) is None


def test_predict_anomaly_handles_none_model_and_prediction_errors():
    class BrokenModel:
        def predict(self, _features):
            raise RuntimeError("broken")

    assert predict_anomaly(None, [1, 2]) is None
    assert predict_anomaly(BrokenModel(), [1, 2]) is None


def test_predict_anomaly_uses_python_model(monkeypatch):
    class Model:
        def predict(self, features):
            assert features.shape == (1, 2)
            return [-1]

    monkeypatch.setattr(anomaly, "is_rust_isolation_forest", lambda _model: False)
    assert predict_anomaly(Model(), [1, 2]) == -1


def test_build_feature_vector_computes_each_feature():
    features = build_feature_vector(
        path="/wp-admin",
        status_code=404,
        response_time=0.25,
        now=100.0,
        history=[(95.0, "/old", 404, 0.1), (50.0, "/older", 200, 0.2)],
        static_keywords=["wp-", "admin"],
        status_index_values=["200", "404"],
        path_exists_current=False,
        is_exempt_path_current=False,
    )

    assert features == [9.0, 2.0, 0.25, 1.0, 1.0, 1.0]


def test_evaluate_anomaly_learns_keywords_only_on_404_and_missing_path():
    now = 1000.0
    outcome = evaluate_anomaly(
        ip="1.2.3.4",
        path="/not-a-real/evil-shell",
        status_code=404,
        response_time=0.2,
        now=now,
        history=[],
        window_seconds=60,
        model=None,
        static_keywords=[".php", "xmlrpc"],
        malicious_keywords=[".php", "xmlrpc"],
        keyword_learning_enabled=True,
        path_exists=lambda _p: False,
        is_exempt_path=lambda _p: False,
        is_malicious_context=lambda seg: seg == "shell",
        legitimate_keywords={"health"},
    )
    assert "shell" in outcome.learned_keywords
