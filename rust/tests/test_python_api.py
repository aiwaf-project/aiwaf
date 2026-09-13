import aiwaf_rust
from datetime import datetime, timezone


def test_keyword_matcher_preserves_literal_order():
    matcher = aiwaf_rust.KeywordMatcher([".env", "admin"])
    assert matcher.first_match("/admin/.env") == ".env"
    assert matcher.first_match("/safe") is None


def test_validate_headers():
    reason = aiwaf_rust.validate_headers({
        "HTTP_USER_AGENT": "Mozilla/5.0",
        "HTTP_ACCEPT": "text/html",
        "HTTP_ACCEPT_LANGUAGE": "en-US",
        "HTTP_ACCEPT_ENCODING": "gzip, deflate",
        "HTTP_CONNECTION": "keep-alive",
    })
    assert reason is None


def test_validate_headers_with_config():
    reason = aiwaf_rust.validate_headers_with_config(
        {
            "HTTP_USER_AGENT": "Mozilla/5.0",
            "HTTP_ACCEPT_LANGUAGE": "en-US",
            "HTTP_ACCEPT_ENCODING": "gzip, deflate",
            "HTTP_CONNECTION": "keep-alive",
        },
        ["HTTP_USER_AGENT"],
        0,
    )
    assert reason is None


def test_full_default_header_policy():
    assert aiwaf_rust.validate_headers_full_default({}, [], 0) is None
    reason = aiwaf_rust.validate_headers_full_default({"HTTP_ACCEPT": "text/html"}, ["HTTP_X_AUTH"], 0)
    assert reason == "Missing required headers: x-auth"


def test_extract_features_and_state():
    records = [
        {
            "ip": "1.2.3.4",
            "path_lower": "/wp-admin",
            "path_len": 9,
            "timestamp": 10.0,
            "response_time": 0.03,
            "status_idx": 3,
            "kw_check": True,
            "total_404": 5,
        }
    ]
    feats = aiwaf_rust.extract_features(records, ["wp"])
    assert len(feats) == 1
    assert feats[0]["kw_hits"] >= 1

    training_feats = aiwaf_rust.extract_training_features(records, ["wp"])
    assert training_feats[0]["burst_count"] == 1

    batch = aiwaf_rust.extract_features_batch_with_state(records, ["wp"], None)
    assert "features" in batch
    assert "state" in batch


def test_training_record_helpers():
    parsed = [
        {
            "ip": "1.2.3.4",
            "path": "/wp-admin",
            "response_time": 0.03,
            "status": 404,
            "timestamp": datetime.fromtimestamp(10, timezone.utc),
        },
        {
            "ip": "1.2.3.4",
            "path": "/wp-admin",
            "response_time": 0.04,
            "status": 500,
            "timestamp": datetime.fromtimestamp(12, timezone.utc),
        },
    ]
    exists_calls = []
    exempt_calls = []

    def path_exists(path):
        exists_calls.append(path)
        return False

    def path_exempt(path):
        exempt_calls.append(path)
        return False

    records = aiwaf_rust.build_records(
        parsed,
        {"1.2.3.4": 2},
        path_exists,
        path_exempt,
        [200, 404],
    )
    assert len(exists_calls) == 1
    assert len(exempt_calls) == 1
    assert records[0]["path_lower"] == "/wp-admin"
    assert records[0]["path_len"] == 9
    assert records[0]["status_idx"] == 1
    assert records[1]["status_idx"] == -1
    assert records[0]["kw_check"] is True
    assert records[0]["total_404"] == 2

    payload = aiwaf_rust.rust_payload_from_records(records)
    assert payload[0]["timestamp"] == 10.0
    assert payload[0]["response_time"] == 0.03

    feature = aiwaf_rust.python_feature_from_record(
        records[0],
        {"1.2.3.4": [datetime.fromtimestamp(1, timezone.utc), datetime.fromtimestamp(20, timezone.utc)]},
        ["wp"],
    )
    assert feature["kw_hits"] == 1
    assert feature["burst_count"] == 2

    batched = aiwaf_rust.python_features_batched(records, {"1.2.3.4": [datetime.fromtimestamp(10, timezone.utc)]}, ["wp"], lambda rows, size: [rows], 1, False, 1, 1)
    assert len(batched) == 2


def test_analyze_recent_behavior():
    entries = [
        {
            "path_lower": "/wp-admin",
            "timestamp": 1.0,
            "status": 404,
            "kw_check": True,
        }
        for _ in range(12)
    ]
    res = aiwaf_rust.analyze_recent_behavior(entries, ["wp"])
    assert res is not None
    assert res["should_block"] in (True, False)


def test_isolation_forest_roundtrip():
    forest = aiwaf_rust.IsolationForest(
        n_estimators=10,
        max_samples=8,
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=7,
        warm_start=False,
    )
    forest.fit([[0.0], [1.0], [2.0], [10.0]])
    score = forest.anomaly_score([10.0])
    assert score >= 0.0

    state = forest.to_json()
    forest2 = aiwaf_rust.IsolationForest.from_json(state)
    forest2.retrain([[0.1], [0.2], [0.3], [9.5]])
    preds = forest2.predict([[0.1], [9.5]])
    assert len(preds) == 2
