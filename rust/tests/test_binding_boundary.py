"""Boundary coverage for the PyO3 layer: argument conversion, return shapes and errors.

Logic edge cases live in the Rust unit tests. These tests check what only Python can see.
"""
from datetime import datetime, timezone
import gzip
import inspect
import json
from pathlib import Path

import aiwaf_rust
import pytest

HEADERS = ["ip", "reason"]
BLOCK_HEADERS = [
    "ip", "reason", "reputation", "added_date", "info", "score", "offenses",
    "blocked_at", "expires_at", "duration", "permanent", "reasons",
]
EMPTY_EVIDENCE = [("1.1.1.1", "t1", "/x", "404"), ("1.1.1.1", "t2", "/y", "404")]
FEATURE_RECORD = {
    "ip": "1.2.3.4",
    "path_lower": "/wp-admin",
    "path_len": 9,
    "timestamp": 10.0,
    "response_time": 0.03,
    "status_idx": 3,
    "kw_check": True,
    "total_404": 5,
}


def csv_path(tmp_path, name="data.csv"):
    return str(tmp_path / name)


def seeded(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_write(path, HEADERS, [["1.1.1.1", "x"], ["2.2.2.2", "y"]], False)
    return path


def test_every_exported_function_and_class_is_named_in_a_binding_test():
    names = {name for name in dir(aiwaf_rust) if not name.startswith("_")}
    names = {name for name in names if callable(getattr(aiwaf_rust, name))}
    # Internal request value handed to the host by NativeCSVPipeline; not constructible.
    names -= {"NativePipelineRequest"}
    source = "\n".join(path.read_text(encoding="utf-8") for path in Path(__file__).parent.glob("test_*.py"))
    assert sorted(name for name in names if f"aiwaf_rust.{name}" not in source) == []


# CSV storage


def test_csv_write_read_and_append_round_trip(tmp_path):
    path = seeded(tmp_path)
    aiwaf_rust.csv_write(path, HEADERS, [["3.3.3.3", "z"]], True)
    assert aiwaf_rust.csv_read(path, False) == (
        HEADERS, [["1.1.1.1", "x"], ["2.2.2.2", "y"], ["3.3.3.3", "z"]],
    )


def test_csv_write_replaces_rows_when_not_appending(tmp_path):
    path = seeded(tmp_path)
    aiwaf_rust.csv_write(path, HEADERS, [["9.9.9.9", "n"]], False)
    assert aiwaf_rust.csv_read(path, False)[1] == [["9.9.9.9", "n"]]


def test_csv_read_keeps_non_ascii_values(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_write(path, HEADERS, [["1.1.1.1", "café, 日本"]], False)
    assert aiwaf_rust.csv_read(path, False)[1] == [["1.1.1.1", "café, 日本"]]


def test_csv_read_cache_sees_later_writes(tmp_path):
    path = seeded(tmp_path)
    assert len(aiwaf_rust.csv_read(path, True)[1]) == 2
    aiwaf_rust.csv_write(path, HEADERS, [["3.3.3.3", "z"]], True)
    assert len(aiwaf_rust.csv_read(path, True)[1]) == 3


@pytest.mark.parametrize("call", [
    lambda path: aiwaf_rust.csv_read(path, False),
    lambda path: aiwaf_rust.csv_read_set(path, "ip", False),
    lambda path: aiwaf_rust.csv_read_dict(path, "ip", "reason", False),
    lambda path: aiwaf_rust.csv_contains(path, "ip", "1.1.1.1", False),
    lambda path: aiwaf_rust.csv_headers_match(path, HEADERS),
])
def test_csv_readers_raise_oserror_for_a_missing_file(tmp_path, call):
    with pytest.raises(OSError):
        call(csv_path(tmp_path, "missing.csv"))


def test_csv_read_set_and_dict_return_python_collections(tmp_path):
    path = seeded(tmp_path)
    assert aiwaf_rust.csv_read_set(path, "ip", False) == {"1.1.1.1", "2.2.2.2"}
    assert aiwaf_rust.csv_read_dict(path, "ip", "reason", False) == {"1.1.1.1": "x", "2.2.2.2": "y"}


def test_csv_readers_return_empty_for_an_unknown_column(tmp_path):
    path = seeded(tmp_path)
    assert aiwaf_rust.csv_read_set(path, "nope", False) == set()
    assert aiwaf_rust.csv_read_dict(path, "nope", "reason", False) == {}


def test_csv_contains_and_lookup(tmp_path):
    path = seeded(tmp_path)
    assert aiwaf_rust.csv_contains(path, "ip", "1.1.1.1", False) is True
    assert aiwaf_rust.csv_contains(path, "ip", "9.9.9.9", False) is False
    assert aiwaf_rust.csv_lookup(path, "ip", "2.2.2.2") == {"ip": "2.2.2.2", "reason": "y"}
    assert aiwaf_rust.csv_lookup(path, "ip", "9.9.9.9") is None


def test_csv_headers_match(tmp_path):
    path = seeded(tmp_path)
    assert aiwaf_rust.csv_headers_match(path, HEADERS) is True
    assert aiwaf_rust.csv_headers_match(path, ["other"]) is False


def test_csv_remove_key_and_set_update(tmp_path):
    path = seeded(tmp_path)
    aiwaf_rust.csv_remove_key(path, HEADERS, "1.1.1.1")
    assert aiwaf_rust.csv_read(path, False)[1] == [["2.2.2.2", "y"]]
    other = csv_path(tmp_path, "set.csv")
    aiwaf_rust.csv_set_update(other, HEADERS, "k", ["k", "v"], False)
    assert aiwaf_rust.csv_read(other, False)[1] == [["k", "v"]]
    aiwaf_rust.csv_set_update(other, HEADERS, "k", ["k", "v"], True)
    assert aiwaf_rust.csv_read(other, False)[1] == []


def test_csv_policy_list_update_normalizes_keys(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_policy_list_update(path, HEADERS, "AbC", ["AbC", "r"], False, "lower")
    assert aiwaf_rust.csv_read(path, False)[1] == [["AbC", "r"]]
    aiwaf_rust.csv_policy_list_update(path, HEADERS, "abc", [], True, "lower")
    assert aiwaf_rust.csv_read(path, False)[1] == []


def test_csv_policy_list_update_rejects_an_unknown_mode(tmp_path):
    with pytest.raises(OSError, match="Invalid key normalization"):
        aiwaf_rust.csv_policy_list_update(csv_path(tmp_path), HEADERS, "a", ["a", "r"], False, "nope")


@pytest.mark.parametrize("append", [aiwaf_rust.csv_log_append, aiwaf_rust.csv_log_append_uncached])
def test_csv_log_append_writes_one_header_and_creates_folders(tmp_path, append):
    path = tmp_path / "nested" / "log.csv"
    append(str(path), HEADERS, ["1", "x"], False)
    append(str(path), HEADERS, ["2", "y"], True)
    aiwaf_rust.csv_log_close(str(path))
    assert path.read_text(encoding="utf-8").splitlines() == ["ip,reason", "1,x", "2,y"]


def test_csv_log_close_accepts_no_path():
    assert aiwaf_rust.csv_log_close() is None


def test_csv_blocklist_block_and_check_follow_expiry(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_blocklist_block(path, BLOCK_HEADERS, "1.1.1.1", "bad", "2026", "{}", 100.0, 50)
    assert aiwaf_rust.csv_blocklist_check(path, BLOCK_HEADERS, "1.1.1.1", 120.0) is True
    assert aiwaf_rust.csv_blocklist_check(path, BLOCK_HEADERS, "9.9.9.9", 120.0) is False
    assert aiwaf_rust.csv_blocklist_check(path, BLOCK_HEADERS, "1.1.1.1", 500.0) is False
    assert aiwaf_rust.csv_read(path, False)[1] == []


def test_csv_blocklist_block_accepts_none_and_builds_reputation(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_blocklist_block(path, BLOCK_HEADERS, "1.1.1.1", "sqli", "2026", None, 100.0, None)
    row = dict(zip(BLOCK_HEADERS, aiwaf_rust.csv_read(path, False)[1][0]))
    assert row["score"] == "40"
    assert row["offenses"] == "1"
    assert row["permanent"] == "False"
    assert json.loads(row["reasons"]) == ["sqli"]


def test_csv_blocklist_nonpositive_duration_is_permanent(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_blocklist_block(path, BLOCK_HEADERS, "1.1.1.1", "bad", "2026", None, 100.0, 0)
    row = dict(zip(BLOCK_HEADERS, aiwaf_rust.csv_read(path, False)[1][0]))
    assert row["permanent"] == "True"
    assert aiwaf_rust.csv_blocklist_check(path, BLOCK_HEADERS, "1.1.1.1", 10.0 ** 12) is True


def test_previous_blocklist_names_are_the_same_functions():
    assert aiwaf_rust.csv_blacklist_block is aiwaf_rust.csv_blocklist_block
    assert aiwaf_rust.csv_blacklist_check is aiwaf_rust.csv_blocklist_check


def test_csv_kv_store_respects_expiry(tmp_path):
    path = csv_path(tmp_path)
    assert aiwaf_rust.csv_kv_update(path, "a", "1", 150.0, 100.0) is True
    assert aiwaf_rust.csv_kv_get(path, "a", 120.0) == "1"
    assert aiwaf_rust.csv_kv_keys(path, 120.0) == ["a"]
    assert aiwaf_rust.csv_kv_get(path, "a", 200.0) is None
    assert aiwaf_rust.csv_kv_keys(path, 200.0) == []


def test_csv_kv_update_accepts_no_expiry(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_kv_update(path, "a", "value", None, 100.0)
    assert aiwaf_rust.csv_kv_get(path, "a", 10.0 ** 12) == "value"


def test_csv_rate_consume_persists_between_calls(tmp_path):
    path = csv_path(tmp_path)
    results = [aiwaf_rust.csv_rate_consume(path, "ip", 100.0 + i, 60.0, 1, 2) for i in range(3)]
    assert results == [
        ("allow", 1, [100.0]),
        ("throttle", 2, [100.0, 101.0]),
        ("flood_block", 3, [100.0, 101.0, 102.0]),
    ]


def test_csv_runtime_keyword_adds_and_removes(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_runtime_keyword(path, "evil", 3, False, 100.0)
    assert json.loads(aiwaf_rust.csv_kv_get(path, "keyword_counts", 100.0)) == {"evil": 3}
    aiwaf_rust.csv_runtime_keyword(path, "evil", 0, True, 100.0)
    assert json.loads(aiwaf_rust.csv_kv_get(path, "keyword_counts", 100.0)) == {}


def test_csv_runtime_keywords_update_removes_then_adds(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_runtime_keyword(path, "old", 5, False, 100.0)
    aiwaf_rust.csv_runtime_keywords_update(path, [("a", 1), ("a", 3), ("b", 2)], {"old"}, 100.0)
    assert json.loads(aiwaf_rust.csv_kv_get(path, "keyword_counts", 100.0)) == {"a": 4, "b": 2}


def test_csv_runtime_list_update_reports_whether_the_value_was_present(tmp_path):
    path = csv_path(tmp_path)
    assert aiwaf_rust.csv_runtime_list_update(path, "add_ip", "1.1.1.1", "why", 100.0) is False
    assert aiwaf_rust.csv_runtime_list_update(path, "add_ip", "1.1.1.1", "why", 100.0) is True
    assert json.loads(aiwaf_rust.csv_kv_get(path, "exemptions", 100.0))["ips"] == ["1.1.1.1"]
    assert aiwaf_rust.csv_runtime_list_update(path, "remove_ip", "1.1.1.1", "", 100.0) is True
    assert aiwaf_rust.csv_runtime_list_update(path, "remove_ip", "1.1.1.1", "", 100.0) is False


def test_csv_runtime_list_update_rejects_an_unknown_operation(tmp_path):
    with pytest.raises(OSError, match="Unknown policy-list operation"):
        aiwaf_rust.csv_runtime_list_update(csv_path(tmp_path), "nope", "1", "w", 100.0)


def test_csv_runtime_block_creates_a_block_entry(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_runtime_block(path, "1.1.1.1", "bad", "{}", 100.0, 60)
    assert "blocked:1.1.1.1" in aiwaf_rust.csv_kv_keys(path, 100.0)


def test_csv_request_exempt_checks_allowlist_then_path_rules(tmp_path):
    allowlist = csv_path(tmp_path)
    aiwaf_rust.csv_write(allowlist, ["ip"], [["1.1.1.1"]], False)
    assert aiwaf_rust.csv_request_exempt(allowlist, "1.1.1.1", "/a", []) is True
    assert aiwaf_rust.csv_request_exempt(allowlist, "2.2.2.2", "/health", [("/health", None)]) is True
    assert aiwaf_rust.csv_request_exempt(allowlist, "2.2.2.2", "/other", [("/health", None)]) is False
    assert aiwaf_rust.csv_request_exempt(allowlist, "2.2.2.2", "", [("/health", None)]) is False


def test_csv_geo_policy_reads_dynamic_countries(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_write(path, ["country"], [["ru"]], False)
    assert aiwaf_rust.csv_geo_policy(path, "ru", set(), set()) == (True, "RU", "Geo blocked: RU")
    assert aiwaf_rust.csv_geo_policy(path, "us", set(), set())[0] is False


def test_csv_runtime_geo_policy_uses_static_and_stored_lists(tmp_path):
    path = csv_path(tmp_path)
    assert aiwaf_rust.csv_runtime_geo_policy(path, "ru", set(), {"RU"}, set(), 100.0) == (
        True, "RU", "Geo blocked: RU",
    )
    aiwaf_rust.csv_runtime_list_update(path, "add_country", "CN", "", 100.0)
    assert aiwaf_rust.csv_runtime_geo_policy(path, "cn", set(), set(), set(), 100.0)[0] is True


def test_csv_keywords_update_removes_and_deduplicates(tmp_path):
    path = csv_path(tmp_path)
    aiwaf_rust.csv_keywords_update(path, "2026", ["a", "b", "a"], set())
    aiwaf_rust.csv_keywords_update(path, "2027", ["c"], {"a"})
    assert aiwaf_rust.csv_read(path, False)[1] == [["b", "2026"], ["c", "2027"]]


def test_csv_keyword_apply_returns_the_policy_decision(tmp_path):
    decision = aiwaf_rust.csv_keyword_apply(
        csv_path(tmp_path), "2026", "/hackme", "", set(), False, False,
        set(), set(), set(), set(), [], set(),
    )
    assert decision == ("Keyword block: Inherently suspicious: hackme", [], ["hackme"])


def test_csv_keyword_apply_from_store_returns_the_policy_decision(tmp_path):
    decision = aiwaf_rust.csv_keyword_apply_from_store(
        csv_path(tmp_path), "2026", "/docs", "", set(), True, False,
        set(), set(), set(), [], set(), 10,
    )
    assert decision == (None, [], ["docs"])


# Training


def test_training_parse_lines_returns_candidates_per_line():
    line = '1.1.1.1 - - [10/Oct/2026:12:00:00 +0000] "GET /a HTTP/1.1" 200 12 "-" "ua" response-time=0.5'
    parsed = aiwaf_rust.training_parse_lines([line, "junk"])
    assert parsed[0][0] == ("1.1.1.1", "10/Oct/2026:12:00:00 +0000", "/a", "200", 0.5)
    assert parsed[1] == []


def test_training_parse_csv_converts_milliseconds():
    text = "ip,timestamp,path,status_code,response_time_ms\n1.1.1.1,t,/a,200,125\n"
    assert aiwaf_rust.training_parse_csv(text) == [("1.1.1.1", "t", "/a", "200", 0.125)]


def test_training_parse_csv_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing training CSV columns"):
        aiwaf_rust.training_parse_csv("a,b\n1,2\n")


def test_training_csv_reader_batches_and_skips_bad_rows(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text(
        "ip,timestamp,path,status_code,response_time_ms\n"
        "1.1.1.1,t,/a,200,125\n2.2.2.2,t,/b,404,x\n3.3.3.3,t,/c,200,5\n",
        encoding="utf-8",
    )
    reader = aiwaf_rust.TrainingCSVReader(str(path))
    assert reader.next_batch(2) == (2, [("1.1.1.1", "t", "/a", "200", 0.125)])
    assert reader.next_batch(2) == (1, [("3.3.3.3", "t", "/c", "200", 0.005)])
    assert reader.next_batch(2) == (0, [])


def test_training_csv_reader_reads_gzip(tmp_path):
    path = tmp_path / "train.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write("ip,timestamp,path,status_code,response_time\n1.1.1.1,t,/a,200,0.5\n")
    assert aiwaf_rust.TrainingCSVReader(str(path)).next_batch(10) == (1, [("1.1.1.1", "t", "/a", "200", 0.5)])


@pytest.mark.parametrize("size", [0, 65537])
def test_training_csv_reader_rejects_invalid_batch_sizes(tmp_path, size):
    path = tmp_path / "train.csv"
    path.write_text("ip,timestamp,path,status_code\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Batch size"):
        aiwaf_rust.TrainingCSVReader(str(path)).next_batch(size)


def test_training_csv_reader_reports_bad_files(tmp_path):
    with pytest.raises(OSError):
        aiwaf_rust.TrainingCSVReader(str(tmp_path / "missing.csv"))
    path = tmp_path / "other.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing training CSV column"):
        aiwaf_rust.TrainingCSVReader(str(path))


def test_training_aggregate_splits_exempt_errors():
    errors, exempt, indices = aiwaf_rust.training_aggregate(
        [("a", "/x", "404"), ("a", "/y", "404"), ("b", "/x", "200")], {"/y"},
    )
    assert errors == {"a": 1}
    assert exempt == {"a": 1}
    assert indices == {"a": [0, 1], "b": [2]}


def test_training_keywords_ranks_tokens_from_malicious_paths():
    records = [("/shell/evil.php", "404"), ("/shell/evil.php", "404"), ("/docs", "200")]
    result = aiwaf_rust.training_keywords(records, {"/docs"}, set(), set(), set(), 5)
    assert [(token, count) for token, count, _ in result] == [("shell", 2), ("evil", 2)]


def test_training_keywords_defers_non_ascii_paths_to_python():
    assert aiwaf_rust.training_keywords([("/über", "404")], set(), set(), set(), set(), 5) is None


def test_training_keyword_counts_returns_counts_with_examples():
    records = [("/shell/evil.php", "404"), ("/shell/evil.php", "404"), ("/docs", "200")]
    result = aiwaf_rust.training_keyword_counts(records, {"/docs"}, set(), set(), set())
    counts = {token: count for token, count, _ in result}
    assert counts["shell"] == 2
    assert all(isinstance(examples, list) for _, _, examples in result)


def test_training_keyword_counts_defers_non_ascii_paths_to_python():
    assert aiwaf_rust.training_keyword_counts([("/über", "404")], set(), set(), set(), set()) is None


def test_training_anomaly_stats_averages_flagged_rows_only():
    stats = aiwaf_rust.training_anomaly_stats(
        [("a", 1.0, 2.0, 3.0), ("a", 3.0, 4.0, 5.0), ("b", 0.0, 0.0, 1.0)], [-1, -1, 1],
    )
    assert stats == {"a": (2.0, 4.0, 4.0, 2)}


def test_training_date_parts_returns_tuple_or_none():
    assert aiwaf_rust.training_date_parts(["2026-10-12T01:02:03Z", "bad"], False) == [
        (2026, 10, 12, 1, 2, 3, 0, 0), None,
    ]


def test_extract_training_features_returns_dicts():
    features = aiwaf_rust.extract_training_features([FEATURE_RECORD], ["wp"])
    assert len(features) == 1
    assert features[0]["ip"] == "1.2.3.4"


def test_extract_training_features_reports_missing_fields():
    with pytest.raises((KeyError, TypeError, ValueError)):
        aiwaf_rust.extract_training_features([{"ip": "1.2.3.4"}], ["wp"])


def test_finalize_feature_state_returns_an_empty_batch():
    result = aiwaf_rust.finalize_feature_state(["wp"])
    assert result["features"] == []


def test_extract_raw_and_native_features_read_raw_records():
    record = {
        "ip": "1.1.1.1",
        "path": "/wp-admin",
        "timestamp": datetime(2026, 10, 12, tzinfo=timezone.utc),
        "response_time": 0.03,
        "status": 404,
    }
    args = ([record], {"/wp-admin": True}, {"1.1.1.1": 1}, ["200", "404"], ["wp"])
    raw = aiwaf_rust.extract_raw_features(*args)
    assert [row["ip"] for row in raw] == ["1.1.1.1"]
    batch = aiwaf_rust.extract_native_training_features(*args)
    assert isinstance(batch, aiwaf_rust.NativeTrainingBatch)
    assert len(batch) == 1


def test_extract_raw_features_reports_missing_fields():
    with pytest.raises(KeyError):
        aiwaf_rust.extract_raw_features([{"ip": "1.1.1.1"}], {}, {}, [], [])


def test_native_training_builder_collects_batches():
    builder = aiwaf_rust.NativeTrainingBuilder()
    builder.add_batch([FEATURE_RECORD], ["wp"])
    builder.add_batch([dict(FEATURE_RECORD, timestamp=11.0)], ["wp"])
    batch = builder.finish()
    assert isinstance(batch, aiwaf_rust.NativeTrainingBatch)
    assert len(batch) == 2
    assert len(batch.to_dicts()) == 2
    assert "ip" not in batch.feature_columns()


def test_native_training_builder_starts_empty():
    assert len(aiwaf_rust.NativeTrainingBuilder().finish()) == 0


# Evidence receipts


def test_training_evidence_prevents_replay_after_restart(tmp_path):
    directory = str(tmp_path / "evidence")
    first = aiwaf_rust.TrainingEvidence(directory)
    assert first.had_history is False
    assert first.fresh_indices(EMPTY_EVIDENCE) == [0, 1]
    assert first.can_block("1.1.1.1") is True
    assert first.can_block("9.9.9.9") is False
    first.commit("1.1.1.1")
    assert first.can_block("1.1.1.1") is False
    first.close()

    second = aiwaf_rust.TrainingEvidence(directory)
    assert second.had_history is True
    assert second.fresh_indices(EMPTY_EVIDENCE) == []
    assert second.can_block("1.1.1.1") is False
    assert second.fresh_indices(EMPTY_EVIDENCE + [("1.1.1.1", "t3", "/z", "404")]) == [2]
    assert second.can_block("1.1.1.1") is True
    second.close()


def test_training_evidence_refuses_commit_without_fresh_evidence(tmp_path):
    evidence = aiwaf_rust.TrainingEvidence(str(tmp_path))
    try:
        with pytest.raises(OSError, match="No fresh training evidence"):
            evidence.commit("1.1.1.1")
    finally:
        evidence.close()


def test_training_evidence_refuses_use_after_close(tmp_path):
    evidence = aiwaf_rust.TrainingEvidence(str(tmp_path))
    evidence.fresh_indices(EMPTY_EVIDENCE)
    evidence.close()
    assert evidence.can_block("1.1.1.1") is False
    with pytest.raises(OSError, match="closed"):
        evidence.fresh_indices(EMPTY_EVIDENCE)


def test_training_evidence_rejects_unknown_state_versions(tmp_path):
    (tmp_path / "training_evidence.json").write_text('{"version":2,"ips":{}}', encoding="utf-8")
    with pytest.raises(OSError, match="Invalid AIWAF training evidence state"):
        aiwaf_rust.TrainingEvidence(str(tmp_path))


# Anomaly history


def test_native_anomaly_history_behaves_like_a_python_sequence():
    history = aiwaf_rust.NativeAnomalyHistory([(100.0, "/a", 200, 0.1), (50.0, "/b", 404, 0.2)])
    assert len(history) == 2
    assert history[0] == (100.0, "/a", 200, 0.1)
    assert history[-1] == (50.0, "/b", 404, 0.2)
    for index in (2, -3):
        with pytest.raises(IndexError):
            history[index]


def test_native_anomaly_history_appended_does_not_change_the_original():
    history = aiwaf_rust.NativeAnomalyHistory([(100.0, "/a", 200, 0.1)])
    longer = history.appended(101.0, "/b", 404, 0.2)
    assert (len(history), len(longer)) == (1, 2)
    assert longer[-1] == (101.0, "/b", 404, 0.2)


def test_native_anomaly_history_filters_by_time():
    history = aiwaf_rust.NativeAnomalyHistory([(100.0, "/a", 200, 0.1), (50.0, "/b", 404, 0.2)])
    assert len(history.trimmed(100.0, 10.0)) == 1
    assert history.recent_paths(101.0) == ["/a", "/b"]
    assert history.recent_paths(1000.0) == []


def test_native_anomaly_history_reports_ascii_support():
    assert aiwaf_rust.NativeAnomalyHistory([(1.0, "/a", 200, 0.1)]).supports_ascii() is True
    assert aiwaf_rust.NativeAnomalyHistory([(1.0, "/ü", 200, 0.1)]).supports_ascii() is False


def test_native_anomaly_history_analyze_returns_stats():
    history = aiwaf_rust.NativeAnomalyHistory([(100.0, "/a", 404, 0.1), (100.5, "/b", 200, 0.2)])
    stats = history.analyze(101.0, ["a"], [True, False])
    assert stats == aiwaf_rust.analyze_recent_tuples(
        [("/a", 100.0, 404, True), ("/b", 100.5, 200, False)], ["a"],
    )
    assert stats[3] == 2


def test_native_anomaly_history_analyze_checks_the_eligibility_count():
    history = aiwaf_rust.NativeAnomalyHistory([(100.0, "/a", 200, 0.1)])
    with pytest.raises(ValueError, match="Eligibility count mismatch"):
        history.analyze(100.0, [], [])


def test_native_anomaly_history_predict_current_returns_features_on_request():
    forest = aiwaf_rust.IsolationForest(n_estimators=10, random_state=42)
    forest.fit([[float(i), 1.0, 0.1, 0.0, 1.0, 0.0] for i in range(20)])
    history = aiwaf_rust.NativeAnomalyHistory([(99.0, "/a", 404, 0.1)])
    prediction, features = history.predict_current(
        forest, "/shell", 404, 0.5, 100.0, True, ["shell"], ["200", "404"], True,
    )
    assert prediction in (-1, 1)
    assert features == [6.0, 1.0, 0.5, 1.0, 1.0, 1.0]
    assert history.predict_current(
        forest, "/shell", 404, 0.5, 100.0, True, ["shell"], ["200", "404"], False,
    )[1] is None


# Isolation forest and matchers


def test_isolation_forest_is_anomaly_and_decision_function_agree():
    forest = aiwaf_rust.IsolationForest(n_estimators=50, random_state=42)
    forest.fit([[(i % 8) * 0.01, (i % 4) * 0.01] for i in range(64)] + [[50.0, 50.0]])
    score = forest.anomaly_score([0.01, 0.01])
    assert forest.is_anomaly([0.01, 0.01], score - 0.01) is True
    assert forest.is_anomaly([0.01, 0.01], 1.0) is False
    decisions = forest.decision_function([[0.01, 0.01], [50.0, 50.0]])
    assert forest.predict([[0.01, 0.01], [50.0, 50.0]]) == [-1 if d < 0 else 1 for d in decisions]


def test_route_matcher_returns_the_matching_prefix():
    assert aiwaf_rust.RouteMatcher(["/api", "/api/v2/"]).match_index("/api/v2/x") == 1
    assert aiwaf_rust.RouteMatcher(["/api"]).match_index("/api") == 0
    assert aiwaf_rust.RouteMatcher(["/api"]).match_index("/other") is None


# Request engine


def request_engine(tmp_path, sql=True):
    return aiwaf_rust.NativeRequestEngine(
        sql, 1024, False, 10, csv_path(tmp_path), False, set(), set(), set(), [], set(),
    )


def test_native_request_engine_detects_sql_in_the_query(tmp_path):
    engine = request_engine(tmp_path)
    decision = engine.evaluate(b"q=1+union+select+2", None, "/a", "", set(), True, "2026", 100.0)
    assert "sql_breakout_numeric" in repr(decision)
    assert engine.processed_requests >= 1


def test_native_request_engine_defers_unsupported_input_to_python(tmp_path):
    engine = request_engine(tmp_path)
    assert engine.evaluate(b"", None, "/ü", "", set(), True, "2026", 100.0) is None
    assert engine.evaluate(b"", None, "/a", "", set(), True, "2026", float("nan")) is None
    assert engine.evaluate(b"q=1", b"body", "/a", "", set(), True, "2026", 100.0) is None


# Header validation


REQUIRED_HEADER_REQUEST = {
    "HTTP_USER_AGENT": "Mozilla/5.0",
    "HTTP_ACCEPT": "text/html",
    "HTTP_ACCEPT_LANGUAGE": "en",
    "HTTP_ACCEPT_ENCODING": "gzip",
    "HTTP_CONNECTION": "keep-alive",
}


def test_validate_headers_full_default_enforces_custom_required_headers():
    reason = aiwaf_rust.validate_headers_full_default(REQUIRED_HEADER_REQUEST, ["HTTP_X_API_KEY"], 0)
    assert reason == "Missing required headers: x-api-key"


@pytest.mark.parametrize("name", ["HTTP_X_API_KEY", "x-api-key", "X-Api-Key"])
def test_validate_headers_with_config_enforces_custom_required_headers(name):
    reason = aiwaf_rust.validate_headers_with_config(REQUIRED_HEADER_REQUEST, [name], 0)
    assert reason == "Missing required headers: x-api-key"


def test_validate_headers_with_config_accepts_a_present_custom_header():
    headers = dict(REQUIRED_HEADER_REQUEST, HTTP_X_API_KEY="secret")
    assert aiwaf_rust.validate_headers_with_config(headers, ["x-api-key"], 0) is None


# Stateless policies


def test_policy_path_exempt_handles_prefixes_and_bad_patterns():
    assert aiwaf_rust.policy_path_exempt("/health", [("/health", None)], False) is True
    assert aiwaf_rust.policy_path_exempt("/health/x", [("/health/", None)], True) is True
    assert aiwaf_rust.policy_path_exempt("/health/x", [("/health/", None)], False) is False
    assert aiwaf_rust.policy_path_exempt("/x", [("/x", "(")], False) is None


def test_policy_ip_exempt_converts_sets_and_tuples():
    assert aiwaf_rust.policy_ip_exempt("10.0.0.1", set(), [], [("10.0.0.0", 8)]) is True
    assert aiwaf_rust.policy_ip_exempt("10.0.0.1", {"10.0.0.1"}, [], []) is True
    assert aiwaf_rust.policy_ip_exempt("x", set(), [], [("bad", 8)]) is None


def test_policy_path_rule_index_prefers_the_longest_prefix():
    assert aiwaf_rust.policy_path_rule_index("/api/v2/x", ["/api/", "/api/v2/"]) == 1
    assert aiwaf_rust.policy_path_rule_index("/api", ["/api/"]) == 0
    assert aiwaf_rust.policy_path_rule_index("/z", ["/api/"]) is None


def test_policy_route_plan_applies_exemptions_and_required_checks():
    rules = [("/api/", {"rate"})]
    assert aiwaf_rust.policy_route_plan("/api/x", rules, ["rate", "geo", "hdr"], False, {"geo"}, set()) == (0, {"hdr"})
    assert aiwaf_rust.policy_route_plan("/z", rules, ["rate", "geo"], True, set(), {"geo"}) == (None, {"geo"})


def test_policy_rate_returns_python_values():
    assert aiwaf_rust.policy_rate([99.0], 100.0, 60.0, 5, 10) == ("allow", 2, [99.0, 100.0])


def test_policy_geo_returns_python_values():
    assert aiwaf_rust.policy_geo("ru", set(), {"RU"}, set()) == (True, "RU", "Geo blocked: RU")


def test_policy_form_accepts_none_for_no_page_view():
    assert aiwaf_rust.policy_form(100.0, None, "/f", 1.0, 300.0) == ("allow", None, None, None)
    assert aiwaf_rust.policy_form(500.0, 100.0, "/f", 1.0, 300.0)[2] == 409


def test_policy_uuid_returns_a_bool():
    assert aiwaf_rust.policy_uuid("123e4567-e89b-12d3-a456-426614174000") is True
    assert aiwaf_rust.policy_uuid("nope") is False


def test_policy_reputation_returns_python_values():
    assert aiwaf_rust.policy_reputation(50, 0, ["old"], "other") == (60, 1, ["old", "other"], True, 900)


def test_policy_keyword_returns_python_values():
    assert aiwaf_rust.policy_keyword(
        "/hackme", set(), False, False, set(), set(), set(), set(), [], set(), set(),
    ) == ("Keyword block: Inherently suspicious: hackme", [], ["hackme"])


def test_policy_keyword_flask_derives_context_from_the_request():
    args = (set(), set(), set(), [], set())
    assert aiwaf_rust.policy_keyword_flask("/shell", "", set(), True, False, {"shell"}, *args)[0] is None
    decision = aiwaf_rust.policy_keyword_flask("/x/../shell", "q=1", {"cmd"}, True, False, {"shell"}, *args)
    assert decision[0] == "Keyword block: Learned keyword: shell"


def test_policy_sql_signature_functions_now_use_the_tokenizer():
    # Kept so older aiwaf releases keep working with this aiwaf-rust.
    assert aiwaf_rust.policy_sql_signature("1 union select 2") == "sql_breakout_numeric"
    assert aiwaf_rust.policy_sql_signature("hello") is None
    assert aiwaf_rust.policy_sql_signatures(["ok", "x'--", "1 union select 2"]) == "sql_breakout_single_quote"
    assert aiwaf_rust.policy_sql_signatures([]) is None


@pytest.mark.parametrize("call", [
    lambda: aiwaf_rust.policy_rate("nope", 1.0, 1.0, 1, 1),
    lambda: aiwaf_rust.policy_uuid(None),
    lambda: aiwaf_rust.policy_sql_signature(b"bytes"),
    lambda: aiwaf_rust.policy_geo(1, set(), set(), set()),
    lambda: aiwaf_rust.policy_path_exempt("/a", "rules", True),
    lambda: aiwaf_rust.validate_headers(["not", "a", "dict"]),
    lambda: aiwaf_rust.training_parse_lines(None),
    lambda: aiwaf_rust.training_aggregate([("too", "short")], set()),
    lambda: aiwaf_rust.csv_read(None, False),
    lambda: aiwaf_rust.RouteMatcher(None),
    lambda: aiwaf_rust.KeywordMatcher(None),
    lambda: aiwaf_rust.NativeAnomalyHistory([("not-a-time", "/a", 200, 0.1)]),
])
def test_wrong_argument_types_raise_python_errors(call):
    with pytest.raises((TypeError, ValueError)):
        call()


def test_exported_callables_have_inspectable_signatures():
    for name in ("policy_rate", "csv_read", "training_parse_lines", "validate_headers"):
        assert inspect.signature(getattr(aiwaf_rust, name)).parameters


SQL_CASES = json.loads((Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sql_injection_cases.json").read_text(encoding="utf-8"))


def test_policy_sql_tokenizer_functions():
    assert aiwaf_rust.policy_sql_tokenizer("1 OR 1=1") == "sql_breakout_numeric"
    assert aiwaf_rust.policy_sql_tokenizer("o'reilly") is None
    assert aiwaf_rust.policy_sql_tokenizers(["ok", "x' OR 1=1--", "1 OR 1=1"]) == "sql_breakout_single_quote"
    assert aiwaf_rust.policy_sql_tokenizers([]) is None


def test_policy_sql_tokenizer_matches_the_shared_cases():
    assert [value for value in SQL_CASES["attacks"] if not aiwaf_rust.policy_sql_tokenizer(value)] == []
    assert [value for value in SQL_CASES["ordinary"] if aiwaf_rust.policy_sql_tokenizer(value)] == []


def test_policy_sql_inspect_checks_whole_requests():
    assert aiwaf_rust.policy_sql_inspect(b"id=1%20OR%201%3D1", b"", 65536) == ("sql_breakout_numeric", 403)
    assert aiwaf_rust.policy_sql_inspect(b"", b'{"a":{"b":"x\\u0027 OR 1=1--"}}', 65536) == ("sql_breakout_single_quote", 403)
    assert aiwaf_rust.policy_sql_inspect(b"q=hello", b'{"name":"O\'Reilly"}', 65536) is None
    assert aiwaf_rust.policy_sql_inspect(b"", b"a" * 20, 10) == ("payload_inspection_limit", 413)
    for value in SQL_CASES["attacks"]:
        assert aiwaf_rust.policy_sql_inspect(b"", json.dumps({"f": value}).encode(), 65536), value


def test_route_risk_checker_and_rules():
    manifest = {"routes": {"/files/<path:name>": {"injection_risks": [{"sink": "path", "input": "name", "confidence": "medium"}]}}}
    checker = aiwaf_rust.RouteRiskChecker(json.dumps(manifest))
    assert not checker.empty
    assert checker.inspect("/files/../secret", "", b"") == [("/files/<path:name>", "path", "name", "path_traversal", "medium")]
    assert checker.inspect("/files/ok", "other=../x", b"") == []
    assert aiwaf_rust.RouteRiskChecker("not json").empty
    assert aiwaf_rust.policy_route_rule("command", "x; ls -la") == "command_chain"
    assert aiwaf_rust.policy_route_rule("command", "Fish & chips; cat food") is None
    assert aiwaf_rust.policy_route_rule("sql", "1 OR 1=1") == "sql_breakout_numeric"
