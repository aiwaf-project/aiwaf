from aiwaf.core import storage_csv_impl


def test_whitelist_and_blacklist_mutations_round_trip(tmp_path):
    storage_csv_impl.ensure_all(tmp_path)
    storage_csv_impl.rewrite_whitelist(tmp_path, {"203.0.113.1"})
    assert storage_csv_impl.read_whitelist(tmp_path) == {"203.0.113.1"}
    storage_csv_impl.append_blacklist(
        tmp_path,
        "203.0.113.2",
        "test",
        '{"path":"/admin"}',
        {"reputation_reason": "legacy_blacklist", "permanent": True},
    )
    rows = storage_csv_impl.legacy_blacklist_entries(tmp_path)
    assert "203.0.113.2" in rows
