from aiwaf.core.reputation import (
    FIRST_BLOCK_SECONDS,
    REPEATED_BLOCK_SECONDS,
    SECOND_BLOCK_SECONDS,
    evaluate_reputation,
    reason_weight,
)
from aiwaf.core.runtime_storage import initialize_storage, get_blacklist_store
from aiwaf.core import storage_csv_impl
from aiwaf.core.storage_schema import BLACKLIST_CSV, CSV_HEADERS


def test_reputation_scores_known_reasons():
    assert reason_weight("SQLi attempt") == 40
    assert reason_weight("XSS payload") == 30
    assert reason_weight("Scanner probe") == 20
    assert reason_weight("unknown custom reason") == 10


def test_reputation_progressive_duration():
    first = evaluate_reputation(existing=None, reason="scanner", now=1000)
    assert first.offenses == 1
    assert first.score == 20
    assert first.duration is None

    second = evaluate_reputation(existing=first.__dict__, reason="SQLi", now=1000)
    assert second.offenses == 2
    assert second.score == 60
    assert second.should_block is True
    assert second.duration == SECOND_BLOCK_SECONDS

    repeated = evaluate_reputation(existing=second.__dict__, reason="XSS", now=1000)
    assert repeated.offenses == 3
    assert repeated.duration == REPEATED_BLOCK_SECONDS


def test_runtime_blacklist_stores_temporary_reputation_metadata():
    initialize_storage("memory")
    store = get_blacklist_store()

    store.block_ip("203.0.113.10", "scanner")
    info = store.get_block_info("203.0.113.10")

    assert info["ip"] == "203.0.113.10"
    assert info["reason"] == "scanner"
    assert info["reputation_reason"]
    assert info["score"] == 20
    assert info["offenses"] == 1
    assert info["duration"] == FIRST_BLOCK_SECONDS
    assert info["expires_at"] > info["blocked_at"]
    assert info["permanent"] is False
    assert "scanner" in [reason.lower() for reason in info["reasons"]]


def test_csv_blacklist_round_trips_reputation_metadata(tmp_path):
    metadata = {
        "reason": "Score 60; offenses 2; reasons: scanner, SQLi; expires in 60 minutes",
        "reasons": ["scanner", "SQLi"],
        "score": 60,
        "offenses": 2,
        "expires_at": 2000.0,
        "duration": 3600,
        "extended_request_info": {"path": "/login"},
    }

    storage_csv_impl.rewrite_blacklist(tmp_path, {"203.0.113.50": metadata})
    blacklist = storage_csv_impl.read_blacklist(tmp_path)

    assert blacklist["203.0.113.50"]["score"] == 60
    assert blacklist["203.0.113.50"]["offenses"] == 2
    assert blacklist["203.0.113.50"]["expires_at"] == 2000.0
    assert blacklist["203.0.113.50"]["reasons"] == ["scanner", "SQLi"]
    assert blacklist["203.0.113.50"]["extended_request_info"] == {"path": "/login"}


def test_csv_blacklist_auto_upgrades_previous_schema(tmp_path):
    blacklist_file = tmp_path / "blacklist.csv"
    blacklist_file.write_text(
        "ip,reason,added_date,extended_request_info\n"
        "203.0.113.70,legacy block,2026-01-01T00:00:00,{\"path\":\"/old\"}\n",
        encoding="utf-8",
    )

    blacklist = storage_csv_impl.read_blacklist(tmp_path)
    headers = blacklist_file.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert headers == CSV_HEADERS[BLACKLIST_CSV]
    assert blacklist["203.0.113.70"]["reason"] == "legacy_import"
    assert blacklist["203.0.113.70"]["reputation_reason"] == "legacy_blacklist"
    assert blacklist["203.0.113.70"]["score"] == 100
    assert blacklist["203.0.113.70"]["permanent"] is True
    assert blacklist["203.0.113.70"]["expires_at"] is None
    assert blacklist["203.0.113.70"]["added_date"] == "2026-01-01T00:00:00"


def test_csv_blacklist_auto_upgrades_old_cli_schema(tmp_path):
    blacklist_file = tmp_path / "blacklist.csv"
    blacklist_file.write_text(
        "ip,timestamp,reason\n"
        "203.0.113.71,2026-01-02T00:00:00,old cli block\n",
        encoding="utf-8",
    )

    blacklist = storage_csv_impl.read_blacklist(tmp_path)
    headers = blacklist_file.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert headers == CSV_HEADERS[BLACKLIST_CSV]
    assert blacklist["203.0.113.71"]["reason"] == "legacy_import"
    assert blacklist["203.0.113.71"]["reputation_reason"] == "legacy_blacklist"
    assert blacklist["203.0.113.71"]["score"] == 100
    assert blacklist["203.0.113.71"]["permanent"] is True
    assert blacklist["203.0.113.71"]["expires_at"] is None
    assert blacklist["203.0.113.71"]["added_date"] == "2026-01-02T00:00:00"


def test_csv_blacklist_auto_upgrades_headerless_legacy_rows(tmp_path):
    blacklist_file = tmp_path / "blacklist.csv"
    blacklist_file.write_text(
        "192.168.1.5\n"
        "192.168.1.6,bad_bot\n",
        encoding="utf-8",
    )

    blacklist = storage_csv_impl.read_blacklist(tmp_path)
    headers = blacklist_file.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert headers == CSV_HEADERS[BLACKLIST_CSV]
    assert blacklist["192.168.1.5"]["reason"] == "legacy_import"
    assert blacklist["192.168.1.5"]["reputation_reason"] == "legacy_blacklist"
    assert blacklist["192.168.1.5"]["permanent"] is True
    assert blacklist["192.168.1.5"]["expires_at"] is None
    assert blacklist["192.168.1.6"]["reasons"] == ["legacy_blacklist", "bad_bot"]


def test_csv_blacklist_converts_legacy_entries_to_temporary(tmp_path):
    blacklist_file = tmp_path / "blacklist.csv"
    blacklist_file.write_text("192.168.1.5,bad_bot\n", encoding="utf-8")

    storage_csv_impl.read_blacklist(tmp_path)
    changed = storage_csv_impl.convert_legacy_blacklist_entries(tmp_path, duration=24 * 60 * 60)
    blacklist = storage_csv_impl.read_blacklist(tmp_path)

    assert changed == 1
    assert blacklist["192.168.1.5"]["permanent"] is False
    assert blacklist["192.168.1.5"]["duration"] == 24 * 60 * 60
    assert blacklist["192.168.1.5"]["expires_at"] > blacklist["192.168.1.5"]["blocked_at"]
