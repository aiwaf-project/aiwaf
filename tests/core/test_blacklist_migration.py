from aiwaf.core.blacklist_migration import migrate_runtime_storage
from aiwaf.core.runtime_storage import BlacklistStore, MemoryStorage


def test_runtime_migration_backfills_legacy_blocks_as_permanent():
    storage = MemoryStorage()
    storage.set("blocked:203.0.113.1", {"ip": "203.0.113.1", "reason": "old scanner"})
    storage.set(
        "blocked:203.0.113.2",
        {"reason": "current", "reputation_reason": "current", "score": 20},
    )

    total, changed = migrate_runtime_storage(storage)
    migrated = storage.get("blocked:203.0.113.1")

    assert (total, changed) == (2, 1)
    assert migrated["reason"] == "old scanner"
    assert migrated["reputation_reason"] == "legacy_blacklist"
    assert migrated["score"] == 100
    assert migrated["offenses"] == 1
    assert migrated["permanent"] is True
    assert migrated["expires_at"] is None


def test_runtime_migration_is_idempotent():
    storage = MemoryStorage()
    storage.set("blocked:203.0.113.3", "old block")

    assert migrate_runtime_storage(storage) == (1, 1)
    assert migrate_runtime_storage(storage) == (1, 0)


def test_unmigrated_runtime_reason_remains_blocked_and_can_be_updated():
    storage = MemoryStorage()
    storage.set("blocked:203.0.113.4", "legacy reason")
    blacklist = BlacklistStore(storage)

    assert blacklist.is_blocked("203.0.113.4")
    assert blacklist.get_block_info("203.0.113.4")["reason"] == "legacy reason"

    blacklist.block_ip("203.0.113.4", "scanner")
    updated = blacklist.get_block_info("203.0.113.4")
    assert updated["offenses"] == 2
    assert "legacy reason" in updated["reasons"]
