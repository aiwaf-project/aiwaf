"""
Storage and blacklist tests for AIWAF.
"""
import time

from aiwaf.core.runtime_blacklist import BlacklistManager
from aiwaf.core.runtime_storage import FileStorage, get_blacklist_store, get_exemption_store, get_storage


def test_memory_storage_ttl_expiration():
    backend = get_storage()
    backend.set("temp", "value", ttl=1)
    assert backend.get("temp") == "value"
    time.sleep(1.1)
    assert backend.get("temp") is None


def test_file_storage_persistence(tmp_path):
    file_path = tmp_path / "aiwaf_store.json"
    backend = FileStorage(str(file_path))
    backend.set("persist", {"ok": True})

    reloaded = FileStorage(str(file_path))
    assert reloaded.get("persist") == {"ok": True}


def test_exemption_store_manages_ips_and_patterns():
    store = get_exemption_store()
    store.add_ip("203.0.113.4", "testing exemption")
    assert store.is_exempted("203.0.113.4")

    store.add_pattern("198.51.100.*", "CIDR range")
    assert store.is_exempted("198.51.100.5")

    assert store.remove_ip("203.0.113.4")
    assert not store.is_exempted("203.0.113.4")

    assert store.remove_pattern("198.51.100.*")


def test_blacklist_store_block_unblock_flow():
    store = get_blacklist_store()
    store.block_ip("198.51.100.10", "test block", duration=60)

    assert store.is_blocked("198.51.100.10")
    info = store.get_block_info("198.51.100.10")
    assert info is not None and info["reason"] == "test block"

    blocked_ips = store.get_blocked_ips()
    assert "198.51.100.10" in blocked_ips

    stats = store.get_block_stats()
    assert stats["total_blocked"] == 1
    assert stats["reason_counts"].get("test block") == 1

    assert store.unblock_ip("198.51.100.10")
    assert not store.is_blocked("198.51.100.10")


def test_blacklist_manager_bulk_operations_and_recent_activity():
    block_ips = ["203.0.113.1", "203.0.113.2"]
    result = BlacklistManager.bulk_block(block_ips, "bulk test", duration=60)
    assert all(result.values())

    assert BlacklistManager.is_blocked("203.0.113.1")
    assert "203.0.113.2" in BlacklistManager.get_blocked_ips()

    recent = BlacklistManager.get_recent_blocks(hours=1)
    assert any(entry["ip"] == "203.0.113.1" for entry in recent)

    top_reasons = BlacklistManager.get_top_blocked_reasons(limit=2)
    assert top_reasons and top_reasons[0]["reason"].startswith("bulk test")

    unblock_result = BlacklistManager.bulk_unblock(["203.0.113.1"])
    assert unblock_result["203.0.113.1"]


def test_blacklist_manager_temporary_and_permanent_blocks():
    assert BlacklistManager.block_temporary("198.51.100.20", "temp reason", minutes=1)
    assert BlacklistManager.block_permanent("198.51.100.21", "perm reason")

    assert BlacklistManager.is_blocked("198.51.100.20")
    assert BlacklistManager.is_blocked("198.51.100.21")

    BlacklistManager.bulk_unblock(["198.51.100.20", "198.51.100.21"])
