from aiwaf.core.runtime_blacklist import BlacklistManager
from aiwaf.core.runtime_storage import initialize_storage


def test_block_info_contains_basic_fields():
    initialize_storage(backend="memory")
    BlacklistManager.block("203.0.113.77", "unit-test reason", duration=60)

    info = BlacklistManager.get_block_info("203.0.113.77")
    assert info is not None
    assert info["ip"] == "203.0.113.77"
    assert info["reason"] == "unit-test reason"
