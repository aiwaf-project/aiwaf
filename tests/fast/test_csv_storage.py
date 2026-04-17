from aiwaf.fast.storage import get_blacklist_store, get_exemption_store, initialize_storage


def test_fast_storage_whitelist_and_blacklist_operations():
    initialize_storage(backend="memory")
    exemptions = get_exemption_store()
    blacklist = get_blacklist_store()

    exemptions.add_ip("192.0.2.10", "trusted")
    assert exemptions.is_exempted("192.0.2.10")

    blacklist.block_ip("198.51.100.20", "test")
    assert blacklist.is_blocked("198.51.100.20")
