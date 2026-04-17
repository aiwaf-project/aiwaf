from aiwaf.fast.storage import get_exemption_store, initialize_storage


def test_path_exemptions_store_roundtrip():
    initialize_storage(backend="memory")
    store = get_exemption_store()
    store.add_pattern("127.0.0.*", "trusted-network")
    assert "127.0.0.*" in store.get_exempted_patterns()
