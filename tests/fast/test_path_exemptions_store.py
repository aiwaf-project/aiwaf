from aiwaf.fast.storage import get_exemption_store, initialize_storage


def test_remove_path_exemption():
    initialize_storage(backend="memory")
    store = get_exemption_store()
    store.add_pattern("10.0.*.*", "private-range")
    store.remove_pattern("10.0.*.*")
    assert "10.0.*.*" not in store.get_exempted_patterns()
