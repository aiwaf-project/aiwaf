from aiwaf.fast.storage import get_storage, initialize_storage


def test_storage_backend_reinitialize_is_safe():
    initialize_storage(backend="memory")
    first = get_storage()
    initialize_storage(backend="memory")
    second = get_storage()
    assert first is not None
    assert second is not None

