from aiwaf.fast.storage import get_storage, initialize_storage


def test_storage_initializes_without_explicit_data_dir():
    initialize_storage(backend="memory")
    assert get_storage() is not None

