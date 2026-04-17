from aiwaf.fast.storage import get_keyword_store, initialize_storage


def test_keyword_store_roundtrip_in_fast_storage():
    initialize_storage(backend="memory")
    keywords = get_keyword_store()
    keywords.add_keyword("union-select")
    assert "union-select" in keywords.get_top_keywords()

