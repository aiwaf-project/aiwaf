from aiwaf.core.keyword_fallback import KeywordFallbackStore


def test_keyword_fallback_persists_and_ranks(tmp_path):
    path = tmp_path / "keywords.json"
    store = KeywordFallbackStore(path)
    store.add("Admin", 2)
    store.add("admin", 1)
    store.add("login", 5)
    assert store.top(1) == [("login", 5)]
    assert KeywordFallbackStore(path).top(5) == [
        ("login", 5),
        ("Admin", 2),
        ("admin", 1),
    ]

    path.write_text("broken", encoding="utf-8")
    assert KeywordFallbackStore(path).top() == []
