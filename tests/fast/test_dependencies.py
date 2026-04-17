def test_fast_core_imports():
    import aiwaf.fast
    import aiwaf.fast.core
    import aiwaf.fast.middleware

    assert aiwaf.fast is not None

