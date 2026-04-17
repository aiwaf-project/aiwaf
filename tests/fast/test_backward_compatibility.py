def test_fast_entrypoints_present():
    import aiwaf.fast as fast

    assert hasattr(fast, "AIWAF")
    assert hasattr(fast, "aiwaf_exempt_from")

