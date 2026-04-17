def test_fast_does_not_require_sqlalchemy_import():
    import aiwaf.fast

    assert aiwaf.fast is not None

