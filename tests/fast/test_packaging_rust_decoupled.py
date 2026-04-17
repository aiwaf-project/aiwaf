from aiwaf.fast.rust_backend import rust_available


def test_rust_backend_availability_call_safe():
    assert isinstance(rust_available(), bool)

