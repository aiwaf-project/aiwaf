from aiwaf.flask.honeypot_timing_middleware import HoneypotTimingMiddleware


def test_honeypot_timing_middleware_module_contract():
    assert HoneypotTimingMiddleware.__name__ == "HoneypotTimingMiddleware"

