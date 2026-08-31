from aiwaf.flask.middleware_logger import AIWAFLoggerMiddleware


def test_middleware_logger_module_contract():
    assert AIWAFLoggerMiddleware.__name__ == "AIWAFLoggerMiddleware"
