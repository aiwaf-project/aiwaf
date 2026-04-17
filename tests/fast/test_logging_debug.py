from aiwaf.fast.middleware.logging_middleware import AIWAFLoggingMiddleware


def test_logging_middleware_has_expected_log_files(tmp_path):
    middleware = AIWAFLoggingMiddleware(
        app=lambda scope, receive, send: None,
        log_dir=str(tmp_path),
    )
    assert middleware.access_log_file.name == "access.log"
    assert middleware.error_log_file.name == "error.log"
    assert middleware.aiwaf_log_file.name == "aiwaf.log"

