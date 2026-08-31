from aiwaf.flask.header_validation_middleware import HeaderValidationMiddleware


def test_header_validation_middleware_module_contract():
    assert HeaderValidationMiddleware.__name__ == "HeaderValidationMiddleware"

