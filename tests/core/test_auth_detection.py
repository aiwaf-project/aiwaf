from aiwaf.core.auth_detection import AuthDetection, detect_auth_endpoint


def test_auth_detection_module_contract():
    def login_handler():
        return None

    result = detect_auth_endpoint(login_handler, framework="flask")
    assert isinstance(result, AuthDetection)
    assert result.action == "auth"
