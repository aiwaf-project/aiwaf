from aiwaf.flask.flask_integration import AIWAF


def test_flask_integration_exports_app_installer():
    assert AIWAF.__name__ == "AIWAF"
