from aiwaf.flask.utils import get_ip


def test_flask_utils_module_contract():
    assert callable(get_ip)
