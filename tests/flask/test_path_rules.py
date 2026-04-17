from flask import Flask

from aiwaf.flask import should_apply_middleware
from aiwaf.flask.exemption_decorators import (
    get_path_rule_for_request,
    get_path_rule_overrides,
    _is_path_rule_disabled,
)


def _make_app_with_rules(rules=None, settings=None):
    app = Flask(__name__)
    app.config['AIWAF_EXEMPT_PATHS'] = set()
    if rules is not None:
        app.config['AIWAF_PATH_RULES'] = rules
    if settings is not None:
        app.config['AIWAF_SETTINGS'] = settings
    return app


def test_path_rules_longest_prefix_wins():
    app = _make_app_with_rules(rules=[
        {'PREFIX': '/myapp/', 'DISABLE': ['header_validation']},
        {'PREFIX': '/myapp/api/', 'DISABLE': ['rate_limit']},
    ])

    with app.test_request_context('/myapp/api/data'):
        rule = get_path_rule_for_request()
        assert rule['PREFIX'] == '/myapp/api/'
        assert _is_path_rule_disabled('rate_limit') is True
        assert _is_path_rule_disabled('header_validation') is False


def test_path_rules_disable_accepts_class_names():
    app = _make_app_with_rules(rules=[
        {'PREFIX': '/v1/', 'DISABLE': ['HeaderValidationMiddleware']},
    ])

    with app.test_request_context('/v1/ping'):
        assert _is_path_rule_disabled('header_validation') is True
        assert should_apply_middleware('header_validation') is False


def test_path_rules_rate_limit_overrides():
    app = _make_app_with_rules(rules=[
        {'PREFIX': '/api/', 'RATE_LIMIT': {'WINDOW': 5, 'MAX': 10, 'FLOOD': 20}},
    ])

    with app.test_request_context('/api/list'):
        overrides = get_path_rule_overrides('RATE_LIMIT')
        assert overrides['WINDOW'] == 5
        assert overrides['MAX'] == 10
        assert overrides['FLOOD'] == 20


def test_path_rules_settings_fallback():
    app = _make_app_with_rules(settings={
        'PATH_RULES': [
            {'PREFIX': '/settings/', 'DISABLE': ['HeaderValidationMiddleware']},
        ]
    })

    with app.test_request_context('/settings/info'):
        rule = get_path_rule_for_request()
        assert rule['PREFIX'] == '/settings/'
        assert _is_path_rule_disabled('header_validation') is True
