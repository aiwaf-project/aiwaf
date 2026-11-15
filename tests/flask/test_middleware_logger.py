import pytest
from flask import Flask

from aiwaf.flask.middleware_logger import AIWAFLoggerMiddleware


def test_middleware_logger_module_contract():
    assert AIWAFLoggerMiddleware.__name__ == "AIWAFLoggerMiddleware"


@pytest.mark.parametrize(
    ('configured', 'expected'),
    [
        ('aiwaf_requests.log', 'aiwaf_requests.csv'),
        ('logs/aiwaf.log', 'logs/aiwaf.csv'),
        ('/var/app.logs/access.log', '/var/app.logs/access.csv'),
        ('/srv/my.log.d/aiwaf.log', '/srv/my.log.d/aiwaf.csv'),
        ('/var/app.logs/x.csv', '/var/app.logs/x.csv'),
        ('access.log.old', 'access.log.old'),
        ('requests.txt', 'requests.txt'),
    ],
)
def test_get_csv_path_only_rewrites_trailing_log(configured, expected):
    app = Flask(__name__)
    app.config["AIWAF_MIDDLEWARE_LOG"] = configured
    middleware = AIWAFLoggerMiddleware(app)
    assert middleware._get_csv_path() == expected
