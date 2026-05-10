from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from flask import Flask

from aiwaf.core.method_validation import fastapi_route_accepts_method, flask_route_accepts_method


def test_fastapi_route_accepts_method_detects_partial_mismatch():
    app = FastAPI()

    @app.get('/read-only')
    async def read_only():
        return {'ok': True}

    # Build a request scope that matches path but not method.
    with TestClient(app) as client:
        scope = {
            'type': 'http',
            'asgi': {'version': '3.0'},
            'http_version': '1.1',
            'method': 'POST',
            'path': '/read-only',
            'raw_path': b'/read-only',
            'query_string': b'',
            'headers': [],
            'client': ('testclient', 50000),
            'server': ('testserver', 80),
            'scheme': 'http',
            'root_path': '',
            'app': app,
        }
        req = SimpleNamespace(scope=scope)
        assert fastapi_route_accepts_method(req, 'POST') is False
        assert fastapi_route_accepts_method(req, 'GET') is True


def test_flask_route_accepts_method_detects_method_not_allowed():
    app = Flask(__name__)

    @app.route('/read-only', methods=['GET'])
    def read_only():
        return 'ok'

    assert flask_route_accepts_method(app, '/read-only', 'GET') is True
    assert flask_route_accepts_method(app, '/read-only', 'POST') is False


def test_flask_route_accepts_method_not_found_is_permissive():
    app = Flask(__name__)
    assert flask_route_accepts_method(app, '/does-not-exist', 'POST') is True
