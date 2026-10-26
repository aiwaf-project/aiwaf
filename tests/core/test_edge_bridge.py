import json
from unittest.mock import Mock, patch
import pytest
from aiwaf.core.edge_bridge import create_agent, edge_configuration, install_flask, EdgeASGIMiddleware


def test_installation_alone_does_not_enable_edge_mode(tmp_path):
    path = tmp_path / '.bouvane.json'
    assert create_agent(path, framework='django') is None


def test_registered_legacy_project_automatically_enables_context_sync(tmp_path):
    path = tmp_path / '.bouvane.json'
    path.write_text(json.dumps({'application': '00000000-0000-0000-0000-000000000001',
                                'api_url': 'https://example.com/api/v1'}))
    config = edge_configuration(path)
    assert config['edge'] == {'enabled': True, 'mode': 'assisted', 'auto_deploy': False}
    with patch('bloud.agent.ReloadAgent') as agent:
        create_agent(path, framework='django')
        agent.assert_called_once()
    with patch.dict('sys.modules', {'bloud.agent': None}):
        assert create_agent(path, framework='django') is None
    path.write_text(json.dumps({'application': 'not-a-valid-id', 'edge': {'enabled': False}}))
    assert create_agent(path, framework='django') is None


def test_explicit_mode_requires_the_optional_package(tmp_path):
    path = tmp_path / '.bouvane.json'
    path.write_text(json.dumps({'edge': {'enabled': True}}))
    with patch.dict('sys.modules', {'bloud.agent': None}):
        with pytest.raises(RuntimeError, match='bouvane Python package'):
            create_agent(path, framework='django')


def test_create_agent_is_lazy_and_uses_actual_route_export(tmp_path):
    from flask import Flask
    app = Flask(__name__)
    app.add_url_rule('/example', view_func=lambda: 'example')
    path = tmp_path / '.bouvane.json'
    path.write_text(json.dumps({'application': '00000000-0000-0000-0000-000000000001',
        'api_url': 'https://example.com/api/v1', 'edge': {'enabled': True}}))
    with patch('bloud.agent.ReloadAgent') as agent:
        create_agent(path, framework='flask', app=app)
        factory = agent.call_args.args[1]
        agent.return_value.start.assert_not_called()
        manifest = factory()
    assert manifest['framework'] == 'flask'
    assert '/example' in manifest['routes']


def test_flask_hook_preserves_local_fallback_and_strips_origin_secret():
    from flask import Flask, request
    app = Flask(__name__)
    agent = Mock()
    agent.trusted_origin.return_value = False
    agent.blocked.return_value = True
    with patch('aiwaf.core.edge_bridge.create_agent', return_value=agent):
        install_flask(app)
        install_flask(app)
    app.add_url_rule('/retired', view_func=lambda: {'secret_present': 'X-Bouvane-Origin-Token' in request.headers})
    client = app.test_client()
    assert client.get('/retired').status_code == 403
    agent.blocked.return_value = False
    response = client.get('/retired', headers={'X-Bouvane-Origin-Token': 'private'})
    assert response.status_code == 200
    assert response.json == {'secret_present': False}
    assert len(app.before_request_funcs[None]) == 1
    assert agent.blocked.call_args.kwargs['origin_token'] == 'private'


def test_asgi_hook_preserves_inner_middleware_and_strips_origin_secret():
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    app = FastAPI()
    @app.get('/retired')
    def view(request: Request):
        return {'secret_present': 'x-bouvane-origin-token' in request.headers}
    agent = Mock()
    agent.trusted_origin.return_value = False
    agent.blocked.return_value = True
    app.add_middleware(EdgeASGIMiddleware, agent=agent)
    with TestClient(app) as client:
        assert client.get('/retired').status_code == 403
        agent.blocked.return_value = False
        response = client.get('/retired', headers={'x-bouvane-origin-token': 'private'})
        assert response.json() == {'secret_present': False}
        assert agent.blocked.call_args.kwargs['origin_token'] == 'private'


def test_django_hook_preserves_inner_middleware_and_strips_origin_secret():
    from django.test import RequestFactory
    from aiwaf.django.edge import EdgeAssistedMiddleware
    agent = Mock()
    agent.trusted_origin.return_value = False
    agent.blocked.return_value = True
    downstream = Mock(return_value='application response')
    with patch('aiwaf.django.edge.create_agent', return_value=agent):
        middleware = EdgeAssistedMiddleware(downstream)
    request = RequestFactory().get('/retired', HTTP_X_BOUVANE_ORIGIN_TOKEN='private')
    assert middleware(request).status_code == 403
    downstream.assert_not_called()
    agent.blocked.return_value = False
    assert middleware(RequestFactory().get('/allowed')) == 'application response'
    assert 'HTTP_X_BOUVANE_ORIGIN_TOKEN' not in request.META


def test_invalid_configuration_fails_without_silent_downgrade(tmp_path):
    path = tmp_path / '.bouvane.json'
    path.write_text('[]')
    with pytest.raises(ValueError):
        edge_configuration(path)
