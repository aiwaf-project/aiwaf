"""Optional Bouvane integration. Installation alone never disables AIWAF."""
import json
from pathlib import Path


def edge_configuration(path='.bouvane.json'):
    path = Path(path)
    if not path.exists():
        return None
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError('Bouvane configuration exceeds 2 MiB.')
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict) or not isinstance(value.get('edge', {}), dict):
        raise ValueError('Invalid Bouvane configuration.')
    if 'edge' not in value and value.get('application') and value.get('api_url'):
        # Existing registered projects need no second enable command. The
        # implicit default only exports context; deployment stays manual.
        value = {**value, 'edge': {'enabled': True, 'mode': 'assisted', 'auto_deploy': False}, '_implicit_edge': True}
    return value if value.get('edge', {}).get('enabled') is True else None


def create_agent(path='.bouvane.json', *, framework, app=None):
    value = edge_configuration(path)
    if value is None:
        return None
    try:
        from bloud.agent import ReloadAgent
    except ImportError:
        if value.get('_implicit_edge'):
            return None
        raise RuntimeError('Edge-assisted mode requires the bouvane Python package.') from None
    from aiwaf import __version__
    from .path_manifest import build_manifest

    def manifest():
        if framework == 'django':
            from aiwaf.django.path_manifest import extract_django_routes
            routes = extract_django_routes()
        elif framework == 'flask':
            from aiwaf.flask.path_manifest import extract_flask_routes
            with app.app_context():
                routes = extract_flask_routes(app)
        elif framework == 'fastapi':
            from aiwaf.fast.path_manifest import extract_fastapi_routes
            routes = extract_fastapi_routes(app)
        else:
            raise ValueError('Unsupported Bouvane application framework.')
        return build_manifest(framework=framework, routes=routes)

    return ReloadAgent(value, manifest, aiwaf_version=__version__, base_dir=Path(path).resolve().parent)


def install_flask(app):
    if 'aiwaf_edge' in app.extensions:
        return
    agent = create_agent(app.config.get('AIWAF_BOUVANE_CONFIG', '.bouvane.json'), framework='flask', app=app)
    if agent is None:
        return
    from flask import request, jsonify
    from urllib.parse import urlsplit
    app.extensions['aiwaf_edge'] = agent

    @app.before_request
    def edge_gate():
        agent.start()
        token = request.environ.pop('HTTP_X_BOUVANE_ORIGIN_TOKEN', '')
        ip = request.headers.get('CF-Connecting-IP', '') if agent.trusted_origin(token) else request.remote_addr
        if agent.blocked(hostname=urlsplit('//' + request.host).hostname or '', path=request.path,
                         method=request.method, ip=ip, origin_token=token):
            return jsonify(error='Forbidden'), 403

    # A new application process queues a fresh export; the agent debounces
    # startup and periodically refreshes routes registered later by factories.
    agent.start()


class EdgeASGIMiddleware:
    def __init__(self, app, agent):
        self.app, self.agent = app, agent
        agent.start()

    async def __call__(self, scope, receive, send):
        self.agent.start()
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from urllib.parse import urlsplit
        request = Request(scope)
        values = request.headers.getlist('x-bouvane-origin-token')
        token = values[0] if len(values) == 1 else ''
        ip = request.headers.get('cf-connecting-ip', '') if self.agent.trusted_origin(token) else (
            request.client.host if request.client else '')
        scope = {**scope, 'headers': [(key, value) for key, value in scope['headers'] if key.lower() != b'x-bouvane-origin-token']}
        if self.agent.blocked(hostname=urlsplit('//' + request.headers.get('host', '')).hostname or '',
                              path=scope['path'], method=scope['method'], ip=ip, origin_token=token):
            return await JSONResponse({'error': 'Forbidden'}, status_code=403)(scope, receive, send)
        return await self.app(scope, receive, send)
