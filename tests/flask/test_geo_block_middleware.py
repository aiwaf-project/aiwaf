from flask import Flask

from aiwaf.flask.geo_block_middleware import GeoBlockMiddleware


def _make_app(config=None):
    app = Flask(__name__)
    app.config.update({
        'TESTING': True,
        'AIWAF_EXEMPT_PATHS': set(),
        'AIWAF_GEO_BLOCK_ENABLED': True,
        'AIWAF_GEO_BLOCK_COUNTRIES': [],
        'AIWAF_GEO_ALLOW_COUNTRIES': [],
    })
    if config:
        app.config.update(config)
    return app


def test_geo_block_allowlist_allows(monkeypatch):
    app = _make_app({'AIWAF_GEO_ALLOW_COUNTRIES': ['US']})

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'US',
    )

    GeoBlockMiddleware(app)

    @app.route('/allow')
    def allow():
        return 'OK'

    client = app.test_client()
    response = client.get('/allow', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 200


def test_geo_block_allowlist_blocks_other(monkeypatch):
    app = _make_app({'AIWAF_GEO_ALLOW_COUNTRIES': ['US']})

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'FR',
    )

    GeoBlockMiddleware(app)

    @app.route('/deny')
    def deny():
        return 'OK'

    client = app.test_client()
    response = client.get('/deny', headers={'User-Agent': 'Test Browser 1.0'})
    assert response.status_code == 403


def test_geo_block_disabled_path_rule(monkeypatch):
    app = _make_app({
        'AIWAF_GEO_BLOCK_COUNTRIES': ['US'],
        'AIWAF_PATH_RULES': [
            {'PREFIX': '/api/', 'DISABLE': ['geo_block']},
        ],
    })

    monkeypatch.setattr(
        'aiwaf.flask.geo_block_middleware.get_country_for_ip',
        lambda ip, config: 'US',
    )

    GeoBlockMiddleware(app)

    @app.route('/api/status')
    def status():
        return 'OK'

    @app.route('/ui')
    def ui():
        return 'OK'

    client = app.test_client()
    headers = {'User-Agent': 'Test Browser 1.0'}

    assert client.get('/api/status', headers=headers).status_code == 200
    assert client.get('/ui', headers=headers).status_code == 403
