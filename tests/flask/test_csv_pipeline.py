import pytest
from flask import Flask, request
from aiwaf.flask.csv_pipeline import NativeAIWAF
from aiwaf.flask.exemption_decorators import aiwaf_exempt, aiwaf_require_protection


def app_for(tmp_path, **config):
    app = Flask(__name__)
    @app.route('/echo', methods=['GET', 'POST'])
    def echo():
        return {'body': request.get_data().decode(), 'ok': True}
    @app.route('/exempt')
    @aiwaf_exempt
    def exempt():
        return {'ok': True}
    @app.route('/required')
    @aiwaf_exempt
    @aiwaf_require_protection('ip_keyword_block')
    def required():
        return {'ok': True}
    NativeAIWAF(app, {'data_dir': str(tmp_path), 'rate_max':10000, 'rate_flood':20000,
                     'min_form_time':0, 'required_headers':[], **config})
    return app


def test_native_flask_body_replay_and_policy_order(tmp_path, monkeypatch):
    from aiwaf.core import ip_keyword, sql_injection
    monkeypatch.setattr(ip_keyword, 'evaluate_keyword_policy', lambda *a, **k: pytest.fail('Python keyword fallback'))
    monkeypatch.setattr(sql_injection.SQLInjectionPolicy, 'inspect', lambda *a, **k: pytest.fail('Python SQL fallback'))
    app = app_for(tmp_path)
    client = app.test_client()
    assert client.post('/echo', json={'field':'value'}).json['body'] == '{"field": "value"}'
    assert client.get('/echo?q=1+union+select+1').status_code == 403
    assert client.post('/echo', json={'field':'1 uni\u006fn select 1'}).status_code == 403
    assert client.get('/wp-content/plugins/pods/init.php').status_code == 403
    assert app.extensions['aiwaf_csv_pipeline'].pipeline.native.processed_requests == 4


def test_native_flask_decorators_and_limit(tmp_path):
    client = app_for(tmp_path, max_bytes=30).test_client()
    assert client.get('/exempt?q=1+union+select+1').status_code == 200
    assert client.get('/required?q=1+union+select+1').status_code == 403
    assert client.post('/echo', data='x'*31, content_type='application/json').status_code == 413


def test_native_flask_stream_uninspected_body_and_unknown_route(tmp_path):
    client = app_for(tmp_path).test_client()
    assert client.post('/echo', data='union select', content_type='application/octet-stream').json['body'] == 'union select'
    assert client.get('/ordinary-missing-page').status_code == 404
    assert client.post('/echo', data='a=hello&b=world', content_type='application/x-www-form-urlencoded').json['body'] == 'a=hello&b=world'


needs_host_store = pytest.mark.skipif(
    not getattr(getattr(__import__("aiwaf_rust"), "NativeCSVPipeline", None), "supports", None)
    or not __import__("aiwaf_rust").NativeCSVPipeline.supports("host_store"), reason="aiwaf-rust without host stores")


@needs_host_store
def test_native_flask_uses_the_regular_flask_stores(tmp_path):
    from aiwaf.flask import storage

    app = Flask(__name__)
    app.config.update(AIWAF_DATA_DIR=str(tmp_path / "flask"), AIWAF_USE_CSV=True)
    app.add_url_rule('/page', 'page', lambda: {'ok': True})
    app.add_url_rule('/style.css', 'style', lambda: 'body{}')
    NativeAIWAF(app, {'data_dir': str(tmp_path / 'native'), 'store': 'flask', 'exempt_paths': [],
                      'enabled': ['ip_keyword_block', 'header_validation', 'ai_anomaly'],
                      'manifest_path': str(tmp_path / 'none.json')})
    class client:  # A test client whose requests come from one address.
        def __init__(self, ip):
            self.ip = ip

        def get(self, path, headers=None):
            return app.test_client().get(path, headers=headers or {}, environ_base={'REMOTE_ADDR': self.ip})
    browser = {'User-Agent': 'Mozilla/5.0 Chrome/130 Safari/537.36', 'Accept': 'text/html', 'Accept-Language': 'en',
               'Accept-Encoding': 'gzip', 'Connection': 'keep-alive'}
    with app.app_context():
        storage.add_ip_blocklist('203.0.113.80', 'stored block')
    assert client('203.0.113.80').get('/page', headers=browser).status_code == 403
    assert client('203.0.113.81').get('/wp-admin/shell.php', headers=browser).status_code == 403
    with app.app_context():
        assert storage.is_ip_blocklisted('203.0.113.81')
        storage.add_ip_allowlist('203.0.113.82')
    assert client('203.0.113.82').get('/wp-admin/shell.php', headers=browser).status_code in (200, 404)
    # Flask's own pattern exemptions match again. (The Flask tests' fixture adds browser
    # headers to every request, so the keyword check shows this rather than header validation.)
    assert client('203.0.113.83').get('/backup.php.css').status_code == 404
    assert client('203.0.113.84').get('/backup.php').status_code == 403
    # Keywords learned natively are saved in Flask's keyword store.
    assert client('203.0.113.85').get('/old/phpmyadmin-panelx').status_code == 404
    with app.app_context():
        assert 'panelx' in storage.get_top_keywords(50)
    with pytest.raises(ValueError):
        NativeAIWAF(Flask('other'), {'data_dir': str(tmp_path), 'store': 'redis'})
