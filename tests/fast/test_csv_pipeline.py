import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware
from aiwaf.fast.decorators import aiwaf_exempt


def app_for(tmp_path, **config):
    app = FastAPI()
    @app.api_route('/echo', methods=['GET', 'POST'])
    async def echo(request: Request):
        return {'body': (await request.body()).decode(), 'ok':True}
    @app.get('/exempt')
    @aiwaf_exempt
    async def exempt():
        return {'ok':True}
    @app.get('/stream')
    async def stream():
        async def chunks():
            yield b'first'
            yield b'second'
        return StreamingResponse(chunks())
    app.add_middleware(NativeAIWAFMiddleware, config={'data_dir':str(tmp_path), 'rate_max':10000,
        'rate_flood':20000, 'min_form_time':0, 'required_headers':[], **config})
    return app


def test_native_fastapi_body_replay_stream_and_exemptions(tmp_path, monkeypatch):
    from aiwaf.core import sql_injection, ip_keyword
    monkeypatch.setattr(sql_injection.SQLInjectionPolicy, 'inspect', lambda *a, **k: pytest.fail('Python SQL fallback'))
    monkeypatch.setattr(ip_keyword, 'evaluate_keyword_policy', lambda *a, **k: pytest.fail('Python keyword fallback'))
    with TestClient(app_for(tmp_path)) as client:
        assert client.post('/echo', json={'field':'value'}).json()['body'] == '{"field":"value"}'
        assert client.get('/echo?q=1+union+select+1').status_code == 403
        assert client.post('/echo', content=b'{"input":"1 uni\\u006fn select 1"}', headers={'content-type':'application/json'}).status_code == 403
        assert client.get('/exempt?q=1+union+select+1').status_code == 200
        assert client.get('/stream').content == b'firstsecond'
        assert client.get('/wp-content/plugins/pods/init.php').status_code == 403


def test_native_fastapi_oversized_and_method_mismatch(tmp_path):
    with TestClient(app_for(tmp_path, max_bytes=30)) as client:
        assert client.post('/echo', content=b'x'*31, headers={'content-type':'application/json'}).status_code == 413
        assert client.post('/stream').status_code == 405


def test_native_fastapi_mounted_routes_remain_protected(tmp_path):
    app = app_for(tmp_path)
    child = FastAPI()
    @child.get('/safe')
    async def safe():
        return {'ok':True}
    app.mount('/nested', child)
    with TestClient(app) as client:
        assert client.get('/nested/safe').status_code == 200
        assert client.get('/nested/ordinary-missing-page').status_code == 404
        assert client.get('/nested/wp-content/plugins/pods/init.php').status_code == 403


def test_native_fastapi_logs_handler_errors(tmp_path):
    import csv
    app = app_for(tmp_path)
    @app.get('/boom')
    async def boom():
        raise RuntimeError('handler error')
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get('/boom').status_code == 500
    with (tmp_path / 'requests.csv').open(newline='') as file:
        assert list(csv.DictReader(file))[-1]['status_code'] == '500'


_rust = __import__("aiwaf_rust")
needs_host_store = pytest.mark.skipif(
    not getattr(getattr(_rust, "NativeCSVPipeline", None), "supports", None)
    or not _rust.NativeCSVPipeline.supports("host_store"), reason="aiwaf-rust without host stores")


@needs_host_store
def test_native_fastapi_uses_the_regular_stores(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from aiwaf.core.runtime_blocklist import BlocklistManager
    from aiwaf.core.runtime_config import AIWAFConfig
    from aiwaf.fast.csv_pipeline import NativeAIWAFMiddleware

    app = FastAPI()
    app.add_api_route('/page', lambda: {'ok': True})
    config = AIWAFConfig(load_from_env=False)
    app.add_middleware(NativeAIWAFMiddleware, config={'data_dir': str(tmp_path / 'native'), 'store': 'fastapi',
                       'exempt_paths': [], 'enabled': ['ip_keyword_block', 'geo_block', 'ai_anomaly'],
                       'manifest_path': str(tmp_path / 'none.json')},
                       aiwaf_config=config)
    client = lambda ip: TestClient(app, client=(ip, 50000))
    # Starlette builds the middleware, and so sets up the stores, on the first request.
    assert client('203.0.113.89').get('/page').status_code == 200
    BlocklistManager.block('203.0.113.90', 'stored block')
    assert client('203.0.113.90').get('/page').status_code == 403
    assert client('203.0.113.91').get('/wp-admin/shell.php').status_code == 403
    assert BlocklistManager.is_blocked('203.0.113.91')
    # The regular private-range exemptions apply again, through the exemption store.
    assert client('10.0.0.5').get('/wp-admin/shell.php').status_code == 404
    # Keywords learned natively are saved in the regular keyword store.
    from aiwaf.core.runtime_storage import get_keyword_store
    assert client('203.0.113.92').get('/old/phpmyadmin-panelx').status_code == 404
    assert 'panelx' in get_keyword_store().get_top_keywords(50)
