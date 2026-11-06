import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from aiwaf.fast.middleware.ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from tests.sql_policy.cases import FrameworkCases, wire_body, echo_body


class FastAPIPolicyTests(FrameworkCases, unittest.TestCase):
    def send(self, payload, mode='block', max_bytes=65536, exempt=False, query='', content_type='application/json'):
        app = FastAPI()
        @app.post('/login')
        async def login(request: Request):
            return JSONResponse(echo_body(await request.body(), request.headers['content-type']), status_code=401)
        app.add_middleware(IPAndKeywordBlockMiddleware, sql_injection_mode=mode, payload_max_bytes=max_bytes)
        module = 'aiwaf.fast.middleware.ip_and_keyword_block_middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        with patch(module + 'is_exempt', return_value=exempt), patch(module + 'should_apply_middleware', return_value=True), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store):
            with TestClient(app) as client:
                result = client.post('/login?' + query, content=wire_body(payload), headers={'content-type': content_type})
        return result.status_code, result.json()


class RouteRiskLogTests(unittest.TestCase):
    def setUp(self):
        from aiwaf.core.route_risks import clear_cache
        clear_cache()
        self.addCleanup(clear_cache)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.manifest = str(Path(directory.name) / 'paths.json')
        Path(self.manifest).write_text(json.dumps({'routes': {'/login': {'injection_risks': [
            {'sink': 'command', 'input': 'email', 'confidence': 'high'}]}}}), encoding='utf-8')

    def send(self, payload, mode='block', exempt=False, query='', risk_mode=None):
        app = FastAPI()
        app.state.aiwaf_config = {'AIWAF_INJECTION_RISK_MODE': risk_mode, 'path_manifest': self.manifest}
        @app.post('/login')
        async def login(request: Request):
            return JSONResponse(echo_body(await request.body(), request.headers['content-type']), status_code=401)
        app.add_middleware(IPAndKeywordBlockMiddleware, sql_injection_mode=mode)
        module = 'aiwaf.fast.middleware.ip_and_keyword_block_middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        with patch(module + 'is_exempt', return_value=exempt), patch(module + 'should_apply_middleware', return_value=True), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store):
            with TestClient(app) as client:
                result = client.post('/login?' + query, content=wire_body(payload), headers={'content-type': 'application/json'})
        return result.status_code, result.json()

    def test_log_mode_logs_and_leaves_the_request_unchanged(self):
        payload = {'email': '$(whoami)', 'password': 'private-secret'}
        with self.assertLogs('aiwaf.injection_risk', 'WARNING') as logs:
            status, body = self.send(payload, risk_mode='log')
        self.assertEqual((status, body['received']), (401, payload))
        event = json.loads(logs.records[0].getMessage())
        self.assertEqual((event['route'], event['input'], event['rule']), ('/login', 'email', 'command_substitution'))
        self.assertNotIn('private-secret', ''.join(logs.output))
        self.assertNotIn('whoami', ''.join(logs.output))

    def test_log_mode_works_with_sql_inspection_off(self):
        with self.assertLogs('aiwaf.injection_risk', 'WARNING'):
            status, _ = self.send({'x': 1}, risk_mode='log', mode='off', query='email=$(id)')
        self.assertEqual(status, 401)

    def test_it_is_off_by_default_and_skips_exempt_requests(self):
        for kwargs in ({'risk_mode': None}, {'risk_mode': 'off'}, {'risk_mode': 'log', 'exempt': True}):
            with self.assertNoLogs('aiwaf.injection_risk', 'WARNING'):
                status, _ = self.send({'email': '$(whoami)'}, **kwargs)
            self.assertEqual(status, 401)
