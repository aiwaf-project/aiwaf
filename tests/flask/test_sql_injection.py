import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch
from flask import Flask, request, jsonify
from aiwaf.flask.ip_and_keyword_block_middleware import IPAndKeywordBlockMiddleware
from tests.sql_policy.cases import FrameworkCases, wire_body, echo_body


class FlaskPolicyTests(FrameworkCases, unittest.TestCase):
    def send(self, payload, mode='block', max_bytes=65536, exempt=False, query='', content_type='application/json'):
        app = Flask(__name__)
        app.config.update(TESTING=True, AIWAF_SQL_INJECTION_MODE=mode, AIWAF_PAYLOAD_MAX_BYTES=max_bytes,
                          AIWAF_ENABLE_KEYWORD_LEARNING=False)
        @app.post('/login')
        def login():
            return jsonify(echo_body(request.get_data(), request.content_type)), 401
        module = 'aiwaf.flask.ip_and_keyword_block_middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        with patch(module + 'is_exempt', return_value=exempt), patch(module + 'should_apply_middleware', return_value=True), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store):
            IPAndKeywordBlockMiddleware(app)
            result = app.test_client().post('/login?' + query, data=wire_body(payload), content_type=content_type)
        return result.status_code, result.get_json()


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
        app = Flask(__name__)
        app.config.update(TESTING=True, AIWAF_SQL_INJECTION_MODE=mode, AIWAF_ENABLE_KEYWORD_LEARNING=False,
                          AIWAF_INJECTION_RISK_MODE=risk_mode, AIWAF_PATH_MANIFEST=self.manifest)
        @app.post('/login')
        def login():
            return jsonify(echo_body(request.get_data(), request.content_type)), 401
        module = 'aiwaf.flask.ip_and_keyword_block_middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        with patch(module + 'is_exempt', return_value=exempt), patch(module + 'should_apply_middleware', return_value=True), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store):
            IPAndKeywordBlockMiddleware(app)
            result = app.test_client().post('/login?' + query, data=wire_body(payload), content_type='application/json')
        return result.status_code, result.get_json()

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

    def test_block_mode_is_the_default_and_blocks(self):
        with self.assertLogs('aiwaf.injection_risk', 'WARNING'):
            status, body = self.send({'email': '$(whoami)', 'password': 'private-secret'})
        self.assertEqual((status, body['error']), (403, 'command_substitution'))
        status, _ = self.send({'email': 'ordinary@example.com'})
        self.assertEqual(status, 401)

    def test_off_and_exempt_requests_are_skipped(self):
        for kwargs in ({'risk_mode': 'off'}, {'risk_mode': 'block', 'exempt': True}):
            with self.assertNoLogs('aiwaf.injection_risk', 'WARNING'):
                status, _ = self.send({'email': '$(whoami)'}, **kwargs)
            self.assertEqual(status, 401)

    def test_ordinary_text_is_never_blocked(self):
        from urllib.parse import urlencode
        from aiwaf.core.route_risks import clear_cache

        cases = Path(__file__).resolve().parents[1] / 'fixtures' / 'ordinary_text.json'
        values = json.loads(cases.read_text(encoding='utf-8'))['values']
        sinks = ['sql', 'nosql', 'command', 'path', 'ssrf', 'code', 'template', 'deserialization']
        Path(self.manifest).write_text(json.dumps({'routes': {'/login': {'injection_risks': [
            {'sink': sink, 'input': '', 'confidence': 'high'} for sink in sinks]}}}), encoding='utf-8')
        clear_cache()
        blocked = [value for value in values
                   if self.send({'email': value}, risk_mode='block', query=urlencode({'q': value}))[0] != 401]
        self.assertEqual(blocked, [])
