import tempfile
from pathlib import Path
import json
import os
import unittest
from unittest.mock import MagicMock, patch
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.django.test_settings')
import django
django.setup()
from django.test import RequestFactory, override_settings
from django.http import JsonResponse
from aiwaf.django.middleware import IPAndKeywordBlockMiddleware
from tests.sql_policy.cases import FrameworkCases, wire_body, echo_body


class DjangoPolicyTests(FrameworkCases, unittest.TestCase):
    def send(self, payload, mode='block', max_bytes=65536, exempt=False, query='', content_type='application/json'):
        module = 'aiwaf.django.middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        def handler(request):
            return JsonResponse(echo_body(request.body, request.content_type), status=401)
        with override_settings(AIWAF_SQL_INJECTION_MODE=mode, AIWAF_PAYLOAD_MAX_BYTES=max_bytes), \
             patch(module + 'is_middleware_disabled', return_value=False), patch(module + 'is_exempt', return_value=exempt), \
             patch(module + 'is_ip_exempted', return_value=False), patch(module + 'get_ip', return_value='93.184.216.90'), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store), \
             patch(module + 'path_exists_in_django', return_value=True), \
             patch.object(IPAndKeywordBlockMiddleware, '_collect_safe_prefixes', return_value=set()), \
             patch.object(IPAndKeywordBlockMiddleware, '_get_exempt_keywords', return_value=set()), \
             patch.object(IPAndKeywordBlockMiddleware, '_get_legitimate_path_keywords', return_value=set()):
            middleware = IPAndKeywordBlockMiddleware(handler)
            request = RequestFactory().generic('POST', '/login?' + query, wire_body(payload), content_type=content_type)
            result = middleware(request)
        return result.status_code, json.loads(result.content)


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
        module = 'aiwaf.django.middleware.'
        store = MagicMock(); store.get_top_keywords.return_value = []
        def handler(request):
            return JsonResponse(echo_body(request.body, request.content_type), status=401)
        with override_settings(AIWAF_SQL_INJECTION_MODE=mode, AIWAF_INJECTION_RISK_MODE=risk_mode, AIWAF_PATH_MANIFEST=self.manifest), \
             patch(module + 'is_middleware_disabled', return_value=False), patch(module + 'is_exempt', return_value=exempt), \
             patch(module + 'is_ip_exempted', return_value=False), patch(module + 'get_ip', return_value='93.184.216.90'), \
             patch(module + 'BlocklistManager.is_blocked', return_value=False), patch(module + 'get_keyword_store', return_value=store), \
             patch(module + 'path_exists_in_django', return_value=True), \
             patch.object(IPAndKeywordBlockMiddleware, '_collect_safe_prefixes', return_value=set()), \
             patch.object(IPAndKeywordBlockMiddleware, '_get_exempt_keywords', return_value=set()), \
             patch.object(IPAndKeywordBlockMiddleware, '_get_legitimate_path_keywords', return_value=set()):
            middleware = IPAndKeywordBlockMiddleware(handler)
            result = middleware(RequestFactory().generic('POST', '/login?' + query, wire_body(payload), content_type='application/json'))
        return result.status_code, json.loads(result.content)

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
