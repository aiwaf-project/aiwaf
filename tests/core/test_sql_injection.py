import asyncio
import io
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from aiwaf.core.sql_injection import SQLInjectionPolicy, inspect_asgi_request, inspect_wsgi_request


class PolicyTests(unittest.TestCase):
    def test_json_surrogates_and_config_environment(self):
        self.assertIsNone(SQLInjectionPolicy().inspect(body=b'{"text":"\\ud800"}'))
        with patch.dict('os.environ', {'AIWAF_SQL_INJECTION_MODE': 'monitor', 'AIWAF_PAYLOAD_MAX_BYTES': '1234'}):
            from aiwaf.core.runtime_config import AIWAFConfig
            config = AIWAFConfig()
            self.assertEqual(config.get('ip_keyword_block.sql_injection_mode'), 'monitor')
            self.assertEqual(config.get('ip_keyword_block.payload_max_bytes'), 1234)
    def test_attack_and_benign_corpus(self):
        policy = SQLInjectionPolicy()
        for value in ["admin@juice-sh.op' OR 1=1--", "admin@juice-sh.op'--", "x'/**/OR/**/1=1--",
                      "x%2527%2520OR%25201%253D1--", "x' OR 'a'='a'--", 'UNION ALL SELECT password']:
            with self.subTest(value=value):
                self.assertEqual(policy.inspect(body={'nested': [{'email': value}]}).status, 403)
        for value in ["O'Reilly", 'select a union membership', 'password#123', 'email@example.com']:
            with self.subTest(value=value):
                self.assertIsNone(policy.inspect(body={'email': value}))

    def test_query_form_and_json_escapes(self):
        policy = SQLInjectionPolicy()
        for body in [b'email=admin%27--', b'{"email":"admin\\u0027--"}']:
            self.assertEqual(policy.inspect(body=body).rule, 'sql_quote_comment')
        self.assertEqual(policy.inspect(query='q=union%20select%20password').rule, 'sql_union_select')
        for encoding in ('utf-8-sig', 'utf-16', 'utf-32'):
            self.assertEqual(policy.inspect(body='{"email":"admin\\u0027--"}'.encode(encoding)).rule, 'sql_quote_comment')

    def test_limits_modes_and_no_secret_evidence(self):
        cyclic = []; cyclic.append(cyclic)
        self.assertEqual(SQLInjectionPolicy().inspect(body=cyclic).status, 413)
        self.assertEqual(SQLInjectionPolicy(max_bytes=8).inspect(body=b'123456789').status, 413)
        self.assertIsNone(SQLInjectionPolicy(mode='off').inspect(body="admin'--"))
        policy = SQLInjectionPolicy(mode='monitor')
        finding = policy.inspect(body="private-secret'--")
        with self.assertLogs('aiwaf.payload', 'WARNING') as logs:
            self.assertIsNone(policy.enforce(finding))
        self.assertNotIn('private-secret', ''.join(logs.output))
        for kwargs in [{'mode': 'typo'}, {'max_bytes': 0}, {'max_bytes': 1.5}, {'max_bytes': True}]:
            with self.assertRaises(ValueError):
                SQLInjectionPolicy(**kwargs)

    def test_wsgi_monitor_replays_over_budget_body(self):
        raw = b'{"text":"' + b'a' * 100 + b'"}'
        request = SimpleNamespace(query_string=b'', content_type='application/json', stream=io.BytesIO(raw))
        with self.assertLogs('aiwaf.payload', 'WARNING'):
            self.assertIsNone(inspect_wsgi_request(request, SQLInjectionPolicy(mode='monitor', max_bytes=16)))
        self.assertEqual(request.stream.read(), raw)

    def test_multipart_is_not_read(self):
        class Unreadable:
            def read(self, *_):
                raise AssertionError('Upload stream consumed')
        request = SimpleNamespace(query_string=b'', content_type='multipart/form-data; boundary=x', stream=Unreadable())
        self.assertIsNone(inspect_wsgi_request(request, SQLInjectionPolicy()))

    def test_asgi_chunk_replay_including_monitor_overflow(self):
        async def exercise(mode, max_bytes):
            original = [{'type': 'http.request', 'body': b'{"x":"', 'more_body': True},
                        {'type': 'http.request', 'body': b'a' * 40, 'more_body': True},
                        {'type': 'http.request', 'body': b'"}', 'more_body': False}]
            queue = list(original)
            async def receive():
                return queue.pop(0)
            class Request:
                headers = {'content-type': 'application/json'}
                scope = {'query_string': b''}
                @property
                def receive(self):
                    return self._receive
            req = Request()
            req._receive = receive
            finding = await inspect_asgi_request(req, SQLInjectionPolicy(mode=mode, max_bytes=max_bytes))
            replayed = [await req.receive() for _ in original]
            self.assertEqual(replayed, original)
            return finding
        self.assertIsNone(asyncio.run(exercise('block', 1024)))
        with self.assertLogs('aiwaf.payload', 'WARNING'):
            self.assertIsNone(asyncio.run(exercise('monitor', 16)))

    def test_asgi_empty_chunk_budget(self):
        async def exercise():
            async def receive():
                return {'type': 'http.request', 'body': b'', 'more_body': True}
            class Request:
                headers = {'content-type': 'application/json'}
                scope = {'query_string': b''}
                @property
                def receive(self):
                    return self._receive
            request = Request()
            request._receive = receive
            return await inspect_asgi_request(request, SQLInjectionPolicy())
        self.assertEqual(asyncio.run(exercise()).status, 413)
