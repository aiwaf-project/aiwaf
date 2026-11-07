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
                      "x%2527%2520OR%25201%253D1--", "x' OR 'a'='a'--", '1 UNION ALL SELECT password']:
            with self.subTest(value=value):
                self.assertEqual(policy.inspect(body={'nested': [{'email': value}]}).status, 403)
        for value in ["O'Reilly", 'select a union membership', 'password#123', 'email@example.com',
                      'UNION ALL SELECT password', '"Item" #3']:
            with self.subTest(value=value):
                self.assertIsNone(policy.inspect(body={'email': value}))

    def test_query_form_and_json_escapes(self):
        policy = SQLInjectionPolicy()
        for body in [b'email=admin%27--', b'{"email":"admin\\u0027--"}']:
            self.assertEqual(policy.inspect(body=body).rule, 'sql_breakout_single_quote')
        self.assertEqual(policy.inspect(query='q=1%20union%20select%20password').rule, 'sql_breakout_numeric')
        # Without a value to close, this text can't change a query.
        self.assertIsNone(policy.inspect(query='q=union%20select%20password'))
        for encoding in ('utf-8-sig', 'utf-16', 'utf-32'):
            self.assertEqual(policy.inspect(body='{"email":"admin\\u0027--"}'.encode(encoding)).rule, 'sql_breakout_single_quote')

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


# The SQL check runs in aiwaf-rust. These cover the shared case files through the Python policy.
import json as _json
from pathlib import Path as _Path

import pytest as _pytest

from aiwaf.core import sql_injection as _sql
from aiwaf.core.native_runtime import native_override as _native_override

_FIXTURES = _Path(__file__).parents[1] / "fixtures"
_CASES = _json.loads((_FIXTURES / "sql_injection_cases.json").read_text(encoding="utf-8"))
_ORDINARY = _json.loads((_FIXTURES / "ordinary_text.json").read_text(encoding="utf-8"))["values"]


@_pytest.mark.parametrize("use_rust", [True, False])
def test_shared_cases_through_the_policy(use_rust):
    policy = _sql.SQLInjectionPolicy("block")
    with _native_override(use_rust):
        missed = [value for value in _CASES["attacks"] if policy.inspect(body={"field": value}) is None]
        flagged = [value for value in _CASES["ordinary"] + _ORDINARY if policy.inspect(body={"field": value}) is not None]
    assert (missed, flagged) == ([], [])


def test_each_rule_and_limit_comes_back_from_rust():
    policy = _sql.SQLInjectionPolicy("block", max_bytes=64)
    assert policy.inspect(b"id=1%20OR%201%3D1&page=2", None) == _sql.Finding("sql_breakout_numeric")
    assert policy.inspect(None, "x' OR 1=1--") == _sql.Finding("sql_breakout_single_quote")
    assert policy.inspect(None, b'{"a": "x\\" OR 1=1--"}') == _sql.Finding("sql_breakout_double_quote")
    assert policy.inspect(None, b"a" * 65) == _sql.Finding("payload_inspection_limit", 413)
    assert policy.inspect(None, {"bad": "\ud800' OR 1=1--"}).rule == "sql_breakout_single_quote"


def test_a_missing_or_old_aiwaf_rust_is_reported(monkeypatch):
    monkeypatch.setattr(_sql.rust_backend, "aiwaf_rust", None)
    with _pytest.raises(RuntimeError, match="aiwaf-rust 0.2.8"):
        _sql.SQLInjectionPolicy("block")


def test_the_accuracy_report_passes(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("sql_accuracy", _Path(__file__).parents[2] / "scripts" / "sql_accuracy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([]) == 0
    assert f"caught {len(_CASES['attacks'])} of {len(_CASES['attacks'])} attacks, flagged 0" in capsys.readouterr().out
