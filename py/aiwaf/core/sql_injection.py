"""SQL injection checks and body buffering that preserves downstream reads.

The check runs in aiwaf-rust (``policy_sql_inspect``), which is a required dependency: it walks
the query and body, decodes values, parses nested JSON, applies the size limits and runs the SQL
tokenizer in one call per request.
"""
from collections import deque
from dataclasses import dataclass
import io
import json
import logging
import os
from . import rust_backend

LOGGER = logging.getLogger("aiwaf.payload")
_RUST_NEEDED = ("policy_sql_inspect", "policy_sql_tokenizer", "policy_route_rule", "RouteRiskChecker")


def rust():
    """Return aiwaf-rust, which these checks require whatever AIWAF_USE_RUST says."""
    module = rust_backend.aiwaf_rust
    if module is None or not all(hasattr(module, name) for name in _RUST_NEEDED):
        raise RuntimeError("AIWAF's request checks need aiwaf-rust 0.2.8 or newer. Run: pip install -U 'aiwaf-rust>=0.2.8'")
    return module


def _as_bytes(value):
    """Turn a query or body into bytes. Parsed bodies are serialized as JSON first."""
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("utf-8", "surrogatepass")
    return json.dumps(value).encode("utf-8")


@dataclass(frozen=True)
class Finding:
    rule: str
    status: int = 403


class SQLInjectionPolicy:
    def __init__(self, mode=None, max_bytes=None):
        self.mode = str(mode if mode is not None else os.getenv("AIWAF_SQL_INJECTION_MODE", "block")).lower()
        if self.mode not in ("block", "monitor", "off"):
            raise ValueError("Invalid AIWAF_SQL_INJECTION_MODE")
        value = max_bytes if max_bytes is not None else os.getenv("AIWAF_PAYLOAD_MAX_BYTES", "65536")
        if isinstance(value, bool) or str(value) != str(int(value)) or not 1 <= int(value) <= 1048576:
            raise ValueError("AIWAF_PAYLOAD_MAX_BYTES must be 1..1048576")
        self.max_bytes = int(value)
        self._inspect = rust().policy_sql_inspect

    def inspect(self, query=None, body=None):
        if self.mode == "off":
            return None
        try:
            query, body = _as_bytes(query), _as_bytes(body)
        except (ValueError, RecursionError):
            # A body that can't be serialized, such as one that contains itself.
            return Finding("payload_inspection_limit", 413)
        found = self._inspect(query, body, self.max_bytes)
        return Finding(*found) if found else None

    def enforce(self, finding):
        if finding and self.mode == "monitor":
            LOGGER.warning("aiwaf.payload rule=%s mode=monitor", finding.rule)
            return None
        return finding


def inspectable_content_type(content_type):
    media = (content_type or "").split(";", 1)[0].strip().lower()
    return media == "application/json" or media.endswith("+json") or media == "application/x-www-form-urlencoded"


class _PrefixStream(io.RawIOBase):
    def __init__(self, prefix, stream):
        super().__init__()
        self.prefix, self.stream = io.BytesIO(prefix), stream

    def readable(self):
        return True

    def readinto(self, target):
        data = self.prefix.read(len(target))
        if not data:
            data = self.stream.read(len(target))
        target[:len(data)] = data
        return len(data)


def inspect_wsgi_request(request, policy, *, django=False, evaluator=None, observer=None):
    """Read at most budget+1 bytes and prepend them for the application's parser.

    The observer runs route-specific checks and may return its own blocking finding.
    """
    if policy.mode == "off":
        if observer is not None:
            finding = observer(request.META.get("QUERY_STRING", "") if django else request.query_string, None)
            if finding:
                return finding
        if evaluator is not None:
            query = request.META.get("QUERY_STRING", "") if django else request.query_string
            return evaluator(query, None)
        return None
    if django:
        query = request.META.get("QUERY_STRING", "")
        content_type = request.META.get("CONTENT_TYPE", "")
        cached = getattr(request, "_body", None)
    else:
        query, content_type = request.query_string, request.content_type
        cached = getattr(request, "_cached_data", None)
    body = None
    if inspectable_content_type(content_type):
        if cached is not None:
            body = cached
        elif django:
            body = request.read(policy.max_bytes + 1)
            request._stream = io.BufferedReader(_PrefixStream(body, request._stream))
            request._read_started = False
        else:
            stream = request.stream
            body = stream.read(policy.max_bytes + 1)
            request.stream = io.BufferedReader(_PrefixStream(body, stream))
    if observer is not None:
        finding = observer(query, body)
        if finding:
            return finding
    return policy.enforce((evaluator or policy.inspect)(query, body))


async def inspect_asgi_request(request, policy, *, evaluator=None, observer=None):
    """Replay exact ASGI messages, including an over-budget monitor-mode prefix."""
    if policy.mode == "off":
        if observer is not None:
            finding = observer(request.scope.get("query_string", b""), None)
            if finding:
                return finding
        if evaluator is not None:
            return evaluator(request.scope.get("query_string", b""), None)
        return None
    body = None
    if inspectable_content_type(request.headers.get("content-type")):
        if hasattr(request, "_body"):
            body = request._body
        else:
            original_receive, messages, chunks, size = request.receive, deque(), [], 0
            buffered, exhausted = 0, False
            while True:
                message = await original_receive()
                messages.append(message)
                if message["type"] != "http.request":
                    break
                chunk = message.get("body", b"")
                size += len(chunk)
                prefix = chunk[:max(0, policy.max_bytes + 1 - buffered)]
                chunks.append(prefix)
                buffered += len(prefix)
                if size > policy.max_bytes or not message.get("more_body", False):
                    break
                if len(messages) >= 4096:
                    exhausted = True
                    break

            async def replay():
                return messages.popleft() if messages else await original_receive()

            request._receive = replay
            body = b"".join(chunks)
            if exhausted:
                return policy.enforce(Finding("payload_inspection_limit", 413))
    query = request.scope.get("query_string", b"")
    if observer is not None:
        finding = observer(query, body)
        if finding:
            return finding
    return policy.enforce((evaluator or policy.inspect)(query, body))
