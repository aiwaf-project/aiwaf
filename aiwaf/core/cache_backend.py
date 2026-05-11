"""Cache backend helpers used by framework adapters.

Currently used for rate limiting buckets in Flask/FastAPI adapters so deployments
with multiple workers can share state via an external cache when desired.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import urlparse, unquote


class CacheBackend(Protocol):
    def get(self, key: str) -> Any: ...

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None: ...

    def clear(self) -> None: ...

    @property
    def is_shared(self) -> bool: ...


@dataclass(frozen=True)
class CacheBackendConfig:
    backend: str = "memory"  # memory | redis
    redis_url: Optional[str] = None
    key_prefix: str = "aiwaf:"


class InMemoryTTLCache(CacheBackend):
    def __init__(self, *, key_prefix: str = "aiwaf:"):
        self._key_prefix = key_prefix or ""
        self._lock = threading.Lock()
        self._items: dict[str, tuple[Optional[float], Any]] = {}

    def _k(self, key: str) -> str:
        return f"{self._key_prefix}{key}" if self._key_prefix else key

    def get(self, key: str) -> Any:
        now = time.time()
        cache_key = self._k(key)
        with self._lock:
            entry = self._items.get(cache_key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and now >= expires_at:
                self._items.pop(cache_key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + max(float(ttl_seconds), 0.0)
        cache_key = self._k(key)
        with self._lock:
            self._items[cache_key] = (expires_at, value)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    @property
    def is_shared(self) -> bool:
        return False


class DictCacheBackend(CacheBackend):
    """A thin wrapper around an external dict-like mapping.

    This is primarily for backwards compatibility with older adapters/tests that
    directly clear module-level dictionaries.
    """

    def __init__(self, mapping: dict, *, key_prefix: str = ""):
        self._mapping = mapping
        self._key_prefix = key_prefix or ""

    def _k(self, key: str) -> str:
        return f"{self._key_prefix}{key}" if self._key_prefix else key

    def get(self, key: str) -> Any:
        return self._mapping.get(self._k(key))

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        # TTL is ignored for this legacy backend.
        self._mapping[self._k(key)] = value

    def clear(self) -> None:
        try:
            self._mapping.clear()
        except Exception:
            return

    @property
    def is_shared(self) -> bool:
        return False


class RedisJSONCache(CacheBackend):
    """Redis cache storing values as JSON.

    Notes:
    - This is a minimal adapter. Rate limiting updates are not atomic across
      concurrent requests; for strict correctness under high concurrency, use a
      Redis-native data structure + Lua script.
    """

    def __init__(self, redis_url: str, *, key_prefix: str = "aiwaf:"):
        self._redis = _redis_from_url(redis_url)
        self._key_prefix = key_prefix or ""

    def _k(self, key: str) -> str:
        return f"{self._key_prefix}{key}" if self._key_prefix else key

    def get(self, key: str) -> Any:
        raw = self._redis.get(self._k(key))
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        cache_key = self._k(key)
        if ttl_seconds is None:
            self._redis.set(cache_key, payload)
        else:
            self._redis.setex(cache_key, int(max(float(ttl_seconds), 0.0)), payload)

    def clear(self) -> None:  # pragma: no cover
        # Not safe to flush the whole DB; best-effort no-op.
        return

    @property
    def is_shared(self) -> bool:
        return True


class _SimpleRedisClient:
    """Minimal Redis client speaking RESP over TCP.

    Implements only what AIWAF needs for rate limiting:
    - PING, GET, SET, SETEX, SELECT, AUTH

    This avoids requiring `redis` (redis-py) at runtime, which is convenient for
    lightweight installs and for Docker-based integration tests.
    """

    def __init__(self, host: str, port: int, *, password: Optional[str] = None, db: int = 0, timeout: float = 2.0):
        self._host = host
        self._port = int(port)
        self._password = password
        self._db = int(db)
        self._timeout = float(timeout)

    def _connect(self) -> socket.socket:
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        sock.settimeout(self._timeout)
        return sock

    def _encode(self, *parts: str) -> bytes:
        out = [f"*{len(parts)}\r\n".encode("utf-8")]
        for part in parts:
            b = part.encode("utf-8")
            out.append(f"${len(b)}\r\n".encode("utf-8"))
            out.append(b + b"\r\n")
        return b"".join(out)

    def _readline(self, sock: socket.socket) -> bytes:
        buf = bytearray()
        while True:
            chunk = sock.recv(1)
            if not chunk:
                raise ConnectionError("redis connection closed")
            buf += chunk
            if buf.endswith(b"\r\n"):
                return bytes(buf[:-2])

    def _readexact(self, sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("redis connection closed")
            buf += chunk
        return bytes(buf)

    def _read_resp(self, sock: socket.socket) -> Any:
        prefix = sock.recv(1)
        if not prefix:
            raise ConnectionError("redis connection closed")
        if prefix == b"+":
            return self._readline(sock).decode("utf-8", errors="ignore")
        if prefix == b"-":
            msg = self._readline(sock).decode("utf-8", errors="ignore")
            raise RuntimeError(f"redis error: {msg}")
        if prefix == b":":
            return int(self._readline(sock))
        if prefix == b"$":
            length = int(self._readline(sock))
            if length == -1:
                return None
            data = self._readexact(sock, length)
            _ = self._readexact(sock, 2)  # \r\n
            return data
        if prefix == b"*":
            length = int(self._readline(sock))
            if length == -1:
                return None
            return [self._read_resp(sock) for _ in range(length)]
        raise RuntimeError(f"unknown redis resp prefix: {prefix!r}")

    def _exec(self, *cmd: str) -> Any:
        with self._connect() as sock:
            # One-time per-connection setup.
            if self._password:
                sock.sendall(self._encode("AUTH", self._password))
                self._read_resp(sock)
            if self._db:
                sock.sendall(self._encode("SELECT", str(self._db)))
                self._read_resp(sock)

            sock.sendall(self._encode(*cmd))
            return self._read_resp(sock)

    def ping(self) -> bool:
        return self._exec("PING") == "PONG"

    def get(self, key: str) -> Optional[bytes]:
        value = self._exec("GET", key)
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, str):
            return value.encode("utf-8")
        return str(value).encode("utf-8")

    def set(self, key: str, value: str) -> None:
        self._exec("SET", key, value)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self._exec("SETEX", key, str(int(ttl_seconds)), value)


def _redis_from_url(redis_url: str):
    """Return a redis client (redis-py if installed, else a simple TCP client)."""
    parsed = urlparse(redis_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in {"redis", "rediss"}:
        raise ValueError(f"Unsupported Redis URL scheme: {parsed.scheme}")
    if scheme == "rediss":
        raise RuntimeError("rediss:// requires redis-py; TLS not supported by built-in client")

    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6379
    password = unquote(parsed.password) if parsed.password else None
    db = 0
    if parsed.path and parsed.path.strip("/"):
        try:
            db = int(parsed.path.strip("/"))
        except Exception:
            db = 0

    try:
        import redis  # type: ignore
    except Exception:
        return _SimpleRedisClient(host, port, password=password, db=db)

    client = redis.Redis.from_url(redis_url)
    return client


def make_cache_backend(cfg: CacheBackendConfig) -> CacheBackend:
    backend = (cfg.backend or "memory").strip().lower()
    if backend in {"memory", "inmemory", "local"}:
        return InMemoryTTLCache(key_prefix=cfg.key_prefix)
    if backend in {"redis"}:
        if not cfg.redis_url:
            raise ValueError("redis_url is required for redis cache backend")
        return RedisJSONCache(cfg.redis_url, key_prefix=cfg.key_prefix)
    raise ValueError(f"Unknown cache backend: {cfg.backend}")
