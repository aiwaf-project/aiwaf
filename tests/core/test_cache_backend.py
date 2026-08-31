import builtins
from contextlib import nullcontext

import pytest

from aiwaf.core import cache_backend as cache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.calls = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.calls.append(("set", key, value))
        self.values[key] = value

    def setex(self, key, ttl, value):
        self.calls.append(("setex", key, ttl, value))
        self.values[key] = value


class FakeSocket:
    def __init__(self, incoming=b""):
        self.incoming = bytearray(incoming)
        self.sent = []

    def recv(self, size):
        data = bytes(self.incoming[:size])
        del self.incoming[:size]
        return data

    def sendall(self, data):
        self.sent.append(data)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_memory_and_dict_cache_lifecycle(monkeypatch):
    memory = cache.InMemoryTTLCache(key_prefix="p:")
    memory.set("a", 1)
    assert memory.get("a") == 1
    assert memory.is_shared is False
    memory.clear()
    assert memory.get("a") is None

    mapping = {}
    backend = cache.DictCacheBackend(mapping, key_prefix="p:")
    backend.set("a", 2, ttl_seconds=1)
    assert backend.get("a") == 2
    assert backend.is_shared is False
    backend.clear()
    assert mapping == {}


def test_redis_json_cache_serializes_values(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache, "_redis_from_url", lambda _url: redis)
    backend = cache.RedisJSONCache("redis://localhost", key_prefix="test:")
    backend.set("one", {"ok": True})
    backend.set("two", [1, 2], ttl_seconds=3)
    assert backend._k("one") == "test:one"
    assert backend.get("one") == {"ok": True}
    redis.values["test:bad"] = b"not-json"
    assert backend.get("bad") is None
    assert backend.get("missing") is None
    assert backend.is_shared is True
    assert backend.clear() is None
    assert redis.calls[-1][:3] == ("setex", "test:two", 3)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        (b"+PONG\r\n", "PONG"),
        (b":7\r\n", 7),
        (b"$3\r\nabc\r\n", b"abc"),
        (b"$-1\r\n", None),
        (b"*2\r\n+OK\r\n:2\r\n", ["OK", 2]),
        (b"*-1\r\n", None),
    ],
)
def test_simple_redis_resp_parser(wire, expected):
    client = cache._SimpleRedisClient("localhost", 6379)
    assert client._read_resp(FakeSocket(wire)) == expected


def test_simple_redis_protocol_and_commands(monkeypatch):
    client = cache._SimpleRedisClient("localhost", 6379, password="pw", db=2)
    assert client._encode("GET", "x") == b"*2\r\n$3\r\nGET\r\n$1\r\nx\r\n"
    sock = FakeSocket(b"+OK\r\n+OK\r\n+PONG\r\n")
    monkeypatch.setattr(client, "_connect", lambda: sock)
    assert client.ping() is True
    assert len(sock.sent) == 3

    replies = iter([None, b"bytes", "text", 4, "OK", "OK"])
    monkeypatch.setattr(client, "_exec", lambda *_: next(replies))
    assert client.get("none") is None
    assert client.get("bytes") == b"bytes"
    assert client.get("text") == b"text"
    assert client.get("number") == b"4"
    assert client.set("key", "value") is None
    assert client.setex("key", 2, "value") is None


def test_simple_redis_connect_configures_socket(monkeypatch):
    client = cache._SimpleRedisClient("redis.local", 6380, timeout=1.5)
    sock = FakeSocket()
    sock.settimeout = lambda value: setattr(sock, "timeout", value)
    monkeypatch.setattr(cache.socket, "create_connection", lambda address, timeout: sock)
    assert client._connect() is sock
    assert sock.timeout == 1.5


def test_simple_redis_io_errors_and_error_reply():
    client = cache._SimpleRedisClient("localhost", 6379)
    with pytest.raises(ConnectionError):
        client._readline(FakeSocket())
    with pytest.raises(ConnectionError):
        client._readexact(FakeSocket(b"x"), 2)
    with pytest.raises(RuntimeError, match="redis error"):
        client._read_resp(FakeSocket(b"-NOPE\r\n"))
    with pytest.raises(RuntimeError, match="unknown redis"):
        client._read_resp(FakeSocket(b"?"))


def test_redis_url_parser_and_backend_factory(monkeypatch):
    with pytest.raises(ValueError):
        cache._redis_from_url("http://localhost")
    with pytest.raises(RuntimeError):
        cache._redis_from_url("rediss://localhost")

    real_import = builtins.__import__
    monkeypatch.setattr(
        builtins,
        "__import__",
        lambda name, *args, **kwargs: (_ for _ in ()).throw(ImportError())
        if name == "redis"
        else real_import(name, *args, **kwargs),
    )
    client = cache._redis_from_url("redis://:p%40ss@host:6380/4")
    assert (client._host, client._port, client._password, client._db) == (
        "host",
        6380,
        "p@ss",
        4,
    )
    assert isinstance(cache.make_cache_backend(cache.CacheBackendConfig()), cache.InMemoryTTLCache)
    with pytest.raises(ValueError, match="redis_url"):
        cache.make_cache_backend(cache.CacheBackendConfig(backend="redis"))
    with pytest.raises(ValueError, match="Unknown"):
        cache.make_cache_backend(cache.CacheBackendConfig(backend="other"))
