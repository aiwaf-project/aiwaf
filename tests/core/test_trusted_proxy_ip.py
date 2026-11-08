"""Client IP resolution must not let a direct client choose its own address."""
from types import SimpleNamespace

import pytest

from aiwaf.core.request_context import (
    extract_ip_from_django_request,
    extract_ip_from_flask_request,
    resolve_ip_from_fastapi_request,
)
from aiwaf.core.utils import resolve_client_ip

PUBLIC_PEER = "198.51.100.7"
PROXY = "10.0.0.1"
CLIENT = "203.0.113.5"
SPOOFED = "127.0.0.1"


@pytest.fixture(autouse=True)
def default_trust(monkeypatch):
    monkeypatch.delenv("AIWAF_TRUSTED_PROXIES", raising=False)


def django_request(peer, xff):
    return SimpleNamespace(META={"REMOTE_ADDR": peer, "HTTP_X_FORWARDED_FOR": xff})


def flask_request(peer, xff):
    return SimpleNamespace(headers={"X-Forwarded-For": xff}, remote_addr=peer)


def fastapi_request(peer, xff):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers={"x-forwarded-for": xff})


@pytest.mark.parametrize(
    "build, resolve",
    [
        (django_request, extract_ip_from_django_request),
        (flask_request, extract_ip_from_flask_request),
        (fastapi_request, resolve_ip_from_fastapi_request),
    ],
    ids=["django", "flask", "fastapi"],
)
class TestAdapters:
    def test_direct_public_client_cannot_spoof(self, build, resolve):
        assert resolve(build(PUBLIC_PEER, SPOOFED)) == PUBLIC_PEER

    def test_spoofed_prefix_through_proxy_is_ignored(self, build, resolve):
        assert resolve(build(PROXY, f"{SPOOFED}, {CLIENT}")) == CLIENT

    def test_env_can_trust_a_public_load_balancer(self, build, resolve, monkeypatch):
        monkeypatch.setenv("AIWAF_TRUSTED_PROXIES", f"{PUBLIC_PEER}/32")
        assert resolve(build(PUBLIC_PEER, CLIENT)) == CLIENT

    def test_env_can_disable_forwarded_headers(self, build, resolve, monkeypatch):
        monkeypatch.setenv("AIWAF_TRUSTED_PROXIES", "")
        assert resolve(build(PROXY, CLIENT)) == PROXY


@pytest.mark.parametrize(
    "peer, xff, expected",
    [
        # Chain of proxies: skip trusted hops from the right.
        (PROXY, f"{CLIENT}, 10.0.0.2, 10.0.0.3", CLIENT),
        # An internal client can't choose another internal address.
        (PROXY, "10.0.0.9, 10.0.0.5", "10.0.0.5"),
        # A malformed hop stops the walk rather than exposing what's left of it.
        (PROXY, f"{SPOOFED}, garbage", PROXY),
        (PROXY, f"{CLIENT}, unknown", PROXY),
        # Addresses with ports and IPv4-mapped forms are normalized.
        (PROXY, f"{CLIENT}:4711", CLIENT),
        (PROXY, "[2001:db8::1]:4711", "2001:db8::1"),
        (f"::ffff:{PROXY}", f"::ffff:{CLIENT}", CLIENT),
        (f"::ffff:{PUBLIC_PEER}", SPOOFED, PUBLIC_PEER),
        # A Unix socket has no peer address and is reachable only locally.
        ("", CLIENT, CLIENT),
        (None, CLIENT, CLIENT),
        # Non-IP peers are never trusted.
        ("testclient", SPOOFED, "testclient"),
        # An oversized header is ignored.
        (PROXY, "1.1.1.1," * 1000, PROXY),
    ],
)
def test_forwarded_for_walk(peer, xff, expected):
    assert resolve_client_ip(peer, xff) == expected


def test_real_ip_only_from_trusted_peer_without_forwarded_for():
    assert resolve_client_ip(PROXY, None, CLIENT) == CLIENT
    assert resolve_client_ip(PUBLIC_PEER, None, SPOOFED) == PUBLIC_PEER
    assert resolve_client_ip(PROXY, CLIENT, SPOOFED) == CLIENT


def test_invalid_trust_entries_do_not_widen_trust():
    assert resolve_client_ip(PUBLIC_PEER, SPOOFED, None, ["not-a-network"]) == PUBLIC_PEER


def test_explicit_configuration_overrides_defaults():
    assert resolve_client_ip(PROXY, CLIENT, None, [PUBLIC_PEER]) == PROXY
    assert resolve_client_ip(PUBLIC_PEER, CLIENT, None, f"{PUBLIC_PEER}/32, 2001:db8::/32") == CLIENT


def test_repeated_header_values_are_all_walked():
    headers = SimpleNamespace(
        getlist=lambda name: ["127.0.0.1", CLIENT] if name == "x-forwarded-for" else [],
        get=lambda name, default=None: default,
    )
    request = SimpleNamespace(client=SimpleNamespace(host=PROXY), headers=headers)
    assert resolve_ip_from_fastapi_request(request) == CLIENT


def test_flask_config_setting_and_public_peer_end_to_end():
    flask = pytest.importorskip("flask")
    app = flask.Flask(__name__)
    environ = {"REMOTE_ADDR": PUBLIC_PEER}
    headers = {"X-Forwarded-For": SPOOFED}
    with app.test_request_context("/", headers=headers, environ_base=environ):
        assert extract_ip_from_flask_request(flask.request) == PUBLIC_PEER
    app.config["AIWAF_TRUSTED_PROXIES"] = [PUBLIC_PEER]
    with app.test_request_context("/", headers={"X-Forwarded-For": CLIENT}, environ_base=environ):
        assert extract_ip_from_flask_request(flask.request) == CLIENT


def test_flask_logging_cannot_be_poisoned_by_other_headers():
    flask = pytest.importorskip("flask")
    from aiwaf.flask.logging_middleware import AIWAFLoggingMiddleware

    app = flask.Flask(__name__)
    headers = {"CF-Connecting-IP": SPOOFED, "X-Forwarded-For": SPOOFED, "X-Real-IP": SPOOFED}
    with app.test_request_context("/", headers=headers, environ_base={"REMOTE_ADDR": PUBLIC_PEER}):
        assert AIWAFLoggingMiddleware._get_client_ip(object()) == PUBLIC_PEER
