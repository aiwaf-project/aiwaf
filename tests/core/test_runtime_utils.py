"""
Tests for AIWAF utility helpers.
"""
from types import SimpleNamespace

from aiwaf.core.runtime_utils import (
    get_ip,
    get_request_fingerprint,
    is_exempt,
    is_static_file,
    parse_user_agent,
    RateLimiter,
    sanitize_header_value,
)


class DummyRequest:
    """Simple stand-in for FastAPI Request objects used in tests."""

    def __init__(self, headers=None, path="/", method="GET", client_ip=None):
        self.headers = headers or {}
        self.url = SimpleNamespace(path=path)
        self.method = method
        self.scope = {"scheme": "https", "http_version": "1.1"}
        self.client = SimpleNamespace(host=client_ip) if client_ip else None


def test_get_ip_prefers_public_client():
    request = DummyRequest(headers={}, client_ip="198.51.100.10")
    assert get_ip(request) == "198.51.100.10"


def test_get_ip_uses_forwarded_for_when_internal_client():
    headers = {"x-forwarded-for": "203.0.113.5, 10.0.0.1"}
    request = DummyRequest(headers=headers, client_ip="127.0.0.1")
    assert get_ip(request) == "203.0.113.5"


def test_is_exempt_respects_default_paths():
    request = DummyRequest(path="/health")
    assert is_exempt(request)
    request = DummyRequest(path="/api/data")
    assert not is_exempt(request)


def test_is_static_file_detects_common_extensions():
    assert is_static_file("/static/main.css")
    assert is_static_file("/assets/image.png")
    assert not is_static_file("/api/data")


def test_sanitize_header_value_truncates_and_cleans():
    raw = "\x01" + "A" * 600 + "\x02"
    sanitized = sanitize_header_value(raw, max_length=10)
    assert sanitized.endswith("...")
    assert "\x01" not in sanitized and "\x02" not in sanitized
    assert len(sanitized) <= 13


def test_parse_user_agent_identifies_browser_and_os():
    chrome_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/92.0.4515.159 Safari/537.36"
    )
    parsed = parse_user_agent(chrome_ua)
    assert parsed["browser"] == "chrome"
    assert parsed["os"] == "windows"


def test_get_request_fingerprint_is_stable():
    request = DummyRequest(
        headers={"user-agent": "test", "accept": "*/*", "accept-language": "en"},
        method="POST",
        path="/api/data",
    )
    fingerprint1 = get_request_fingerprint(request)
    fingerprint2 = get_request_fingerprint(request)
    assert fingerprint1 == fingerprint2


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter()
    ip = "203.0.113.9"
    path = "/check"
    for _ in range(5):
        assert not limiter.is_rate_limited(ip, path, max_requests=5, window_seconds=10)
    assert limiter.is_rate_limited(ip, path, max_requests=5, window_seconds=10)
def test_private_ip_and_allowlist_compatibility():
    from aiwaf.core.runtime_utils import ip_in_allowlist, is_private_ip

    assert is_private_ip("10.0.0.1")
    assert not is_private_ip("not-an-ip")
    assert ip_in_allowlist("203.0.113.1", ["203.0.113.1"])
