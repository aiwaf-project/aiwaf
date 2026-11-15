"""The Python and Rust implementations must give the same answer for the same input."""
from datetime import datetime

import pytest

import aiwaf_rust  # noqa: E402

pytest.importorskip("aiwaf")

from aiwaf.core.logs import parse_log_line  # noqa: E402

PREFIX = '203.0.113.5 - - [10/Oct/2026:12:00:00 +0000] '
LOG_LINES = [
    PREFIX + '"GET /a HTTP/1.1" 200 12 "-" "ua" response-time=0.125',
    PREFIX + '"GET /a HTTP/1.1" 200 12 "-" "ua"',
    PREFIX + '"POST /a?b=1 HTTP/1.1" 404 12',
    PREFIX + '"PATCH /a HTTP/1.1" 200 12 "-" "ua"',
    PREFIX + '"GET /old HTTP/1.1" 304 - "-" "ua"',
    PREFIX + '"GET /a HTTP/1.1" 304 -',
    PREFIX + '"GET /a HTTP/1.1" 200 12 "-" "ua" 5ms - "-"',
    PREFIX + '"GET /a HTTP/1.1" 200 - "-" "ua" 5ms - "-"',
    PREFIX + '"GET /a HTTP/1.1" 200 - "-" "ua" response-time=1',
    PREFIX + '"GET /a HTTP/1.1" 200 1234 "-" "ua" 1500ms BLOCKED "rate"',
    PREFIX + '"GET /items HTTP/1.1" 200 1234 "-" "ua"',
    PREFIX + '"GET /a HTTP/1.1" 200 12 "-" "1.5"',
    PREFIX + '"GET /a HTTP/1.1" 200 12 "-" "ua" 5msx',
    '2001:db8::1 - - [10/Oct/2026:12:00:00 +0000] "GET /a HTTP/1.1" 200 12 "-" "ua"',
    "server started on port 8000",
    "",
]


def python_fields(line):
    record = parse_log_line(line)
    if record is None:
        return None
    return record["ip"], record["path"], record["status"], record["response_time"]


def rust_fields(line):
    candidates = aiwaf_rust.training_parse_lines([line])[0]
    if not candidates:
        return None
    ip, _timestamp, path, status, elapsed = candidates[0]
    return ip, path, status, elapsed


@pytest.mark.parametrize("line", LOG_LINES)
def test_log_parsers_agree(line):
    assert rust_fields(line) == python_fields(line)


@pytest.mark.parametrize("stamp", [
    "2026-10-12T01:02:03",
    "2026-10-12T01:02:03Z",
    "2026-10-12T01:02:03+05:30",
    "2026-10-12T01:02:03-05:00",
    "2026-10-12T01:02:03.5",
    "2026-10-12T01:02:03.123456+00:00",
])
def test_iso_date_parsers_agree(stamp):
    expected = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    offset = expected.utcoffset()
    parts = aiwaf_rust.training_date_parts([stamp], False)[0]
    assert parts == (
        expected.year, expected.month, expected.day, expected.hour, expected.minute,
        expected.second, expected.microsecond,
        None if offset is None else int(offset.total_seconds()),
    )


@pytest.mark.parametrize("text", [
    "123e4567-e89b-12d3-a456-426614174000",
    "123E4567-E89B-12D3-A456-426614174000",
    "123e4567e89b12d3a456426614174000",
    "g23e4567-e89b-12d3-a456-426614174000",
])
def test_uuid_checks_agree_with_the_standard_library(text):
    import uuid

    try:
        uuid.UUID(text)
        valid = len(text) == 36
    except ValueError:
        valid = False
    assert aiwaf_rust.policy_uuid(text) is valid


# Client IP resolution: aiwaf-rust's TrustedProxySet must match the Python fallback exactly.
from aiwaf.core import utils as _utils  # noqa: E402

DEFAULT_TRUST = _utils.DEFAULT_TRUSTED_PROXIES
TRUST_CONFIGS = [
    DEFAULT_TRUST,
    DEFAULT_TRUST + _utils.CLOUDFLARE_PROXIES,
    ("10.0.0.0/255.0.0.0", "192.168.1.0/0.0.0.255", "2001:db8::/32", "203.0.113.9"),
    ("10.0.0.0/33", "not-a-network", "::/129", "10.0.0.0/8/1", "010.0.0.0/8", "10.0.0.0/+8"),
    (),
]
ADDRESSES = [
    "", " ", "unknown", "10.0.0.1", " 10.0.0.1 ", "10.0.0.1:8080", "203.0.113.5", "203.0.113.5:443",
    "010.0.0.1", "1.2.3", "256.1.1.1", "::1", "[::1]", "[::1]:443", "[2001:DB8::1]:8443", "2001:DB8:0:0:0:0:0:1",
    "::ffff:10.0.0.1", "::ffff:203.0.113.5", "::ffff:a00:1", "::1.2.3.4", "::ffff:0:1.2.3.4", "fe80::1%eth0",
    "fe80::1%", "fe80::1%a%b", "2001:db8::1%25", "[fe80::1%eth0]:80", "104.16.0.1", "2606:4700::1",
    "2c0f:f248::1", "2001:0db8::0001", "1:2:3:4:5:6:7:8", "1::", "[", "[::1", "::1]", "\x1c10.0.0.1\x1f",
    "１０.0.0.1", "10.0.0.1\n",
]
FORWARDED = [
    None, "", "203.0.113.5", "127.0.0.1, 203.0.113.5, 10.0.0.2", "203.0.113.5, junk, 10.0.0.2",
    "10.0.0.3, 10.0.0.2", " 203.0.113.5 ,10.0.0.2 ", "2001:db8::5, 104.16.0.1", "203.0.113.5,,10.0.0.2",
    ", ".join(["10.0.0.2"] * 40), ", ".join(["203.0.113.7"] + ["104.16.0.1"] * 31),
    "1" * 4096, "1" * 4100 + ",203.0.113.5", "é" * 3000 + ",203.0.113.5", "[2001:DB8::9]:1234, 10.0.0.2",
]
REAL_IPS = [None, "", "203.0.113.9", "unknown", " 2001:DB8::7 "]


def _both(peer, forwarded_for, real_ip, entries, monkeypatch):
    monkeypatch.setattr(_utils, "_untrusted_peer_warned", True)  # keep logs quiet
    native = aiwaf_rust.TrustedProxySet(list(entries))
    rust_ip, _ = native.resolve(str(peer or ""), str(forwarded_for or ""), str(real_ip or ""))
    return rust_ip, _utils._resolve_client_ip_python(peer, forwarded_for, real_ip, tuple(entries))


@pytest.mark.parametrize("entries", TRUST_CONFIGS, ids=["default", "cloudflare", "netmasks", "invalid", "none"])
def test_client_ip_resolution_agrees(entries, monkeypatch):
    for peer in ADDRESSES:
        for forwarded_for in FORWARDED:
            for real_ip in REAL_IPS:
                rust_ip, python_ip = _both(peer, forwarded_for, real_ip, entries, monkeypatch)
                assert rust_ip == python_ip, (peer, forwarded_for, real_ip)


@pytest.mark.parametrize("address", ADDRESSES + [a for f in FORWARDED if f for a in f.split(",")])
def test_address_normalization_agrees(address, monkeypatch):
    # With every address trusted, a single hop is returned as Rust or Python writes it.
    rust_ip, python_ip = _both("10.0.0.1", address, None, ("0.0.0.0/0", "::/0"), monkeypatch)
    assert rust_ip == python_ip


def test_invalid_entries_agree():
    entries = TRUST_CONFIGS[3]
    python_valid = len(_utils._parse_trusted_networks(tuple(entries)))
    native = aiwaf_rust.TrustedProxySet(list(entries))
    assert len(entries) - len(native.invalid) == python_valid
    assert native.is_empty() == (python_valid == 0)


def test_ignored_header_is_reported_like_python(monkeypatch, caplog):
    native = aiwaf_rust.TrustedProxySet(list(DEFAULT_TRUST))
    assert native.resolve("198.51.100.7", "203.0.113.5", "") == ("198.51.100.7", "X-Forwarded-For")
    assert native.resolve("198.51.100.7", "", "203.0.113.5") == ("198.51.100.7", "X-Real-IP")
    assert native.resolve("198.51.100.7", "", "") == ("198.51.100.7", None)
    assert native.resolve("unknown", "203.0.113.5", "") == ("unknown", None)
    assert native.resolve("10.0.0.1", "203.0.113.5", "") == ("203.0.113.5", None)
