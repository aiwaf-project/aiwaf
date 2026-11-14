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
