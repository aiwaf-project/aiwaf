"""Property-based tests for utility and middleware invariants."""

import importlib.util

import pytest

from aiwaf.fast.middleware.header_validation import HeaderValidationMiddleware
from aiwaf.core.runtime_utils import is_private_ip, sanitize_header_value


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, strategies as st


@pytest.mark.slow
def _middleware():
    from fastapi import FastAPI

    return HeaderValidationMiddleware(FastAPI())


@given(st.text(min_size=0, max_size=5000), st.integers(min_value=1, max_value=2000))
def test_sanitize_header_value_never_exceeds_expected_bound(value, max_len):
    out = sanitize_header_value(value, max_length=max_len)
    assert len(out) <= max_len + 3


@given(st.text(min_size=1, max_size=120))
def test_is_private_ip_never_raises(ip_text):
    result = is_private_ip(ip_text)
    assert isinstance(result, bool)


@given(
    st.booleans(),
    st.booleans(),
    st.booleans(),
    st.booleans(),
    st.booleans(),
)
def test_header_quality_score_stays_in_reasonable_range(has_ua, has_accept, has_lang, has_enc, keep_alive):
    mw = _middleware()
    headers = {}
    if has_ua:
        headers["user-agent"] = "Mozilla/5.0"
    if has_accept:
        headers["accept"] = "text/html"
    if has_lang:
        headers["accept-language"] = "en-US"
    if has_enc:
        headers["accept-encoding"] = "gzip"
    if keep_alive:
        headers["connection"] = "keep-alive"

    score = mw._calculate_header_quality(headers)
    assert 0 <= score <= 12
