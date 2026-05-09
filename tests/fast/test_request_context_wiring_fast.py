from types import SimpleNamespace
from unittest.mock import patch

from aiwaf.core.runtime_utils import get_ip


class DummyRequest:
    def __init__(self, headers=None, client_ip=None):
        self.headers = headers or {}
        self.client = SimpleNamespace(host=client_ip) if client_ip else None


def test_fast_get_ip_calls_core_extractor_before_fallback_logic():
    req = DummyRequest(headers={"x-forwarded-for": "203.0.113.3"}, client_ip="8.8.8.8")
    with patch("aiwaf.core.runtime_utils.resolve_ip_from_fastapi_request", return_value="8.8.8.8"):
        assert get_ip(req) == "8.8.8.8"


def test_fast_get_ip_falls_back_when_core_extractor_unknown():
    req = DummyRequest(headers={"x-forwarded-for": "203.0.113.44"}, client_ip="127.0.0.1")
    with patch("aiwaf.core.runtime_utils.resolve_ip_from_fastapi_request", return_value="203.0.113.44"):
        assert get_ip(req) == "203.0.113.44"
