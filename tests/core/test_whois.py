import sys
from types import SimpleNamespace

from aiwaf.core.whois import run_whois_lookup


def test_whois_module_delegates_resolved_domain(monkeypatch):
    lookup = lambda domain: {"domain": domain}
    monkeypatch.setitem(sys.modules, "whois", SimpleNamespace(whois=lookup))
    assert run_whois_lookup("example.com") == {"domain": "example.com"}
