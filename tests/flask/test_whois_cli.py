import sys

from aiwaf.flask import whois_cli


def test_whois_entrypoint_inserts_subcommand(monkeypatch):
    calls = []
    monkeypatch.setattr(whois_cli, "cli_main", lambda: calls.append(list(sys.argv)))
    monkeypatch.setattr(sys, "argv", ["aiwaf-whois", "example.com"])
    whois_cli.main()
    assert calls == [["aiwaf-whois", "whois", "example.com"]]
