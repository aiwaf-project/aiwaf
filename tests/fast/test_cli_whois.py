import sys


def test_fast_cli_whois_command(monkeypatch, capsys):
    from aiwaf.fast import cli as fast_cli
    from aiwaf.core import whois as whois_mod

    monkeypatch.setattr(
        whois_mod,
        "run_whois_lookup",
        lambda target: {"domain_name": "EXAMPLE.COM", "target": target},
    )
    monkeypatch.setattr(sys, "argv", ["aiwaf fast", "whois", "example.com"])

    try:
        fast_cli.main()
    except SystemExit:
        pass

    out = capsys.readouterr().out
    assert "WHOIS result" in out

