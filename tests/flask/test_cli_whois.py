import sys


def test_cli_whois_command(monkeypatch, capsys):
    from aiwaf.flask.cli import main
    from aiwaf.flask import aiwaf_whois

    monkeypatch.setattr(aiwaf_whois, "run_whois_lookup", lambda target: {"domain_name": "EXAMPLE.COM", "target": target})
    monkeypatch.setattr(sys, "argv", ["aiwaf_console.py", "whois", "example.com"])

    main()
    out = capsys.readouterr().out
    assert "WHOIS result:" in out
    assert "EXAMPLE.COM" in out


def test_cli_whois_missing_dependency(monkeypatch, capsys):
    from aiwaf.flask.cli import main
    from aiwaf.flask import aiwaf_whois

    def _raise_module_not_found(_target):
        raise ModuleNotFoundError("No module named 'whois'")

    monkeypatch.setattr(aiwaf_whois, "run_whois_lookup", _raise_module_not_found)
    monkeypatch.setattr(sys, "argv", ["aiwaf_console.py", "whois", "example.com"])

    main()
    out = capsys.readouterr().out
    assert "python-whois is not installed" in out
