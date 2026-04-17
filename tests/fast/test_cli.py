import sys


def test_fast_cli_main_dispatches_without_crash(monkeypatch):
    from aiwaf.fast import cli as fast_cli

    monkeypatch.setattr(sys, "argv", ["aiwaf fast", "list", "all"])
    try:
        fast_cli.main()
    except SystemExit:
        pass


def test_fast_cli_module_exports_main():
    from aiwaf.fast import cli as fast_cli

    assert callable(fast_cli.main)
