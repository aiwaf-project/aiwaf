#!/usr/bin/env python3
"""
AIWAF FastAPI CLI.

For now, FastAPI management commands mirror the Flask CLI command set so users
can manage storage, logs, and training workflows with:
    aiwaf fast <command> ...
"""

from aiwaf.flask.cli import main as _flask_cli_main


def main() -> None:
    _flask_cli_main()


if __name__ == "__main__":
    main()
