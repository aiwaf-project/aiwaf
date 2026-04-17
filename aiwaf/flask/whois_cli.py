"""Dedicated WHOIS console entrypoint."""

from __future__ import annotations

import sys

from .cli import main as cli_main


def main() -> None:
    argv = [sys.argv[0], "whois"]
    argv.extend(sys.argv[1:])
    sys.argv = argv
    cli_main()


if __name__ == "__main__":
    main()
