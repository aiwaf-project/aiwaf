from __future__ import annotations

import argparse
import sys


def aiwaf_detect() -> None:
    try:
        from aiwaf.django.trainer import train
    except Exception as exc:
        sys.stderr.write(
            "aiwaf-detect requires Django integration. Install with:\n"
            "  pip install aiwaf[django]\n"
        )
        raise SystemExit(1) from exc

    train()


def main() -> None:
    """Top-level AIWAF CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="aiwaf",
        description="AIWAF CLI",
    )
    parser.add_argument(
        "framework",
        choices=["django", "flask", "fast"],
        help="Framework integration CLI to run",
    )
    parser.add_argument(
        "framework_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to framework CLI",
    )
    args = parser.parse_args()

    if args.framework == "flask":
        from aiwaf.flask.cli import main as flask_main

        sys.argv = ["aiwaf flask"] + list(args.framework_args)
        flask_main()
        return
    if args.framework == "fast":
        from aiwaf.fast.cli import main as fast_main

        sys.argv = ["aiwaf fast"] + list(args.framework_args)
        fast_main()
        return
    if args.framework == "django":
        try:
            from django.core.management import execute_from_command_line
        except Exception as exc:
            sys.stderr.write(
                "aiwaf django requires Django integration. Install with:\n"
                "  pip install aiwaf[django]\n"
            )
            raise SystemExit(1) from exc

        if not args.framework_args:
            sys.stderr.write(
                "Usage: aiwaf django <management-command> [args]\n"
                "Example: aiwaf django aiwaf_list --all\n"
            )
            raise SystemExit(2)

        execute_from_command_line(["aiwaf django"] + list(args.framework_args))
        return


if __name__ == "__main__":
    main()
