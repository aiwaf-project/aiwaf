#!/usr/bin/env python3
"""Run zipfile testzip to find first bad CRC entry."""
from __future__ import annotations

import sys
import zipfile
import zlib
from pathlib import Path


def check(path: Path) -> int:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad:
                print(f"{path}: bad CRC for {bad}")
                return 1
            print(f"{path}: CRC OK")
            return 0
    except (zipfile.BadZipFile, zlib.error) as exc:
        print(f"{path}: invalid zip: {exc}")
        return 1


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: wheel_crc_check.py <zip> [<zip>...]", file=sys.stderr)
        return 2

    failed = False
    for arg in argv[1:]:
        p_arg = Path(arg)
        if p_arg.is_absolute():
            if p_arg.is_file():
                failed |= check(p_arg) != 0
            continue
        for p in Path().glob(arg):
            if p.is_file():
                failed |= check(p) != 0

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
