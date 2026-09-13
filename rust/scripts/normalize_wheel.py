#!/usr/bin/env python3
"""Rewrite wheel ZIPs to strip trailing data and normalize EOCD."""
from __future__ import annotations

import sys
from pathlib import Path
import zipfile
import tempfile
import os
import shutil


def rewrite_wheel(path: Path) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="wheel-normalize-"))
    tmp_zip = tmp_dir / path.name

    try:
        with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(
            tmp_zip, "w", compression=zipfile.ZIP_DEFLATED
        ) as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                new_info = zipfile.ZipInfo(info.filename)
                new_info.date_time = info.date_time
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                new_info.flag_bits = info.flag_bits
                new_info.create_system = info.create_system
                new_info.extra = info.extra
                new_info.comment = info.comment
                dst.writestr(new_info, data)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"{path}: invalid wheel zip: {exc}") from exc

    # Replace original atomically
    shutil.move(str(tmp_zip), str(path))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: normalize_wheel.py <wheel> [<wheel>...]", file=sys.stderr)
        return 2

    failed = False
    for arg in argv[1:]:
        for p in Path().glob(arg):
            if p.is_file():
                try:
                    rewrite_wheel(p)
                except Exception as exc:
                    print(exc, file=sys.stderr)
                    failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
