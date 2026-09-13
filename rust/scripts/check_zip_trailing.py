#!/usr/bin/env python3
"""Fail if ZIP files contain trailing data after EOCD record."""
from __future__ import annotations

import os
import struct
import sys
from pathlib import Path

EOCD_SIG = b"\x50\x4b\x05\x06"  # End of central directory
EOCD_MIN_SIZE = 22
MAX_COMMENT = 0xFFFF


def find_eocd(data: bytes) -> int:
    # Search backwards for EOCD signature
    idx = data.rfind(EOCD_SIG)
    if idx == -1:
        return -1
    return idx


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    size = path.stat().st_size
    with path.open("rb") as f:
        read_size = min(size, EOCD_MIN_SIZE + MAX_COMMENT)
        f.seek(size - read_size)
        tail = f.read(read_size)

    eocd_off_tail = find_eocd(tail)
    if eocd_off_tail == -1:
        return [f"{path}: EOCD not found"]

    eocd_off = (size - read_size) + eocd_off_tail

    if size - eocd_off < EOCD_MIN_SIZE:
        return [f"{path}: EOCD truncated"]

    # Parse EOCD to get comment length
    # struct: signature(4), disk(2), disk_cd(2), entries_disk(2), entries(2),
    # cd_size(4), cd_offset(4), comment_len(2)
    with path.open("rb") as f:
        f.seek(eocd_off)
        eocd = f.read(EOCD_MIN_SIZE)

    try:
        _, _, _, _, _, cd_size, cd_offset, comment_len = struct.unpack("<4sHHHHIIH", eocd)
    except struct.error:
        return [f"{path}: EOCD unpack failed"]

    expected_end = eocd_off + EOCD_MIN_SIZE + comment_len
    if expected_end != size:
        errors.append(
            f"{path}: trailing data detected (expected end {expected_end}, size {size})"
        )

    # Basic sanity: central directory should be within file bounds
    if cd_offset + cd_size > eocd_off:
        errors.append(
            f"{path}: central directory overlaps EOCD or trailing data"
        )

    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: check_zip_trailing.py <zip> [<zip>...]", file=sys.stderr)
        return 2

    failures: list[str] = []
    for arg in argv[1:]:
        p_arg = Path(arg)
        if p_arg.is_absolute():
            if p_arg.is_file():
                failures.extend(check_file(p_arg))
            continue
        for p in Path().glob(arg):
            if p.is_file():
                failures.extend(check_file(p))

    if failures:
        for msg in failures:
            print(msg, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
