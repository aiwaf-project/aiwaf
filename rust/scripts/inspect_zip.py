#!/usr/bin/env python3
"""Print ZIP EOCD info and trailing data size for wheels."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

EOCD_SIG = b"\x50\x4b\x05\x06"
EOCD_MIN_SIZE = 22
MAX_COMMENT = 0xFFFF


def inspect(path: Path) -> int:
    data = path.read_bytes()
    size = len(data)
    idx = data.rfind(EOCD_SIG)
    if idx == -1:
        print(f"{path}: EOCD not found")
        return 1
    if size - idx < EOCD_MIN_SIZE:
        print(f"{path}: EOCD truncated")
        return 1
    eocd = data[idx:idx + EOCD_MIN_SIZE]
    _, _, _, _, _, cd_size, cd_offset, comment_len = struct.unpack("<4sHHHHIIH", eocd)
    expected_end = idx + EOCD_MIN_SIZE + comment_len
    tail = size - expected_end
    print(f"{path}: size={size} eocd={idx} expected_end={expected_end} tail={tail} cd_offset={cd_offset} cd_size={cd_size} comment_len={comment_len}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: inspect_zip.py <zip> [<zip>...]", file=sys.stderr)
        return 2

    failed = False
    for arg in argv[1:]:
        p_arg = Path(arg)
        if p_arg.is_absolute():
            if p_arg.is_file():
                failed |= inspect(p_arg) != 0
            continue
        for p in Path().glob(arg):
            if p.is_file():
                failed |= inspect(p) != 0

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
