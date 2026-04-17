from __future__ import annotations

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

