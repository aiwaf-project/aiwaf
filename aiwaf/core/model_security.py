"""Helpers for guarding model artifact loading."""

from __future__ import annotations

from pathlib import Path


def is_trusted_model_path(path: str | None, *, default_path: str | None = None, allow_custom: bool = False) -> bool:
    """Return True only for the built-in model path unless custom loading is explicitly allowed."""
    if not path:
        return False
    if allow_custom:
        return True

    candidate = Path(path).expanduser()
    if default_path:
        default_candidate = Path(default_path).expanduser()
        try:
            return candidate.resolve(strict=False) == default_candidate.resolve(strict=False)
        except Exception:
            return False

    return False