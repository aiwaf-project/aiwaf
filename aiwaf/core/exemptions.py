"""
Shared exemption helpers for paths and path rules.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Iterable


def normalize_path(path: str, trailing_slash: bool | None = None) -> str:
    if not path:
        return "/"
    cleaned = re.sub(r"/{2,}", "/", str(path).strip())
    if not cleaned.startswith("/"):
        cleaned = "/" + cleaned
    if trailing_slash is True and not cleaned.endswith("/"):
        cleaned += "/"
    if trailing_slash is False and cleaned != "/":
        cleaned = cleaned.rstrip("/")
    return cleaned.lower()


def normalize_paths(paths: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for path in paths:
        if not path:
            continue
        normalized.append(normalize_path(path))
    return normalized


def is_path_exempt(path: str, exempt_paths: Iterable[str], allow_wildcards: bool, allow_prefix: bool) -> bool:
    if not path:
        return False
    path_lower = normalize_path(path, trailing_slash=None)
    for exempt in exempt_paths:
        if not exempt:
            continue
        exempt_norm = normalize_path(exempt, trailing_slash=None)
        if allow_wildcards and "*" in exempt_norm:
            if fnmatch.fnmatch(path_lower, exempt_norm):
                return True
            continue
        if path_lower == exempt_norm:
            return True
        if allow_prefix:
            prefix = exempt_norm.rstrip("/")
            if prefix:
                if path_lower == prefix or path_lower.startswith(prefix + "/"):
                    return True
    return False


def get_path_rule_for_path(path: str, rules: Iterable[dict]) -> dict | None:
    if not path:
        return None
    normalized_path = normalize_path(path, trailing_slash=False)
    best = None
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        prefix = normalize_path(rule.get("PREFIX"), trailing_slash=True) if rule.get("PREFIX") else None
        if not prefix:
            continue
        if normalized_path == prefix.rstrip("/") or normalized_path.startswith(prefix):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, rule)
    return best[1] if best else None


def normalize_middleware_name(name) -> str:
    if not name:
        return ""
    if not isinstance(name, str):
        name = getattr(name, "__name__", str(name))
    name = name.strip()
    if "." in name:
        name = name.split(".")[-1]
    return name.lower()
