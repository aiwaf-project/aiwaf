"""Route capability detection shared by adapter startup wiring."""

from __future__ import annotations

import re


_UUID_HINT_RE = re.compile(r"(?:<uuid:|uuid|guid)", re.IGNORECASE)


def path_looks_uuid_capable(path: str) -> bool:
    text = str(path or "")
    return bool(_UUID_HINT_RE.search(text))


def detect_uuid_routes_in_flask_app(app) -> bool:
    try:
        for rule in getattr(app, "url_map", {}).iter_rules():
            if path_looks_uuid_capable(getattr(rule, "rule", "")):
                return True
    except Exception:
        return False
    return False


def detect_uuid_routes_in_fastapi_app(app) -> bool:
    try:
        for route in getattr(app, "routes", []):
            path = getattr(route, "path", "") or getattr(route, "path_format", "")
            if path_looks_uuid_capable(path):
                return True
            convertors = getattr(route, "param_convertors", {}) or {}
            for name, conv in convertors.items():
                if path_looks_uuid_capable(str(name)) or path_looks_uuid_capable(conv.__class__.__name__):
                    return True
    except Exception:
        return False
    return False


def detect_uuid_routes_in_django_resolver(resolver) -> bool:
    def _walk(patterns):
        for entry in patterns or []:
            pattern = getattr(entry, "pattern", None)
            route = getattr(pattern, "_route", "") if pattern is not None else ""
            regex = getattr(pattern, "regex", None)
            regex_text = getattr(regex, "pattern", "") if regex is not None else ""
            if path_looks_uuid_capable(route) or path_looks_uuid_capable(regex_text):
                return True
            nested = getattr(entry, "url_patterns", None)
            if nested and _walk(nested):
                return True
        return False

    try:
        return _walk(getattr(resolver, "url_patterns", []))
    except Exception:
        return False
