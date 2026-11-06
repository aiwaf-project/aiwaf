"""Use audit findings on live requests, in log-only mode.

``aiwaf audit --manifest`` saves which operation each route's input reaches. When
``AIWAF_INJECTION_RISK_MODE`` is ``log``, requests to those routes are checked for patterns
that matter to that operation, and matches are logged. This never blocks or changes a
request, and a route with no saved risks keeps its normal checks. Log events carry names
from the manifest and rule names only, never request values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import re
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import parse_qsl, unquote_plus

from .path_manifest import DEFAULT_MANIFEST_PATH, load_manifest, now_utc_iso, write_manifest
from .sql_injection import RULES as SQL_RULES

LOGGER = logging.getLogger("aiwaf.injection_risk")
MODE_SETTING = "AIWAF_INJECTION_RISK_MODE"

_MAX_VALUE_CHARS = 4096
_MAX_ITEMS = 512
_MAX_DEPTH = 8
_NOSQL_OPERATORS = "where|ne|eq|gt|gte|lt|lte|in|nin|regex|exists|or|and|nor|not|expr"

# Keep these in step with js/lib/routeRisks.js and RouteRiskCore.java. The shared cases in
# tests/fixtures/route_risk_cases.json run against all three.
SINK_RULES: dict[str, tuple] = {
    "sql": SQL_RULES,
    "command": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("command_chain", r"[;&|]\s*(?:cat|curl|wget|nc|ncat|bash|sh|zsh|cmd|powershell|rm|id|whoami|uname|ping)\b"),
        ("command_substitution", r"\$\([^)]{0,200}\)|`[^`]{1,200}`"),
    )),
    "path": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("path_traversal", r"(?:^|[/\\])\.\.(?:[/\\]|$)"),
        ("path_sensitive_location", r"^(?:/etc/|/proc/|/root/|/var/log/|[a-z]:[/\\]|\\\\)"),
        ("path_null_byte", r"\x00"),
    )),
    "ssrf": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("ssrf_scheme", r"^\s*(?:file|gopher|dict|ftp|ldap|tftp|netdoc)://"),
        ("ssrf_internal_host", r"^\s*(?:https?:)?//(?:[^/@?#\s]{0,128}@)?(?:localhost|127\.\d|0\.0\.0\.0|10\.\d|192\.168\.|169\.254\.|172\.(?:1[6-9]|2\d|3[01])\.|\[::1\]|metadata\.google\.internal)"),
    )),
    "nosql": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("nosql_operator", rf"^\$(?:{_NOSQL_OPERATORS})$|\[\$(?:{_NOSQL_OPERATORS})\]|\{{\s*\"?\$(?:{_NOSQL_OPERATORS})\b"),
    )),
    "code": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("code_expression", r"__import__|\b(?:eval|exec|compile)\s*\(|\brequire\s*\(|\bprocess\.(?:env|exit|mainModule)|java\.lang\.|Runtime\.getRuntime|\bT\s*\(\s*java"),
    )),
    "template": tuple((name, re.compile(pattern, re.I)) for name, pattern in (
        ("template_expression", r"\{\{.{0,200}\}\}|\{%.{0,200}%\}|\$\{.{0,200}\}|<%.{0,200}%>|#\{.{0,200}\}"),
    )),
    "deserialization": tuple((name, re.compile(pattern)) for name, pattern in (
        ("deserialization_marker", r"^rO0AB|^aced0005|^gASV|!!python/|!!javax\.|^O:\d+:\"|__reduce__|_\$\$ND_FUNC\$\$_"),
    )),
}

_KEY_IN_SOURCE = re.compile(r"\['(.+?)'\]")
_ROUTE_SEGMENT = re.compile(r"^(?:<(?:[^:>]+:)?([^>]+)>|\{\*?([^:}]+)(?::[^}]*)?\}|:([^/?*]+)[?*]?)$")
_WILDCARD_SEGMENT = re.compile(r"^(?:<path:[^>]+>|\{\*?[^:}]+:(?:path|\.\*|\.\+)\}|\{\*[^}]+\}|:[^/?*]+\*|\*\*)$")
_NAME_SUFFIX = re.compile(r"\[.*$")


def rule_for(sink: str, value: str) -> Optional[str]:
    """Return the first rule for this operation that matches the value."""
    text = str(value)[:_MAX_VALUE_CHARS]
    forms = (text, unquote_plus(text)) if "%" in text else (text,)
    for form in forms:
        for name, pattern in SINK_RULES.get(sink, ()):
            if pattern.search(form):
                return name
    return None


def compile_route(path: str) -> tuple[re.Pattern, list[str]]:
    """Compile a manifest route from any supported framework's syntax."""
    pattern, names = "", []
    for segment in str(path).strip("/").split("/"):
        if not segment:
            continue
        if _WILDCARD_SEGMENT.match(segment):
            match = _ROUTE_SEGMENT.match(segment)
            names.append(next((item for item in (match.groups() if match else ()) if item), ""))
            pattern += "/(.*)"
        elif segment == "*":
            names.append("")
            pattern += "/([^/]+)"
        else:
            match = _ROUTE_SEGMENT.match(segment)
            if match:
                names.append(next(item for item in match.groups() if item))
                pattern += "/([^/]+)"
            else:
                pattern += "/" + re.escape(segment)
    return re.compile(f"^{pattern}/?$"), names


def input_name(name: str) -> str:
    return _NAME_SUFFIX.sub("", str(name))


def _walk_json(value: Any, name: str, root: str, items: list, depth: int = 0) -> None:
    # Nested values also count as the top-level field that holds them, as in {"user": {"$ne": 1}}.
    if len(items) >= _MAX_ITEMS or depth > _MAX_DEPTH:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key, top = str(key), root or str(key)
            items.append((key, key))
            if top != key:
                items.append((top, key))
            _walk_json(item, key, top, items, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _walk_json(item, name, root, items, depth + 1)
    elif value is not None:
        items.append((name, str(value)))
        if root and root != name:
            items.append((root, str(value)))


def request_inputs(query: Any, body: Any) -> list[tuple[str, str]]:
    """Return (name, value) pairs from a query string and a buffered body."""
    items: list[tuple[str, str]] = []
    if isinstance(query, bytes):
        query = query.decode("utf-8", errors="ignore")
    for key, value in parse_qsl(str(query or ""), keep_blank_values=True, max_num_fields=None)[:_MAX_ITEMS]:
        items.extend(((input_name(key), key), (input_name(key), value)))
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", errors="ignore")
    if isinstance(body, str) and body:
        stripped = body.lstrip()
        if stripped[:1] in {"{", "["}:
            try:
                _walk_json(json.loads(stripped), "", "", items)
                return items[:_MAX_ITEMS]
            except ValueError:
                items.append(("", body))
        elif "=" in body:
            for key, value in parse_qsl(body, keep_blank_values=True)[:_MAX_ITEMS]:
                items.extend(((input_name(key), key), (input_name(key), value)))
        else:
            items.append(("", body))
    return items[:_MAX_ITEMS]


@dataclass(frozen=True)
class _Route:
    path: str
    pattern: re.Pattern
    names: tuple
    risks: tuple


class RouteRisks:
    """Saved risks for one manifest, compiled for lookup by request path."""

    def __init__(self, manifest: Optional[Mapping[str, Any]]):
        routes = manifest.get("routes") if isinstance(manifest, Mapping) else None
        compiled = []
        for path, entry in (routes.items() if isinstance(routes, Mapping) else ()):
            risks = entry.get("injection_risks") if isinstance(entry, Mapping) else None
            if isinstance(risks, list) and risks:
                pattern, names = compile_route(path)
                compiled.append(_Route(str(path), pattern, tuple(names), tuple(item for item in risks if isinstance(item, Mapping))))
        # Exact routes win over routes with parameters.
        self.routes = sorted(compiled, key=lambda route: (len(route.names), -len(route.path)))

    def inspect(self, path: str, query: Any = None, body: Any = None) -> list[dict[str, str]]:
        for route in self.routes:
            match = route.pattern.match(str(path or "/"))
            if match:
                return self._inspect_route(route, match, query, body)
        return []

    @staticmethod
    def _inspect_route(route: _Route, match: re.Match, query: Any, body: Any) -> list[dict[str, str]]:
        items = request_inputs(query, body)
        items.extend((name, value) for name, value in zip(route.names, match.groups()) if value)
        events, seen = [], set()
        for risk in route.risks:
            sink, wanted = str(risk.get("sink", "")), str(risk.get("input", ""))
            for name, value in items:
                if wanted and name != wanted:
                    continue
                rule = rule_for(sink, value)
                if rule and (sink, wanted, rule) not in seen:
                    seen.add((sink, wanted, rule))
                    # The input name comes from the manifest, because request names are untrusted.
                    events.append({
                        "route": route.path,
                        "sink": sink,
                        "input": wanted,
                        "rule": rule,
                        "confidence": str(risk.get("confidence", "")),
                    })
        return events


_cache: dict[str, RouteRisks] = {}


def clear_cache() -> None:
    _cache.clear()


def observer_for(mode: Any, manifest_path: Any, path: str, method: str = ""):
    """Return a callable taking (query, body), or None unless log mode applies to this route."""
    selected = mode if mode is not None else os.getenv(MODE_SETTING, "off")
    if str(selected).strip().lower() != "log":
        return None
    try:
        location = str(manifest_path or DEFAULT_MANIFEST_PATH)
        risks = _cache.get(location)
        if risks is None:
            risks = _cache[location] = RouteRisks(load_manifest(location))
        if not risks.routes:
            return None
    except Exception:
        return None

    def observe(query: Any, body: Any) -> None:
        # Log-only checks must never affect the request, whatever they receive.
        try:
            for event in risks.inspect(path, query, body):
                LOGGER.warning("%s", json.dumps({"event": "aiwaf.injection_risk", "mode": "log", "method": str(method), **event}, sort_keys=True))
        except Exception:
            LOGGER.debug("Injection risk check failed", exc_info=True)

    return observe


def input_from_source(source: str) -> str:
    """Find the request parameter named in a finding, or return "" when it isn't known."""
    text = str(source or "")
    if text.startswith("route parameter "):
        return text[len("route parameter "):]
    match = _KEY_IN_SOURCE.search(text)
    return match.group(1) if match else ""


def _view_matches(view: str, entry: str, filename: str) -> bool:
    parts = [part for part in str(view).split(".") if part]
    if not parts or not entry:
        return False
    names = entry.split(".")
    if parts[-1] != names[0] and parts[-1] != names[-1]:
        return False
    if len(names) > 1 and parts[-1] == names[-1] and (len(parts) < 2 or parts[-2] != names[0]):
        return False
    modules = [part for part in parts[:-1] if part != names[0]]
    if not modules or modules[-1] == "__main__":
        return True
    stem = re.sub(r"\.py$", "", str(filename).replace("\\", "/")).split("/")
    return stem[-1] == modules[-1] or (stem[-1] == "__init__" and len(stem) > 1 and stem[-2] == modules[-1])


def _routes_for(finding: Mapping[str, Any], routes: Mapping[str, Any]) -> list[str]:
    entry = str(finding.get("function", "")).split(" -> ")[0]
    return [
        path for path, data in routes.items()
        if isinstance(data, Mapping) and _view_matches(str(data.get("view", "")), entry, str(finding.get("file", "")))
    ]


def apply_report(report: Mapping[str, Any], manifest: dict) -> dict[str, int]:
    """Replace saved risks in a manifest with this report's findings, and count the results."""
    routes = manifest.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("The manifest has no routes")
    for data in routes.values():
        if isinstance(data, dict):
            data.pop("injection_risks", None)
    linked, unlinked = 0, 0
    collected: dict[str, dict[tuple, dict[str, str]]] = {}
    for finding in report.get("findings", ()):
        matches = _routes_for(finding, routes)
        linked += bool(matches)
        unlinked += not matches
        for path in matches:
            risk = {"sink": str(finding["sink"]), "input": input_from_source(finding.get("source", "")), "confidence": str(finding["confidence"])}
            collected.setdefault(path, {})[(risk["sink"], risk["input"])] = risk
    for path, risks in collected.items():
        routes[path]["injection_risks"] = [risks[key] for key in sorted(risks)]
    manifest["injection_audit"] = {
        "generated_at": now_utc_iso(),
        "files_scanned": int(report.get("files_scanned", 0)),
        "linked_findings": linked,
        "unlinked_findings": unlinked,
    }
    return {"routes": len(collected), "linked": linked, "unlinked": unlinked}


def update_manifest(report: Mapping[str, Any], manifest_path: str) -> dict[str, int]:
    """Save a report's findings into an existing manifest file."""
    manifest = load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Manifest not found or unreadable: {manifest_path}. Generate it first with init.")
    counts = apply_report(report, manifest)
    write_manifest(manifest, manifest_path)
    return counts


def format_counts(counts: Mapping[str, int], manifest_path: str) -> str:
    text = f"Saved {counts['linked']} possible injection paths to {counts['routes']} routes in {manifest_path}."
    if counts["unlinked"]:
        text += f" {counts['unlinked']} couldn't be tied to a route, so they get no route-specific checks."
    return text
