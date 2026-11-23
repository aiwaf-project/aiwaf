"""Use audit findings to check the inputs of risky routes on live requests.

``init`` and ``audit --manifest`` save which operation each route's input reaches. Requests to
those routes are then checked for patterns that matter to that operation. Matches are logged
and, with ``AIWAF_INJECTION_RISK_MODE=block`` (the default), the request is blocked. ``log``
only logs, and ``off`` turns this off. A route with no saved risks keeps its normal checks.
Log events carry names from the manifest and rule names only, never request values.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Mapping, Optional

from .path_manifest import DEFAULT_MANIFEST_PATH, load_manifest, now_utc_iso, write_manifest
from .input_schema import merge_into
from . import input_sources
from .sql_injection import Finding, _as_bytes, rust

LOGGER = logging.getLogger("aiwaf.injection_risk")
SCHEMA_LOGGER = logging.getLogger("aiwaf.route_schema")
MODE_SETTING = "AIWAF_INJECTION_RISK_MODE"
# What the inputs init saved for each route do: enforce blocks a typed input sent with a value
# of another type and lets through findings the schema clears; monitor only logs both.
SCHEMA_SETTING = "AIWAF_ROUTE_SCHEMA_MODE"
SCHEMA_MODES = ("enforce", "monitor", "off")
# Strict inputs, off by default: on a route whose saved input list is complete, a query or body
# name the route never reads is unexpected, logged in monitor mode and blocked in enforce mode.
# AIWAF_STRICT_INPUTS_ALLOW adds names to always allow (a trailing * matches any rest) to the
# engine's own: CSRF tokens, utm_*, fbclid, gclid, msclkid and _.
STRICT_SETTING = "AIWAF_STRICT_INPUTS"
STRICT_ALLOW_SETTING = "AIWAF_STRICT_INPUTS_ALLOW"
# Path prefixes strict inputs skips, as /webhooks/ or /legacy/*.
STRICT_EXEMPT_SETTING = "AIWAF_STRICT_INPUTS_EXEMPT"
STRICT_WARNING = ("Strict inputs need aiwaf-rust 0.2.16 or newer, so they aren't checked. "
                  "Run: pip install -U aiwaf-rust")
STRICT_EXEMPT_WARNING = ("AIWAF_STRICT_INPUTS_EXEMPT needs aiwaf-rust 0.2.17 or newer, so strict inputs aren't checked "
                         "rather than checked on exempt paths. Run: pip install -U aiwaf-rust")


def _names(value: Any, setting: str) -> list[str]:
    listed = value if value is not None else os.getenv(setting, "")
    names = listed if isinstance(listed, (list, tuple, set)) else str(listed).split(",")
    return [str(name).strip() for name in names if str(name).strip()]


def strict_inputs(value: Any = None, allow: Any = None, exempt: Any = None) -> Optional[dict]:
    """{"allow": [...], "exempt": [...]} when strict inputs is on (the setting, or
    AIWAF_STRICT_INPUTS), else None."""
    raw = value if value is not None else os.getenv(STRICT_SETTING, "")
    if not (raw is True or str(raw).strip().lower() in {"1", "true", "yes", "on"}):
        return None
    return {"allow": _names(allow, STRICT_ALLOW_SETTING), "exempt": _names(exempt, STRICT_EXEMPT_SETTING)}

# The operations a saved finding can name. The rules for each run in aiwaf-rust.
SINKS = (
    "sql", "nosql", "command", "path", "ssrf", "code", "template", "deserialization", "xss", "redirect", "ldap", "xpath",
    "xxe", "redos", "prototype",
)

_KEY_IN_SOURCE = re.compile(r"\['(.+?)'\]")


def rule_for(sink: str, value: str) -> Optional[str]:
    """Return the first rule for this operation that matches the value."""
    return rust().policy_route_rule(str(sink), str(value))


class RouteRisks:
    """Saved risks for one manifest, compiled in aiwaf-rust for lookup by request path."""

    def __init__(self, manifest: Optional[Mapping[str, Any]], strict: Optional[dict] = None):
        # Strict inputs ride in the manifest the engine gets.
        given = {**manifest, "strict_inputs": strict} if strict and isinstance(manifest, Mapping) else manifest
        self._checker = rust().RouteRiskChecker(json.dumps(given if isinstance(given, Mapping) else None))
        if strict and strict.get("exempt") and not getattr(self._checker, "strict_exempt", False):
            # An older engine would check the exempt paths too: better none at all.
            SCHEMA_LOGGER.warning(STRICT_EXEMPT_WARNING)
            self._checker = rust().RouteRiskChecker(json.dumps(manifest if isinstance(manifest, Mapping) else None))
        elif strict and not getattr(self._checker, "strict", False):
            SCHEMA_LOGGER.warning(STRICT_WARNING)
        routes = manifest.get("routes") if isinstance(manifest, Mapping) else None
        self.has_inputs = isinstance(routes, Mapping) and any(
            isinstance(entry, Mapping) and isinstance(entry.get("inputs"), Mapping) for entry in routes.values())

    @property
    def empty(self) -> bool:
        return self._checker.empty

    def inspect(self, path: str, query: Any = None, body: Any = None) -> list[dict[str, str]]:
        if isinstance(query, (bytes, bytearray)):
            query = bytes(query).decode("utf-8", errors="replace")
        events = self._checker.inspect(str(path or "/"), str(query or ""), _as_bytes(body))
        return [dict(zip(("route", "sink", "input", "rule", "confidence"), event)) for event in events]

    def review(self, path: str, query: Any, body: Any, max_bytes: int, found: Optional[str], method: str = "") -> Optional[tuple]:
        """("mismatch", route, input, type), ("unexpected", route, input, ""), ("relax", route, rule,
        ""), or None. See aiwaf-rust. With method, strict inputs check that method's own list."""
        if isinstance(query, (bytes, bytearray)):
            query = bytes(query).decode("utf-8", errors="replace")
        by_method = getattr(self._checker, "review_method", None)
        if method and by_method is not None:
            return by_method(str(method), str(path or "/"), str(query or ""), _as_bytes(body), int(max_bytes), found)
        return self._checker.review(str(path or "/"), str(query or ""), _as_bytes(body), int(max_bytes), found)

    def review_path(self, path: str, max_bytes: int, found: str) -> Optional[tuple]:
        """A finding from the URL path, reviewed against the route's parameters, as review does."""
        return self._checker.review_path(str(path or "/"), int(max_bytes), str(found))


_cache: dict[str, RouteRisks] = {}


def clear_cache() -> None:
    _cache.clear()


MODES = ("block", "log", "off")


def observer_for(mode: Any, manifest_path: Any, path: str, method: str = "", schema_mode: Any = None,
                 strict: Any = None, strict_allow: Any = None, strict_exempt: Any = None):
    """Return a callable taking (query, body), or None when there's nothing to check.

    The callable returns a blocking Finding in block mode, and None otherwise. Its ``review``
    attribute, when set, takes (query, body, finding, max_bytes) after the general check and
    returns the finding to use: see ``AIWAF_ROUTE_SCHEMA_MODE``.
    """
    selected = str(mode if mode is not None else os.getenv(MODE_SETTING, "block")).strip().lower()
    if selected not in MODES:
        raise ValueError("AIWAF_INJECTION_RISK_MODE must be block, log, or off")
    schema = str(schema_mode if schema_mode is not None else os.getenv(SCHEMA_SETTING, "monitor")).strip().lower()
    if schema not in SCHEMA_MODES:
        raise ValueError("AIWAF_ROUTE_SCHEMA_MODE must be enforce, monitor, or off")
    if selected == "off" and schema == "off":
        return None
    location = str(manifest_path or DEFAULT_MANIFEST_PATH)
    strictness = strict_inputs(strict, strict_allow, strict_exempt) if schema != "off" else None
    key = (location, json.dumps(strictness, sort_keys=True))
    risks = _cache.get(key)
    if risks is None:
        risks = _cache[key] = RouteRisks(load_manifest(location), strictness)
    if risks.empty or (selected == "off" and not risks.has_inputs):
        return None
    if not risks.has_inputs:
        schema = "off"

    def observe(query: Any, body: Any) -> Optional[Finding]:
        if selected == "off":
            return None
        try:
            events = risks.inspect(path, query, body)
            for event in events:
                LOGGER.warning("%s", json.dumps({"event": "aiwaf.injection_risk", "mode": selected, "method": str(method), **event}, sort_keys=True))
        except Exception:
            # A failing check must not break the request. Other protections still apply.
            LOGGER.debug("Injection risk check failed", exc_info=True)
            return None
        return Finding(events[0]["rule"]) if events and selected == "block" else None

    def review(query: Any, body: Any, finding: Optional[Finding], max_bytes: int) -> Optional[Finding]:
        found = finding.rule if finding is not None and finding.status == 403 else None
        return decide(lambda: risks.review(path, query, body, max_bytes, found, str(method or "")), finding)

    def review_path(request_path: str, finding: Finding, max_bytes: int) -> Optional[Finding]:
        if finding.status != 403:
            return finding
        return decide(lambda: risks.review_path(request_path, max_bytes, finding.rule), finding)

    def decide(run, finding: Optional[Finding]) -> Optional[Finding]:
        try:
            result = run()
        except Exception:
            SCHEMA_LOGGER.debug("Route schema review failed", exc_info=True)
            return finding
        if result is None:
            return finding
        action, route, name, kind = result
        event = {"event": "aiwaf.route_schema", "mode": schema, "method": str(method), "route": route, "action": action}
        if action == "mismatch":
            event.update({"input": name, "type": kind})
        elif action == "unexpected":
            event["input"] = name
        else:
            event["rule"] = name
        SCHEMA_LOGGER.warning("%s", json.dumps(event, sort_keys=True))
        if schema != "enforce":
            return finding
        if action == "mismatch":
            return Finding("schema_type_mismatch")
        return Finding("schema_unexpected_input") if action == "unexpected" else None

    if schema != "off":
        observe.review = review
        observe.review_path = review_path
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


def _views(data: Mapping[str, Any]) -> list[str]:
    """Every view serving a route: its views when several share the path, else its view."""
    views = data.get("views")
    return [str(view) for view in views] if isinstance(views, list) and views else [str(data.get("view", ""))]


def _matching_views(finding: Mapping[str, Any], data: Mapping[str, Any]) -> list[str]:
    entry = str(finding.get("function", "")).split(" -> ")[0]
    filename = str(finding.get("entry_file") or finding.get("file", ""))
    return [view for view in _views(data) if _view_matches(view, entry, filename)]


def _routes_for(finding: Mapping[str, Any], routes: Mapping[str, Any]) -> list[str]:
    return [path for path, data in routes.items() if isinstance(data, Mapping) and _matching_views(finding, data)]


def apply_report(report: Mapping[str, Any], manifest: dict, sources: Optional[Mapping[str, Any]] = None) -> dict[str, int]:
    """Replace saved risks in a manifest with this report's findings, and count the results.

    sources, from input_sources.sources_from_environment, adds declared, OpenAPI and recorded
    inputs to what the report's code reading found.
    """
    routes = manifest.get("routes")
    if not isinstance(routes, dict):
        raise ValueError("The manifest has no routes")
    for data in routes.values():
        if isinstance(data, dict):
            for key in ("injection_risks", "inputs", "inputs_complete", "inputs_incomplete", "inputs_by_method", "inputs_nested"):
                data.pop(key, None)
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
    # What each route reads, with types. A route whose views weren't all found gets none.
    typed = 0
    read: dict[str, set] = {}
    claims: dict[str, dict[str, set]] = {}
    for entry in report.get("inputs", ()):
        for path in _routes_for(entry, routes):
            merge_into(routes, path, entry)
            matched = _matching_views(entry, routes[path])
            read.setdefault(path, set()).update(matched)
            # The names each method's views read, so strict inputs can check each method's own.
            view_methods = routes[path].get("view_methods") or {}
            for view in matched:
                for method in view_methods.get(view, routes[path].get("methods") or ()):
                    claims.setdefault(path, {}).setdefault(str(method).upper(), set()).update(entry["inputs"])
    for path, methods in claims.items():
        routes[path]["_input_methods"] = {method: sorted(names) for method, names in methods.items()}
    # A path several views serve is only complete when every one's code was read.
    for path, seen in read.items():
        data = routes[path]
        for view in sorted(set(_views(data)) - seen):
            data["inputs_complete"] = False
            reasons = data.setdefault("inputs_incomplete", [])
            reason = {"code": "unread_view", "detail": f"{view} also serves this path, and its code wasn't found"}
            if reason not in reasons:
                reasons.append(reason)
    completeness = input_sources.finalize(routes, **dict(sources or {}))
    for data in routes.values():
        if isinstance(data, dict) and "inputs" in data:
            typed += 1
    manifest["injection_audit"] = {
        "generated_at": now_utc_iso(),
        "files_scanned": int(report.get("files_scanned", 0)),
        "linked_findings": linked,
        "unlinked_findings": unlinked,
    }
    manifest["injection_audit"]["routes_with_inputs"] = typed
    manifest["injection_audit"]["routes_with_complete_inputs"] = completeness["complete"]
    if completeness["reasons"]:
        manifest["injection_audit"]["incomplete_input_reasons"] = dict(sorted(completeness["reasons"].items()))
    return {"routes": len(collected), "linked": linked, "unlinked": unlinked, "inputs": typed}


def audit_into_manifest(manifest: dict, source: Optional[str]) -> Optional[dict[str, int]]:
    """Scan source and save its findings into a manifest that's about to be written.

    Returns the counts, or None when source is None or the scan fails. A failed scan leaves the
    manifest without saved risks, so routes keep their normal checks.
    """
    try:
        if source is None:
            return None
        try:
            from .injection_analysis import analyze_path

            try:
                sources = input_sources.sources_from_environment()
            except (OSError, ValueError) as error:
                LOGGER.warning("Declared, OpenAPI or recorded inputs skipped: %s", error)
                sources = {}
            return apply_report(analyze_path(str(source)), manifest, sources)
        except Exception as error:  # noqa: BLE001 - a scan problem must not stop init
            LOGGER.warning("Injection audit skipped: %s", error)
            return None
    finally:
        # What the framework declared is only for combining with the audit; never saved alone.
        for data in (manifest.get("routes") or {}).values():
            if isinstance(data, dict):
                data.pop("declared_inputs", None)
                data.pop("input_declaration", None)


def describe_audit(manifest: Mapping[str, Any]) -> str:
    """Describe a manifest's saved audit for init's summary."""
    audit = manifest.get("injection_audit")
    if not isinstance(audit, Mapping):
        return "Injection checks: not run"
    routes = sum(1 for entry in manifest.get("routes", {}).values() if isinstance(entry, Mapping) and entry.get("injection_risks"))
    text = f"Injection checks: {routes} routes from {audit.get('files_scanned', 0)} files"
    if audit.get("routes_with_inputs"):
        text += f"; inputs and types for {audit['routes_with_inputs']} routes"
        text += f", complete for {audit.get('routes_with_complete_inputs', 0)}"
        reasons = audit.get("incomplete_input_reasons") or {}
        if reasons:
            top = sorted(reasons.items(), key=lambda item: (-item[1], item[0]))[:4]
            text += " (incomplete because of " + ", ".join(f"{code} {count}" for code, count in top) + ")"
    if audit.get("unlinked_findings"):
        text += f" ({audit['unlinked_findings']} findings couldn't be tied to a route)"
    return text


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
