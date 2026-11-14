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
from .sql_injection import Finding, _as_bytes, rust

LOGGER = logging.getLogger("aiwaf.injection_risk")
MODE_SETTING = "AIWAF_INJECTION_RISK_MODE"

# The operations a saved finding can name. The rules for each run in aiwaf-rust.
SINKS = ("sql", "nosql", "command", "path", "ssrf", "code", "template", "deserialization")

_KEY_IN_SOURCE = re.compile(r"\['(.+?)'\]")


def rule_for(sink: str, value: str) -> Optional[str]:
    """Return the first rule for this operation that matches the value."""
    return rust().policy_route_rule(str(sink), str(value))


class RouteRisks:
    """Saved risks for one manifest, compiled in aiwaf-rust for lookup by request path."""

    def __init__(self, manifest: Optional[Mapping[str, Any]]):
        self._checker = rust().RouteRiskChecker(json.dumps(manifest if isinstance(manifest, Mapping) else None))

    @property
    def empty(self) -> bool:
        return self._checker.empty

    def inspect(self, path: str, query: Any = None, body: Any = None) -> list[dict[str, str]]:
        if isinstance(query, (bytes, bytearray)):
            query = bytes(query).decode("utf-8", errors="replace")
        events = self._checker.inspect(str(path or "/"), str(query or ""), _as_bytes(body))
        return [dict(zip(("route", "sink", "input", "rule", "confidence"), event)) for event in events]


_cache: dict[str, RouteRisks] = {}


def clear_cache() -> None:
    _cache.clear()


MODES = ("block", "log", "off")


def observer_for(mode: Any, manifest_path: Any, path: str, method: str = ""):
    """Return a callable taking (query, body), or None when there's nothing to check.

    The callable returns a blocking Finding in block mode, and None otherwise.
    """
    selected = str(mode if mode is not None else os.getenv(MODE_SETTING, "block")).strip().lower()
    if selected not in MODES:
        raise ValueError("AIWAF_INJECTION_RISK_MODE must be block, log, or off")
    if selected == "off":
        return None
    location = str(manifest_path or DEFAULT_MANIFEST_PATH)
    risks = _cache.get(location)
    if risks is None:
        risks = _cache[location] = RouteRisks(load_manifest(location))
    if risks.empty:
        return None

    def observe(query: Any, body: Any) -> Optional[Finding]:
        try:
            events = risks.inspect(path, query, body)
            for event in events:
                LOGGER.warning("%s", json.dumps({"event": "aiwaf.injection_risk", "mode": selected, "method": str(method), **event}, sort_keys=True))
        except Exception:
            # A failing check must not break the request. Other protections still apply.
            LOGGER.debug("Injection risk check failed", exc_info=True)
            return None
        return Finding(events[0]["rule"]) if events and selected == "block" else None

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


def audit_into_manifest(manifest: dict, source: Optional[str]) -> Optional[dict[str, int]]:
    """Scan source and save its findings into a manifest that's about to be written.

    Returns the counts, or None when source is None or the scan fails. A failed scan leaves the
    manifest without saved risks, so routes keep their normal checks.
    """
    if source is None:
        return None
    try:
        from .injection_analysis import analyze_path

        return apply_report(analyze_path(str(source)), manifest)
    except Exception as error:  # noqa: BLE001 - a scan problem must not stop init
        LOGGER.warning("Injection audit skipped: %s", error)
        return None


def describe_audit(manifest: Mapping[str, Any]) -> str:
    """Describe a manifest's saved audit for init's summary."""
    audit = manifest.get("injection_audit")
    if not isinstance(audit, Mapping):
        return "Injection checks: not run"
    routes = sum(1 for entry in manifest.get("routes", {}).values() if isinstance(entry, Mapping) and entry.get("injection_risks"))
    text = f"Injection checks: {routes} routes from {audit.get('files_scanned', 0)} files"
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
