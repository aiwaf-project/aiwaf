"""Path manifest generation and compilation helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .exemptions import normalize_path

SCHEMA_VERSION = "1.0"
DEFAULT_MANIFEST_PATH = ".aiwaf/paths.json"

MIDDLEWARE_NAMES = {
    "geo_block",
    "ip_keyword_block",
    "rate_limit",
    "ai_anomaly",
    "honeypot",
    "uuid_tamper",
    "header_validation",
    "logging",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def compute_context_hash(routes: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(stable_json(routes).encode("utf-8")).hexdigest()


def classify_route(path: str, *, methods: Iterable[str] | None = None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_path(path, trailing_slash=None)
    path_lower = normalized.lower()
    methods_set = {str(method).upper() for method in (methods or []) if method}
    metadata = metadata or {}

    category = "unknown"
    response_type = "html"
    auth_required = bool(metadata.get("auth_required", False))
    protections: dict[str, Any] = {
        "rate_limit": {"requests": 60, "window_seconds": 60},
        "header_validation": {"enabled": True},
        "ip_keyword_block": {"enabled": True},
        "ai_anomaly": {"enabled": True},
    }

    if path_lower.startswith(("/static/", "/media/", "/assets/")) or any(
        path_lower.endswith(ext) for ext in (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2")
    ):
        category = "static"
        protections.update({
            "header_validation": {"enabled": False},
            "ai_anomaly": {"enabled": False},
            "honeypot": {"enabled": False},
        })
    elif any(part in path_lower for part in ("/admin/", "/admin")):
        category = "admin"
        auth_required = True
        protections["rate_limit"] = {"requests": 30, "window_seconds": 60}
    elif any(part in path_lower for part in ("/login", "/signin", "/auth/login")):
        category = "auth"
        protections["rate_limit"] = {"requests": 30, "window_seconds": 60}
        protections["honeypot"] = {"enabled": True}
    elif path_lower.startswith("/api/") or metadata.get("response_type") == "json":
        category = "api"
        response_type = "json"
        protections["rate_limit"] = {"requests": 120, "window_seconds": 60}
        protections["honeypot"] = {"enabled": False}
    elif any(token in path_lower for token in ("/upload", "/uploads", "/files")):
        category = "upload"
        protections["rate_limit"] = {"requests": 20, "window_seconds": 60}
        protections["payload_validation"] = {"max_body_bytes": 1048576}

    if "POST" in methods_set and category not in {"api", "upload"}:
        protections["rate_limit"] = {"requests": 30, "window_seconds": 60}

    return {
        "category": category,
        "response_type": str(metadata.get("response_type") or response_type),
        "auth_required": auth_required,
        "protections": protections,
    }


def build_route_entry(
    *,
    path: str,
    methods: Iterable[str] | None = None,
    view: str = "",
    name: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    normalized = normalize_path(path, trailing_slash=None)
    methods_list = sorted({str(method).upper() for method in (methods or []) if method and str(method).upper() not in {"HEAD", "OPTIONS"}})
    classified = classify_route(normalized, methods=methods_list, metadata=metadata)
    entry = {
        "methods": methods_list,
        "view": str(view or ""),
        "name": str(name or ""),
        **classified,
    }
    return normalized, entry


def build_manifest(*, framework: str, routes: Mapping[str, Any], app_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized_routes = {
        normalize_path(path, trailing_slash=None): dict(data)
        for path, data in sorted(routes.items(), key=lambda item: normalize_path(item[0], trailing_slash=None))
    }
    context = {
        "framework": framework,
        "routes": normalized_routes,
        "app_context": dict(app_context or {}),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "framework": framework,
        "context_hash": compute_context_hash(context),
        "generated_at": now_utc_iso(),
        "routes": normalized_routes,
    }


def write_manifest(manifest: Mapping[str, Any], output_path: str | Path = DEFAULT_MANIFEST_PATH) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any] | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _rate_limit_override(config: Mapping[str, Any]) -> dict[str, Any]:
    override: dict[str, Any] = {}
    if "WINDOW" in config:
        override["WINDOW"] = config["WINDOW"]
    if "MAX" in config:
        override["MAX"] = config["MAX"]
    if "FLOOD" in config:
        override["FLOOD"] = config["FLOOD"]
    if "window_seconds" in config:
        override["WINDOW"] = config["window_seconds"]
    if "requests" in config:
        override["MAX"] = config["requests"]
    if "flood" in config:
        override["FLOOD"] = config["flood"]
    return override


def compile_manifest_to_path_rules(manifest: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return []
    routes = manifest.get("routes")
    if not isinstance(routes, Mapping):
        return []

    rules: list[dict[str, Any]] = []
    for path, entry in routes.items():
        if not isinstance(entry, Mapping):
            continue
        protections = entry.get("protections", {})
        if not isinstance(protections, Mapping):
            protections = {}
        rule: dict[str, Any] = {"PREFIX": normalize_path(str(path), trailing_slash=True)}
        disabled: list[str] = []
        for name, config in protections.items():
            normalized_name = str(name).strip().lower()
            if normalized_name not in MIDDLEWARE_NAMES:
                continue
            if isinstance(config, Mapping) and config.get("enabled") is False:
                disabled.append(normalized_name)
            elif config is False:
                disabled.append(normalized_name)
        rate_cfg = protections.get("rate_limit") or protections.get("api_rate_limit")
        if isinstance(rate_cfg, Mapping):
            override = _rate_limit_override(rate_cfg)
            if override:
                rule["RATE_LIMIT"] = override
        if disabled:
            rule["DISABLE"] = sorted(set(disabled))
        if "DISABLE" in rule or "RATE_LIMIT" in rule:
            rules.append(rule)
    return rules


def get_effective_path_rules(
    explicit_rules: Iterable[dict] | None = None,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> list[dict]:
    manifest_rules = compile_manifest_to_path_rules(load_manifest(manifest_path))
    explicit = list(explicit_rules or [])
    return explicit + manifest_rules
