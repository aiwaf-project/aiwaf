"""Flask route extraction for AIWAF path manifests."""

from __future__ import annotations

from typing import Any

from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest


def _view_name(view_func: Any) -> str:
    if view_func is None:
        return ""
    return f"{getattr(view_func, '__module__', '')}.{getattr(view_func, '__name__', view_func.__class__.__name__)}".strip(".")


def extract_flask_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view_func = app.view_functions.get(rule.endpoint)
        metadata = {
            "response_type": "json" if str(rule.rule).startswith("/api/") else None,
            "blueprint": rule.endpoint.rsplit(".", 1)[0] if "." in rule.endpoint else "",
        }
        path, entry = build_route_entry(
            path=rule.rule,
            methods=sorted(rule.methods or []),
            view=_view_name(view_func),
            name=rule.endpoint,
            metadata=metadata,
        )
        if metadata["blueprint"]:
            entry["blueprint"] = metadata["blueprint"]
        routes[path] = entry
    return routes


def generate_flask_manifest(app, output_path: str = ".aiwaf/paths.json") -> dict[str, Any]:
    manifest = build_manifest(framework="flask", routes=extract_flask_routes(app))
    write_manifest(manifest, output_path)
    return manifest
