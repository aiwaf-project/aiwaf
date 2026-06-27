"""FastAPI route extraction for AIWAF path manifests."""

from __future__ import annotations

from typing import Any

from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest


def _view_name(endpoint: Any) -> str:
    if endpoint is None:
        return ""
    return f"{getattr(endpoint, '__module__', '')}.{getattr(endpoint, '__name__', endpoint.__class__.__name__)}".strip(".")


def _auth_required(route: Any) -> bool:
    dependant = getattr(route, "dependant", None)
    dependencies = getattr(dependant, "dependencies", None)
    return bool(dependencies)


def extract_fastapi_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for route in getattr(app, "routes", []):
        path_template = getattr(route, "path", None)
        endpoint = getattr(route, "endpoint", None)
        if not path_template or endpoint is None:
            continue
        route_name = getattr(route, "name", "") or ""
        if route_name in {"openapi", "swagger_ui_html", "swagger_ui_redirect", "redoc_html"}:
            continue
        methods = sorted(getattr(route, "methods", []) or [])
        tags = list(getattr(route, "tags", []) or [])
        metadata = {
            "response_type": "json",
            "auth_required": _auth_required(route),
        }
        path, entry = build_route_entry(
            path=path_template,
            methods=methods,
            view=_view_name(endpoint),
            name=route_name,
            metadata=metadata,
        )
        if tags:
            entry["tags"] = tags
        routes[path] = entry
    return routes


def generate_fastapi_manifest(app, output_path: str = ".aiwaf/paths.json") -> dict[str, Any]:
    manifest = build_manifest(framework="fastapi", routes=extract_fastapi_routes(app))
    write_manifest(manifest, output_path)
    return manifest
