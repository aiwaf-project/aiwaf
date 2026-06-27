"""FastAPI route extraction for AIWAF path manifests."""

from __future__ import annotations

from typing import Any

from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.auth_detection import detect_auth_endpoint
from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest
from aiwaf.core.source_methods import infer_methods_from_source

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _view_name(endpoint: Any) -> str:
    if endpoint is None:
        return ""
    return f"{getattr(endpoint, '__module__', '')}.{getattr(endpoint, '__name__', endpoint.__class__.__name__)}".strip(".")


def _auth_required(route: Any) -> bool:
    dependant = getattr(route, "dependant", None)
    dependencies = getattr(dependant, "dependencies", None)
    return bool(dependencies)


def _methods(route: Any, endpoint: Any = None) -> list[str]:
    methods = {
        str(method).upper()
        for method in (getattr(route, "methods", None) or [])
        if str(method).upper() in HTTP_METHODS
    }
    if methods:
        return sorted(methods)

    endpoint_methods = getattr(endpoint, "methods", None)
    methods.update(str(method).upper() for method in (endpoint_methods or []) if str(method).upper() in HTTP_METHODS)
    if methods:
        return sorted(methods)

    methods.update(infer_methods_from_source(endpoint) if endpoint is not None else [])

    return sorted(methods or {"GET"})


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
        methods = _methods(route, endpoint)
        tags = list(getattr(route, "tags", []) or [])
        auth_detection = detect_auth_endpoint(endpoint, framework="fastapi", methods=methods)
        api_detection = detect_api_endpoint(endpoint, framework="fastapi", path=str(path_template), methods=methods, route=route)
        metadata = {
            "response_type": api_detection.response_type or "json",
            "payload_type": api_detection.payload_type or ("json" if api_detection.is_api else None),
            "auth_required": _auth_required(route),
            "auth_action": auth_detection.action if auth_detection.is_auth else None,
            "auth_confidence": auth_detection.confidence if auth_detection.is_auth else None,
            "auth_signals": auth_detection.signals if auth_detection.is_auth else None,
            "api_confidence": api_detection.confidence if api_detection.is_api else None,
            "api_signals": api_detection.signals if api_detection.is_api else None,
            "form_confidence": api_detection.form_confidence if api_detection.form_confidence else None,
            "form_signals": api_detection.form_signals if api_detection.form_signals else None,
            "request_body": api_detection.request_body if api_detection.request_body else None,
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
