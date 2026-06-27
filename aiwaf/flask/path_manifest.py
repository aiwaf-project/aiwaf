"""Flask route extraction for AIWAF path manifests."""

from __future__ import annotations

from typing import Any

from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.auth_detection import detect_auth_endpoint
from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest
from aiwaf.core.source_methods import infer_methods_from_source

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _view_name(view_func: Any) -> str:
    if view_func is None:
        return ""
    return f"{getattr(view_func, '__module__', '')}.{getattr(view_func, '__name__', view_func.__class__.__name__)}".strip(".")


def _methods(rule: Any, view_func: Any = None) -> list[str]:
    methods = {
        str(method).upper()
        for method in (getattr(rule, "methods", None) or [])
        if str(method).upper() in HTTP_METHODS
    }
    if methods:
        return sorted(methods)

    required = getattr(view_func, "required_methods", None)
    methods.update(str(method).upper() for method in (required or []) if str(method).upper() in HTTP_METHODS)

    provided = getattr(view_func, "methods", None)
    methods.update(str(method).upper() for method in (provided or []) if str(method).upper() in HTTP_METHODS)
    if methods:
        return sorted(methods)

    methods.update(infer_methods_from_source(view_func) if view_func is not None else [])

    return sorted(methods or {"GET"})


def extract_flask_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        view_func = app.view_functions.get(rule.endpoint)
        methods = _methods(rule, view_func)
        auth_detection = detect_auth_endpoint(view_func, framework="flask", methods=methods)
        api_detection = detect_api_endpoint(view_func, framework="flask", path=str(rule.rule), methods=methods)
        metadata = {
            "response_type": api_detection.response_type if api_detection.response_type else ("json" if str(rule.rule).startswith("/api/") else None),
            "payload_type": api_detection.payload_type or None,
            "blueprint": rule.endpoint.rsplit(".", 1)[0] if "." in rule.endpoint else "",
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
            path=rule.rule,
            methods=methods,
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
