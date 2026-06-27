"""Django route extraction for AIWAF path manifests."""

from __future__ import annotations

import inspect
from typing import Any

from django.urls import get_resolver
from django.urls.resolvers import URLPattern, URLResolver

from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.auth_detection import detect_auth_endpoint
from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest
from aiwaf.core.source_methods import infer_methods_from_source

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _clean_pattern(pattern: Any) -> str:
    text = str(pattern).replace("^", "").replace("$", "").strip().lstrip("/")
    return text


def _view_name(callback: Any) -> str:
    callback = _unwrap_callback(callback)
    view_class = getattr(callback, "view_class", None)
    if view_class is not None:
        return f"{view_class.__module__}.{view_class.__name__}"
    return f"{getattr(callback, '__module__', '')}.{getattr(callback, '__name__', callback.__class__.__name__)}".strip(".")


def _unwrap_callback(callback: Any) -> Any:
    if callback is None:
        return callback
    try:
        unwrapped = inspect.unwrap(callback)
    except Exception:
        unwrapped = callback
    if unwrapped is not callback:
        return unwrapped

    if getattr(callback, "__name__", "") != "_wrapped_view":
        return callback
    closure = getattr(callback, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if callable(value) and value is not callback and getattr(value, "__name__", "") != "_wrapped_view":
            return value
    return callback


def _auth_required(callback: Any, raw_path: str) -> bool:
    path_lower = f"/{raw_path.lstrip('/')}".lower()
    if path_lower.startswith(("/portal/", "/dashboard/", "/account/", "/accounts/", "/profile/", "/settings/")):
        return True

    candidates = [callback, _unwrap_callback(callback)]
    for candidate in candidates:
        if candidate is None:
            continue
        if getattr(candidate, "login_url", None) is not None or getattr(candidate, "redirect_field_name", None) is not None:
            return True
        view_class = getattr(candidate, "view_class", None)
        if view_class is None:
            continue
        for cls in getattr(view_class, "__mro__", ()):
            if cls.__name__ == "LoginRequiredMixin":
                return True
        if getattr(view_class, "login_url", None) is not None:
            return True
    return False


def _normalize_methods(methods: Any) -> list[str]:
    normalized = sorted({
        str(method).upper()
        for method in (methods or [])
        if str(method).upper() in HTTP_METHODS
    })
    return normalized


def _methods_from_closure(callback: Any) -> list[str]:
    closure = getattr(callback, "__closure__", None) or ()
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, (list, tuple, set)):
            methods = _normalize_methods(value)
            if methods:
                return methods
    return []


def _methods_from_view_class(view_class: Any) -> list[str]:
    if view_class is None:
        return []
    method_handlers = {
        "GET": ("get",),
        "POST": ("post", "form_valid", "form_invalid"),
        "PUT": ("put",),
        "PATCH": ("patch",),
        "DELETE": ("delete",),
    }
    method_names = getattr(view_class, "http_method_names", None)
    candidates = _normalize_methods(method_names) or sorted(method_handlers)
    detected = [
        method
        for method in candidates
        if any(hasattr(view_class, handler) for handler in method_handlers.get(method, ()))
    ]
    return detected


def _methods_from_route_hint(raw_path: str = "", route_name: str = "") -> list[str]:
    text = f"/{raw_path.lstrip('/')} {route_name}".lower()
    methods = {"GET"}
    if any(
        token in text
        for token in (
            "login",
            "signin",
            "upload",
            "create",
            "edit",
            "update",
            "delete",
            "submit",
            "save",
            "add",
            "post",
        )
    ):
        methods.add("POST")
    return sorted(methods)


def _methods(callback: Any, raw_path: str = "", route_name: str = "") -> list[str]:
    original = callback
    callback = _unwrap_callback(callback)
    actions = getattr(original, "actions", None) or getattr(callback, "actions", None)
    if isinstance(actions, dict):
        methods = _normalize_methods(actions)
        if methods:
            return methods

    view_class = getattr(original, "view_class", None) or getattr(callback, "view_class", None)
    methods = _methods_from_view_class(view_class)
    if methods:
        return methods

    for candidate in (original, callback):
        method_names = getattr(candidate, "http_method_names", None)
        methods = _normalize_methods(method_names)
        if methods:
            return methods
        methods = _methods_from_closure(candidate)
        if methods:
            return methods

    methods = set(infer_methods_from_source(callback) or ["GET"])
    methods.update(_methods_from_route_hint(raw_path, route_name))
    path_lower = f"/{raw_path.lstrip('/')}".lower()
    if any(token in path_lower for token in ("/login", "/signin", "/upload", "/uploads", "/files")):
        methods.add("POST")
    return sorted(methods)


def _collect_routes(patterns: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        if isinstance(pattern, URLResolver):
            nested = prefix + _clean_pattern(pattern.pattern)
            if nested and not nested.endswith("/"):
                nested += "/"
            routes.update(_collect_routes(pattern.url_patterns, nested))
            continue
        if not isinstance(pattern, URLPattern):
            continue
        raw_path = prefix + _clean_pattern(pattern.pattern)
        callback = getattr(pattern, "callback", None)
        name = getattr(pattern, "name", "") or ""
        methods = _methods(callback, raw_path, name)
        auth_detection = detect_auth_endpoint(callback, framework="django", methods=methods)
        api_detection = detect_api_endpoint(callback, framework="django", path=raw_path, methods=methods)
        path, entry = build_route_entry(
            path=raw_path or "/",
            methods=methods,
            view=_view_name(callback),
            name=name,
            metadata={
                "auth_required": _auth_required(callback, raw_path),
                "auth_action": auth_detection.action if auth_detection.is_auth else None,
                "auth_confidence": auth_detection.confidence if auth_detection.is_auth else None,
                "auth_signals": auth_detection.signals if auth_detection.is_auth else None,
                "response_type": api_detection.response_type if api_detection.response_type else ("json" if raw_path.startswith("api/") else None),
                "payload_type": api_detection.payload_type or None,
                "api_confidence": api_detection.confidence if api_detection.is_api else None,
                "api_signals": api_detection.signals if api_detection.is_api else None,
                "form_confidence": api_detection.form_confidence if api_detection.form_confidence else None,
                "form_signals": api_detection.form_signals if api_detection.form_signals else None,
                "request_body": api_detection.request_body if api_detection.request_body else None,
            },
        )
        routes[path] = entry
    return routes

def extract_django_routes() -> dict[str, dict[str, Any]]:
    resolver = get_resolver()
    return _collect_routes(resolver.url_patterns)

def generate_django_manifest(output_path: str = ".aiwaf/paths.json") -> dict[str, Any]:
    manifest = build_manifest(framework="django", routes=extract_django_routes())
    write_manifest(manifest, output_path)
    return manifest
