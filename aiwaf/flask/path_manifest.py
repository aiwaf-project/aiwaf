"""Flask route extraction for AIWAF path manifests."""

from __future__ import annotations

import ast
import inspect
import textwrap

from typing import Any

from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.auth_detection import detect_auth_endpoint
from aiwaf.core.path_manifest import build_manifest, build_route_entry, is_internal_aiwaf_path, write_manifest
from aiwaf.core.source_methods import infer_methods_from_source

HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _view_name(view_func: Any) -> str:
    if view_func is None:
        return ""
    view_func = _unwrap_view(view_func)
    view_class = getattr(view_func, "view_class", None)
    if view_class is not None:
        return f"{view_class.__module__}.{view_class.__name__}"
    return f"{getattr(view_func, '__module__', '')}.{getattr(view_func, '__name__', view_func.__class__.__name__)}".strip(".")


def _unwrap_view(view_func: Any) -> Any:
    if view_func is None:
        return view_func
    try:
        return inspect.unwrap(view_func)
    except Exception:
        return view_func


def _normalize_methods(methods: Any) -> list[str]:
    return sorted({
        str(method).upper()
        for method in (methods or [])
        if str(method).upper() in HTTP_METHODS
    })


def _methods_from_view_class(view_class: Any) -> list[str]:
    if view_class is None:
        return []
    method_handlers = {
        "GET": ("get",),
        "POST": ("post",),
        "PUT": ("put",),
        "PATCH": ("patch",),
        "DELETE": ("delete",),
    }
    method_names = getattr(view_class, "methods", None) or getattr(view_class, "http_method_names", None)
    candidates = _normalize_methods(method_names) or sorted(method_handlers)
    return [
        method
        for method in candidates
        if any(hasattr(view_class, handler) for handler in method_handlers.get(method, ()))
    ]


def _methods_from_route_hint(raw_path: str = "", route_name: str = "") -> list[str]:
    text = f"{raw_path} {route_name}".lower()
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


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _has_login_required_decorator(view_func: Any) -> bool:
    try:
        source = inspect.getsource(view_func)
        tree = ast.parse(textwrap.dedent(source))
    except Exception:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if _decorator_name(decorator) in {"login_required", "fresh_login_required"}:
                return True
    return False


def _methods(rule: Any, view_func: Any = None) -> list[str]:
    methods = set(_normalize_methods(getattr(rule, "methods", None)))
    if methods:
        return sorted(methods)

    view_func = _unwrap_view(view_func)
    required = getattr(view_func, "required_methods", None)
    methods.update(_normalize_methods(required))

    provided = getattr(view_func, "methods", None)
    methods.update(_normalize_methods(provided))
    if methods:
        return sorted(methods)

    methods.update(_methods_from_view_class(getattr(view_func, "view_class", None)))
    if methods:
        return sorted(methods)

    methods.update(infer_methods_from_source(view_func) if view_func is not None else [])
    methods.update(_methods_from_route_hint(getattr(rule, "rule", ""), getattr(rule, "endpoint", "")))

    return sorted(methods or {"GET"})


def _auth_required(view_func: Any, raw_path: str) -> bool:
    path_lower = str(raw_path or "").lower()
    if path_lower.startswith(("/portal/", "/dashboard/", "/account/", "/accounts/", "/profile/", "/settings/")):
        return True

    candidates = [view_func, _unwrap_view(view_func)]
    for candidate in candidates:
        if candidate is None:
            continue
        if getattr(candidate, "login_required", False):
            return True
        if getattr(candidate, "__login_required__", False):
            return True
        if getattr(candidate, "_login_required", False):
            return True
        if getattr(candidate, "login_view", None) is not None:
            return True
        if _has_login_required_decorator(candidate):
            return True
        view_class = getattr(candidate, "view_class", None)
        if view_class is None:
            continue
        for cls in getattr(view_class, "__mro__", ()):
            if cls.__name__ in {"LoginRequiredMixin", "FreshLoginRequiredMixin"}:
                return True
        if getattr(view_class, "login_required", False):
            return True
    return False


def extract_flask_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for rule in app.url_map.iter_rules():
        if rule.endpoint == "static":
            continue
        if is_internal_aiwaf_path(str(rule.rule)):
            continue
        raw_view_func = app.view_functions.get(rule.endpoint)
        view_func = _unwrap_view(raw_view_func)
        methods = _methods(rule, view_func)
        auth_detection = detect_auth_endpoint(view_func, framework="flask", methods=methods)
        api_detection = detect_api_endpoint(view_func, framework="flask", path=str(rule.rule), methods=methods)
        metadata = {
            "auth_required": _auth_required(raw_view_func, str(rule.rule)),
            "response_type": api_detection.response_type if api_detection.response_type else ("json" if str(rule.rule).startswith("/api/") else None),
            "payload_type": api_detection.payload_type or None,
            "blueprint": rule.endpoint.rsplit(".", 1)[0] if "." in rule.endpoint else "",
            "auth_action": auth_detection.action if auth_detection.is_auth else None,
            "auth_confidence": auth_detection.confidence if auth_detection.is_auth else None,
            "auth_signals": auth_detection.signals if auth_detection.is_auth else None,
            "api_confidence": api_detection.confidence if api_detection.is_api else None,
            "api_signals": api_detection.signals if api_detection.is_api else None,
            "payload_fields": api_detection.payload_fields if api_detection.payload_fields else None,
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
