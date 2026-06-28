"""FastAPI route extraction for AIWAF path manifests."""

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


def _view_name(endpoint: Any) -> str:
    if endpoint is None:
        return ""
    endpoint = _unwrap_endpoint(endpoint)
    view_class = getattr(endpoint, "view_class", None)
    if view_class is not None:
        return f"{view_class.__module__}.{view_class.__name__}"
    bound_self = getattr(endpoint, "__self__", None)
    if bound_self is not None:
        view_class = bound_self.__class__
        return f"{view_class.__module__}.{view_class.__name__}.{getattr(endpoint, '__name__', endpoint.__class__.__name__)}"
    return f"{getattr(endpoint, '__module__', '')}.{getattr(endpoint, '__name__', endpoint.__class__.__name__)}".strip(".")


def _unwrap_endpoint(endpoint: Any) -> Any:
    if endpoint is None:
        return endpoint
    try:
        return inspect.unwrap(endpoint)
    except Exception:
        return endpoint


def _normalize_methods(methods: Any) -> list[str]:
    return sorted({
        str(method).upper()
        for method in (methods or [])
        if str(method).upper() in HTTP_METHODS
    })


def _methods_from_endpoint_class(endpoint: Any) -> list[str]:
    if endpoint is None:
        return []
    view_class = getattr(endpoint, "view_class", None)
    bound_self = getattr(endpoint, "__self__", None)
    if view_class is None and bound_self is not None:
        view_class = bound_self.__class__
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
            "token",
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


def _has_auth_dependency_in_source(endpoint: Any) -> bool:
    try:
        source = inspect.getsource(endpoint)
        tree = ast.parse(textwrap.dedent(source))
    except Exception:
        return False
    auth_names = {"Security", "OAuth2PasswordBearer", "OAuth2PasswordRequestForm", "HTTPBearer", "HTTPBasic"}
    auth_tokens = ("auth", "current_user", "user_required", "require_user", "require_login", "permission")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                name = _decorator_name(decorator)
                if name in {"login_required", "auth_required", "requires_auth", "require_auth"}:
                    return True
        if isinstance(node, ast.Call):
            name = _decorator_name(node)
            if name in auth_names:
                return True
        if isinstance(node, ast.Name) and any(token in node.id.lower() for token in auth_tokens):
            return True
    return False

def _auth_required(route: Any, endpoint: Any = None) -> bool:
    dependant = getattr(route, "dependant", None)
    dependencies = getattr(dependant, "dependencies", None)
    if dependencies:
        return True
    if getattr(dependant, "security_requirements", None):
        return True
    if getattr(route, "dependencies", None):
        return True
    endpoint = _unwrap_endpoint(endpoint)
    if endpoint is not None:
        if getattr(endpoint, "login_required", False) or getattr(endpoint, "auth_required", False):
            return True
        if _has_auth_dependency_in_source(endpoint):
            return True
    return False


def _methods(route: Any, endpoint: Any = None) -> list[str]:
    methods = set(_normalize_methods(getattr(route, "methods", None)))
    if methods:
        return sorted(methods)

    endpoint = _unwrap_endpoint(endpoint)
    endpoint_methods = getattr(endpoint, "methods", None)
    methods.update(_normalize_methods(endpoint_methods))
    if methods:
        return sorted(methods)

    methods.update(_methods_from_endpoint_class(endpoint))
    if methods:
        return sorted(methods)

    methods.update(infer_methods_from_source(endpoint) if endpoint is not None else [])
    methods.update(_methods_from_route_hint(getattr(route, "path", ""), getattr(route, "name", "")))

    return sorted(methods or {"GET"})


def extract_fastapi_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    for route in getattr(app, "routes", []):
        path_template = getattr(route, "path", None)
        raw_endpoint = getattr(route, "endpoint", None)
        endpoint = _unwrap_endpoint(raw_endpoint)
        if not path_template or endpoint is None:
            continue
        if is_internal_aiwaf_path(str(path_template)):
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
            "auth_required": _auth_required(route, raw_endpoint),
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
