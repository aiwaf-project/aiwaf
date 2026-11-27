"""FastAPI route extraction for AIWAF path manifests."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Optional

from aiwaf.core.api_detection import detect_api_endpoint
from aiwaf.core.auth_detection import detect_auth_endpoint
from aiwaf.core.input_sources import declared_on, fastapi_inputs
from aiwaf.core.route_risks import audit_into_manifest
from aiwaf.core.path_manifest import build_manifest, build_route_entry, is_internal_aiwaf_path, write_manifest
from aiwaf.core.source_methods import (
    decorator_name as _decorator_name,
    infer_methods_from_source,
    methods_from_route_hint,
    normalize_methods as _normalize_methods,
)
from aiwaf.core.request_declarations import fastapi_declarations, merge_entries, mask_shadowed_declarations


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
    # FastAPI token endpoints (OAuth2 password flow) are conventionally POST.
    return methods_from_route_hint(raw_path, route_name, extra_tokens=("token",))


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


def _join(prefix: str, path: str) -> str:
    """A route's path under a mount or include prefix."""
    if not prefix:
        return path
    if not path or path == "/":
        return prefix.rstrip("/") or "/"
    return prefix.rstrip("/") + "/" + path.lstrip("/")


def _walk_routes(routes, prefix: str = "", mounts: Optional[list] = None, depth: int = 0):
    """Every endpoint route the app serves, as (full path, route), with its mounts.

    Routes from include_router() (which FastAPI keeps as one entry whose routes it resolves when
    serving), from apps and routers attached with mount(), and from those apps' own included
    routers, all under their prefixes. mounts collects the mount prefixes.
    """
    if depth > 8:
        return
    for route in routes or ():
        candidates = getattr(route, "effective_candidates", None)
        if callable(candidates):  # FastAPI's include_router(): the router, with its prefix applied.
            try:
                found = candidates()
            except Exception:  # noqa: BLE001 - a private FastAPI API; skip it rather than fail init
                continue
            for candidate in found:
                if callable(getattr(candidate, "effective_candidates", None)):
                    yield from _walk_routes([candidate], prefix, mounts, depth + 1)
                    continue
                original = getattr(candidate, "original_route", None)
                path = getattr(candidate, "path", "") or getattr(getattr(candidate, "starlette_route", None), "path", "")
                if original is not None and path:
                    yield _join(prefix, path), original, candidate
            continue
        path = getattr(route, "path", None)
        if getattr(route, "endpoint", None) is None and hasattr(route, "routes"):
            # A Mount (an app or router attached with mount()), or a Host.
            inner = prefix if path is None else _join(prefix, path)
            if mounts is not None and path:
                mounts.append(inner.rstrip("/") + "/")
            try:
                children = list(route.routes or ())
            except Exception:  # noqa: BLE001 - a mounted app without routes, as StaticFiles
                children = []
            yield from _walk_routes(children, inner, mounts, depth + 1)
            continue
        if path:
            yield _join(prefix, path), route, None


class _WithIncludeDependencies:
    """An included route as FastAPI serves it: its own parameters and the dependencies the
    include_router() call adds (include_router(router, dependencies=[...])), which the original
    route doesn't carry. Everything else is the original route's."""

    def __init__(self, original: Any, context: Any) -> None:
        self._original = original
        self._context = context

    def __getattr__(self, name: str) -> Any:
        if name in {"dependant", "dependencies"} and getattr(self._context, name, None) is not None:
            return getattr(self._context, name)
        return getattr(self._original, name)


def extract_fastapi_routes(app) -> dict[str, dict[str, Any]]:
    routes: dict[str, dict[str, Any]] = {}
    mounted: list = []
    for full_path, route, context in _walk_routes(getattr(app, "routes", []), mounts=mounted):
        path_template = full_path
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
            "payload_fields": api_detection.payload_fields if api_detection.payload_fields else None,
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
        entry['request_constraints'] = fastapi_declarations(route, methods)
        framework_inputs = fastapi_inputs(_WithIncludeDependencies(route, context) if context is not None else route)
        if framework_inputs is not None:
            entry["declared_inputs"] = framework_inputs
        declaration = declared_on(raw_endpoint)
        if declaration is not None:
            entry["input_declaration"] = declaration
        routes[path] = merge_entries(routes.get(path), entry)
    return mask_shadowed_declarations(routes, mounted)


def generate_fastapi_manifest(app, output_path: str = ".aiwaf/paths.json", *, audit_source: str | None = None, write: bool = True) -> dict[str, Any]:
    """Write a manifest (or only build it, with write=False). With audit_source, also save that
    source's injection findings."""
    manifest = build_manifest(framework="fastapi", routes=extract_fastapi_routes(app))
    audit_into_manifest(manifest, audit_source)
    if write:
        write_manifest(manifest, output_path)
    return manifest
