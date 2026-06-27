"""Django route extraction for AIWAF path manifests."""

from __future__ import annotations

from typing import Any

from django.urls import get_resolver
from django.urls.resolvers import URLPattern, URLResolver

from aiwaf.core.path_manifest import build_manifest, build_route_entry, write_manifest


def _clean_pattern(pattern: Any) -> str:
    text = str(pattern).replace("^", "").replace("$", "").strip().lstrip("/")
    return text


def _view_name(callback: Any) -> str:
    view_class = getattr(callback, "view_class", None)
    if view_class is not None:
        return f"{view_class.__module__}.{view_class.__name__}"
    return f"{getattr(callback, '__module__', '')}.{getattr(callback, '__name__', callback.__class__.__name__)}".strip(".")


def _methods(callback: Any) -> list[str]:
    view_class = getattr(callback, "view_class", None)
    source = view_class or callback
    method_names = getattr(source, "http_method_names", None)
    if method_names:
        return [str(method).upper() for method in method_names if str(method).lower() not in {"head", "options", "trace"}]
    actions = getattr(callback, "actions", None)
    if isinstance(actions, dict):
        return [str(method).upper() for method in actions]
    return []


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
        path, entry = build_route_entry(
            path=raw_path or "/",
            methods=_methods(callback),
            view=_view_name(callback),
            name=getattr(pattern, "name", "") or "",
            metadata={"response_type": "json" if raw_path.startswith("api/") else None},
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
