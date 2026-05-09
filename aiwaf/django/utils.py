import re
from django.conf import settings
from .storage import get_exemption_store, get_path_exemption_store
from ..core.defaults import DEFAULT_EXEMPT_PATHS_DJANGO
from ..core.exemptions import (
    get_path_rule_for_path as core_get_path_rule_for_path,
    get_path_rule_overrides_for_path as core_get_path_rule_overrides_for_path,
    should_apply_middleware_for_path as core_should_apply_middleware_for_path,
    is_path_exempt as core_is_path_exempt,
    normalize_paths as core_normalize_paths,
)
from ..core.utils import ip_in_allowlist
from ..core.request_context import extract_ip_from_django_request
from ..core.logs import read_rotated_logs, parse_log_line as core_parse_log_line

def get_ip(request):
    return extract_ip_from_django_request(request)

def parse_log_line(line):
    return core_parse_log_line(line)

def is_ip_exempted(ip):
    """Check if IP is in exemption list"""
    exempt_ips = getattr(settings, "AIWAF_EXEMPT_IPS", [])
    if ip_in_allowlist(ip, exempt_ips):
        return True
    store = get_exemption_store()
    return store.is_exempted(ip)

def is_view_exempt(request):
    """Check if the current view is marked as AI-WAF exempt"""
    fully_exempt, _, _ = _get_view_exemption_data(request)
    return fully_exempt

def is_exempt_path(path):
    """Check if path should be exempt from AI-WAF"""
    exempt_paths = list(DEFAULT_EXEMPT_PATHS_DJANGO) + get_exempt_paths()
    return core_is_path_exempt(path, exempt_paths, allow_wildcards=False, allow_prefix=True)

def is_exempt(request):
    """Check if request should be exempt (either by path or view decorator)"""
    return is_exempt_path(request.path) or is_view_exempt(request)


def get_exempt_paths():
    """Return all exempt paths from settings and database."""
    paths = []
    settings_paths = getattr(settings, "AIWAF_EXEMPT_PATHS", [])
    if settings_paths:
        paths.extend(settings_paths)
    store = get_path_exemption_store()
    db_paths = store.get_all_exempted_paths()
    if db_paths:
        paths.extend(db_paths)
    return core_normalize_paths(paths)


def get_path_rule_for_path(path):
    """Return the most specific PATH_RULES entry matching the path."""
    if not path:
        return None
    settings_block = getattr(settings, "AIWAF_SETTINGS", {}) or {}
    return core_get_path_rule_for_path(path, settings_block.get("PATH_RULES", []) or [])


def is_middleware_disabled(request, middleware_name):
    """Check if middleware should be skipped for this request."""
    path = getattr(request, "path", "")
    settings_block = getattr(settings, "AIWAF_SETTINGS", {}) or {}
    rules = settings_block.get("PATH_RULES", []) or []
    fully_exempt, exempt_middlewares, required_middlewares = _get_view_exemption_data(request)
    return not core_should_apply_middleware_for_path(
        path,
        rules,
        middleware_name,
        fully_exempt=fully_exempt,
        exempt_middlewares=exempt_middlewares,
        required_middlewares=required_middlewares,
    )


def get_rate_limit_overrides(request):
    """Return rate limit overrides from PATH_RULES for this request."""
    path = getattr(request, "path", "")
    settings_block = getattr(settings, "AIWAF_SETTINGS", {}) or {}
    rules = settings_block.get("PATH_RULES", []) or []
    return core_get_path_rule_overrides_for_path(path, rules, "RATE_LIMIT")


def _get_view_exemption_data(request):
    """Return (fully_exempt, exempt_middlewares, required_middlewares) from resolver target."""
    fully_exempt = False
    exempt_middlewares = set()
    required_middlewares = set()
    resolver_match = getattr(request, "resolver_match", None)
    if not resolver_match:
        return fully_exempt, exempt_middlewares, required_middlewares

    view_func = getattr(resolver_match, "func", None)
    if view_func is None:
        return fully_exempt, exempt_middlewares, required_middlewares

    def _collect(target):
        nonlocal fully_exempt, exempt_middlewares, required_middlewares
        if target is None:
            return
        if bool(getattr(target, "aiwaf_exempt", False) or getattr(target, "_aiwaf_exempt", False)):
            fully_exempt = True
        exempt_middlewares.update(getattr(target, "_aiwaf_exempt_middlewares", set()) or set())
        required_middlewares.update(getattr(target, "_aiwaf_required_middlewares", set()) or set())

    _collect(view_func)
    view_class = getattr(view_func, "view_class", None)
    _collect(view_class)
    _collect(getattr(view_class, "dispatch", None) if view_class is not None else None)
    return fully_exempt, exempt_middlewares, required_middlewares
