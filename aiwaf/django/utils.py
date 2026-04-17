import re
from django.conf import settings
from .storage import get_exemption_store, get_path_exemption_store
from ..core.defaults import DEFAULT_EXEMPT_PATHS_DJANGO
from ..core.exemptions import (
    get_path_rule_for_path as core_get_path_rule_for_path,
    is_path_exempt as core_is_path_exempt,
    normalize_middleware_name as core_normalize_middleware_name,
    normalize_paths as core_normalize_paths,
)
from ..core.utils import get_ip_from_meta, ip_in_allowlist
from ..core.logs import read_rotated_logs, parse_log_line as core_parse_log_line

def get_ip(request):
    return get_ip_from_meta(getattr(request, "META", {}))

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
    if hasattr(request, 'resolver_match') and request.resolver_match:
        view_func = request.resolver_match.func
        
        # Check if view function has aiwaf_exempt attribute
        if hasattr(view_func, 'aiwaf_exempt'):
            return True
            
        # For class-based views, check the view class
        if hasattr(view_func, 'view_class'):
            view_class = view_func.view_class
            if hasattr(view_class, 'aiwaf_exempt'):
                return True
                
            # Check dispatch method for method_decorator usage
            dispatch_method = getattr(view_class, 'dispatch', None)
            if dispatch_method and hasattr(dispatch_method, 'aiwaf_exempt'):
                return True
                
    return False

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
    """Check if middleware is disabled by PATH_RULES for this request."""
    rule = get_path_rule_for_path(getattr(request, "path", ""))
    if not rule:
        return False
    disabled = rule.get("DISABLE", []) or []
    if not isinstance(disabled, (list, tuple, set)):
        return False
    target = core_normalize_middleware_name(middleware_name)
    for entry in disabled:
        entry_norm = core_normalize_middleware_name(entry)
        if entry_norm == target:
            return True
    return False


def get_rate_limit_overrides(request):
    """Return rate limit overrides from PATH_RULES for this request."""
    rule = get_path_rule_for_path(getattr(request, "path", ""))
    if not rule:
        return {}
    overrides = rule.get("RATE_LIMIT", {}) or {}
    if not isinstance(overrides, dict):
        return {}
    return overrides
