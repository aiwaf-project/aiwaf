from flask import request
from .storage import is_ip_whitelisted, get_path_exemptions
from aiwaf.core.defaults import DEFAULT_EXEMPT_PATHS_FLASK
from aiwaf.core.exemptions import is_path_exempt as core_is_path_exempt
from aiwaf.core.request_context import (
    extract_blacklist_extended_info_from_flask_request,
    extract_ip_from_flask_request,
)

def get_ip():
    return extract_ip_from_flask_request(request)

def is_exempt(request):
    """Check if request should be exempt from AIWAF protection."""
    ip = get_ip()
    
    # IP-based exemption
    if is_ip_whitelisted(ip):
        return True
    
    # Path-based exemption
    if is_path_exempt(request.path):
        return True
    
    # Decorator-based exemption
    if hasattr(request, 'endpoint') and request.endpoint:
        try:
            from flask import current_app
            endpoint_func = current_app.view_functions.get(request.endpoint)
            if endpoint_func and getattr(endpoint_func, '_aiwaf_exempt', False):
                return True
        except:
            pass
    
    return False

def is_path_exempt(path):
    """Check if a path should be exempt from AIWAF protection."""
    try:
        from flask import current_app
        exempt_paths = get_exempt_paths()
    except Exception:
        exempt_paths = get_default_exempt_paths()

    return core_is_path_exempt(path, exempt_paths, allow_wildcards=True, allow_prefix=True)


def get_exempt_paths():
    """Get configured path exemptions combined with stored exemptions."""
    try:
        from flask import current_app
        configured = current_app.config.get('AIWAF_EXEMPT_PATHS', get_default_exempt_paths())
    except Exception:
        configured = get_default_exempt_paths()
    stored = get_path_exemptions()
    combined = set()
    for path in configured:
        combined.add(str(path).lower())
    for path in stored:
        combined.add(str(path).lower())
    return combined

def get_default_exempt_paths():
    """Get default list of paths that should be exempt from AIWAF protection."""
    return set(DEFAULT_EXEMPT_PATHS_FLASK)


def get_blacklist_extended_info(request):
    """Build optional extended-request-info payload for blacklist entries."""
    try:
        from flask import current_app
        enabled = bool(
            current_app.config.get("AIWAF_BLACKLIST_STORE_EXTENDED_INFO", False)
            or current_app.config.get("AIWAF_CAPTURE_EXTENDED_REQUEST_INFO", False)
        )
        return extract_blacklist_extended_info_from_flask_request(
            request,
            enabled=enabled,
            max_headers=int(current_app.config.get("AIWAF_BLACKLIST_MAX_HEADERS", 50)),
            max_value_len=int(current_app.config.get("AIWAF_BLACKLIST_MAX_HEADER_VALUE_LENGTH", 512)),
            redact_headers=current_app.config.get(
                "AIWAF_BLACKLIST_REDACT_HEADERS",
                ["Authorization", "Cookie", "Set-Cookie"],
            ),
        )
    except Exception:
        return None
