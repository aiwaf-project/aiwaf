"""
AIWAF Flask Exemption Decorators

Provides decorators for exempting routes from AIWAF protection with fine-grained control.
"""
from functools import wraps
from flask import request, g, current_app
from aiwaf.core.exemptions import get_path_rule_for_path as core_get_path_rule_for_path


def aiwaf_exempt(func):
    """
    Decorator to exempt a Flask route from ALL AIWAF middleware protection.
    
    Usage:
        @app.route('/health')
        @aiwaf_exempt
        def health_check():
            return {'status': 'ok'}
    
    This will completely bypass:
    - IP blocking/keyword detection
    - Rate limiting  
    - Honeypot detection
    - Header validation
    - AI anomaly detection
    - UUID tampering protection
    - Security logging (optional)
    
    Returns:
        Decorated function that marks request as fully exempt
    """
    # Store exemption data on the function itself
    func._aiwaf_exempt = True
    func._aiwaf_exempt_middlewares = set()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Mark this request as exempt from ALL AIWAF protection
        g.aiwaf_exempt = True
        g.aiwaf_exempt_middlewares = set()  # Clear any partial exemptions
        return func(*args, **kwargs)
    
    # Copy exemption data to wrapper for middleware access
    wrapper._aiwaf_exempt = True
    wrapper._aiwaf_exempt_middlewares = set()
    
    return wrapper


def aiwaf_exempt_from(*middleware_names):
    """
    Decorator to exempt a Flask route from specific AIWAF middlewares only.
    
    Args:
        *middleware_names: Names of middlewares to exempt from
        
    Available middleware names:
    - 'ip_keyword_block': IP blocking and keyword detection
    - 'rate_limit': Rate limiting protection
    - 'honeypot': Honeypot detection
    - 'header_validation': HTTP header validation
    - 'geo_block': Geo-blocking by country
    - 'ai_anomaly': AI-based anomaly detection
    - 'uuid_tamper': UUID tampering protection
    - 'logging': Security event logging
    
    Usage:
        @app.route('/api/webhook')
        @aiwaf_exempt_from('rate_limit', 'ai_anomaly')
        def webhook():
            return {'received': True}
    
    Returns:
        Decorated function that exempts from specified middlewares
    """
    def decorator(func):
        # Store exemption data on the function itself
        func._aiwaf_exempt_middlewares = set(middleware_names)
        func._aiwaf_exempt = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store which middlewares to exempt from
            g.aiwaf_exempt_middlewares = set(middleware_names)
            g.aiwaf_exempt = False  # Not fully exempt, just partial
            return func(*args, **kwargs)
        
        # Copy exemption data to wrapper for middleware access
        wrapper._aiwaf_exempt_middlewares = set(middleware_names)
        wrapper._aiwaf_exempt = False
        
        return wrapper
    return decorator


def aiwaf_only(*middleware_names):
    """
    Decorator to apply ONLY specific AIWAF middlewares to a route.
    All other middlewares will be bypassed.
    
    Args:
        *middleware_names: Names of middlewares to apply (all others exempted)
        
    Available middleware names:
    - 'ip_keyword_block': IP blocking and keyword detection
    - 'rate_limit': Rate limiting protection
    - 'honeypot': Honeypot detection
    - 'header_validation': HTTP header validation
    - 'geo_block': Geo-blocking by country
    - 'ai_anomaly': AI-based anomaly detection
    - 'uuid_tamper': UUID tampering protection
    - 'logging': Security event logging
        
    Usage:
        @app.route('/sensitive-endpoint')
        @aiwaf_only('ip_keyword_block', 'rate_limit')
        def sensitive_endpoint():
            return {'data': 'sensitive'}
    
    Returns:
        Decorated function that applies only specified middlewares
    """
    def decorator(func):
        # Get all available middleware names
        all_middlewares = {
            'ip_keyword_block', 'rate_limit', 'honeypot',
            'header_validation', 'geo_block', 'ai_anomaly',
            'uuid_tamper', 'logging'
        }
        
        # Exempt from all middlewares except the specified ones
        exempt_middlewares = all_middlewares - set(middleware_names)
        
        # Store exemption data on the function itself
        func._aiwaf_exempt_middlewares = exempt_middlewares
        func._aiwaf_exempt = False
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Exempt from all middlewares except the specified ones
            g.aiwaf_exempt_middlewares = exempt_middlewares
            g.aiwaf_exempt = False  # Not fully exempt, just selective
            return func(*args, **kwargs)
        
        # Copy exemption data to wrapper for middleware access
        wrapper._aiwaf_exempt_middlewares = exempt_middlewares
        wrapper._aiwaf_exempt = False
        
        return wrapper
    return decorator


def is_request_exempt(middleware_name=None):
    """
    Check if the current request is exempt from AIWAF protection.
    
    Args:
        middleware_name (str, optional): Name of specific middleware to check.
                                       If None, checks for full exemption.
    
    Returns:
        bool: True if request is exempt from the specified middleware or fully exempt
    
    Usage in middleware:
        if is_request_exempt('rate_limit'):
            return  # Skip rate limiting for this request
    """
    # Check for full exemption first
    if getattr(g, 'aiwaf_exempt', False):
        return True
    
    # If no specific middleware requested, return False (not fully exempt)
    if middleware_name is None:
        return False
    
    # Check for specific middleware exemption
    exempt_middlewares = getattr(g, 'aiwaf_exempt_middlewares', set())
    return middleware_name in exempt_middlewares


def get_exempt_middlewares():
    """
    Get the set of middlewares the current request is exempt from.
    
    Returns:
        set: Set of middleware names the current request is exempt from
    """
    if getattr(g, 'aiwaf_exempt', False):
        # If fully exempt, return all middleware names
        return {
            'ip_keyword_block', 'rate_limit', 'honeypot', 
            'header_validation', 'ai_anomaly', 'uuid_tamper', 'logging'
        }
    
    return getattr(g, 'aiwaf_exempt_middlewares', set())


def reset_exemption_status():
    """
    Reset exemption status for the current request.
    Useful for testing or manual control.
    """
    g.aiwaf_exempt = False
    g.aiwaf_exempt_middlewares = set()


def aiwaf_require_protection(*middleware_names):
    """
    Decorator to explicitly require specific AIWAF middlewares for a route.
    This is useful for ensuring critical endpoints are always protected.
    
    Args:
        *middleware_names: Names of middlewares that MUST be applied
        
    Usage:
        @app.route('/admin/delete-user')
        @aiwaf_require_protection('ip_keyword_block', 'rate_limit', 'ai_anomaly')
        def delete_user():
            return {'status': 'deleted'}
    
    Note: This decorator forces middlewares to run even if exempted elsewhere.
    """
    def decorator(func):
        func._aiwaf_required_middlewares = set(middleware_names)
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Store required middlewares that cannot be exempted
            g.aiwaf_required_middlewares = set(middleware_names)
            return func(*args, **kwargs)
        wrapper._aiwaf_required_middlewares = set(middleware_names)
        return wrapper
    return decorator


def is_middleware_required(middleware_name):
    """
    Check if a specific middleware is required for the current request.
    
    Args:
        middleware_name (str): Name of middleware to check
        
    Returns:
        bool: True if middleware is required and cannot be exempted
    """
    required_middlewares = getattr(g, 'aiwaf_required_middlewares', set())
    return middleware_name in required_middlewares


def should_apply_middleware(middleware_name):
    """
    Determine if a middleware should be applied to the current request.
    This is the main function middlewares should use to check exemption status.
    
    Args:
        middleware_name (str): Name of middleware checking exemption
        
    Returns:
        bool: True if middleware should be applied, False if exempt
        
    Logic:
        1. If middleware is explicitly required, always apply
        2. If request is fully exempt, don't apply (unless required)
        3. If middleware is in exemption list, don't apply (unless required)
        4. Otherwise, apply middleware
    """
    # Check if middleware is explicitly required (route or runtime)
    route_required = _check_route_required(middleware_name)
    if route_required:
        return True
    if is_middleware_required(middleware_name):
        return True

    # Path-specific rules (disable/override by URL prefix)
    if _is_path_rule_disabled(middleware_name):
        return False
    
    # Check route-level exemptions first (from function decorators)
    route_exempt = _check_route_exemption(middleware_name)
    if route_exempt is not None:
        return not route_exempt  # If exempt, don't apply
    
    # Check if request is exempt from this middleware (runtime exemptions)
    if is_request_exempt(middleware_name):
        return False
    
    # Default: apply middleware
    return True


def _check_route_exemption(middleware_name):
    """
    Check if the current route is exempt from a middleware.
    
    Args:
        middleware_name (str): Name of middleware to check
        
    Returns:
        bool or None: True if exempt, False if not exempt, None if unknown
    """
    try:
        # Get the current endpoint
        endpoint = request.endpoint
        if not endpoint:
            return None
            
        # Get the view function for this endpoint
        view_func = current_app.view_functions.get(endpoint)
        if not view_func:
            return None
        
        # Check for full exemption
        if getattr(view_func, '_aiwaf_exempt', False):
            return True
            
        # Check for specific middleware exemption
        exempt_middlewares = getattr(view_func, '_aiwaf_exempt_middlewares', set())
        if middleware_name in exempt_middlewares:
            return True
            
        return False
        
    except Exception:
        # If we can't determine route exemption, fall back to runtime checking
        return None

def _check_route_required(middleware_name):
    """
    Check if the current route explicitly requires a middleware.
    
    Args:
        middleware_name (str): Name of middleware to check
        
    Returns:
        bool: True if required, False otherwise
    """
    try:
        endpoint = request.endpoint
        if not endpoint:
            return False
            
        view_func = current_app.view_functions.get(endpoint)
        if not view_func:
            return False
        
        required_middlewares = getattr(view_func, '_aiwaf_required_middlewares', set())
        return middleware_name in required_middlewares
        
    except Exception:
        return False


def get_path_rule_for_request():
    """Return the best matching path rule for the current request path."""
    try:
        path = request.path or ""
        rules = _get_path_rules()
        if not rules or not path:
            return None
        return core_get_path_rule_for_path(path, rules)
    except Exception:
        return None


def get_path_rule_overrides(section_key):
    """Return override dict for a section (e.g., RATE_LIMIT) for the current path."""
    rule = get_path_rule_for_request()
    if not rule:
        return {}
    return rule.get(section_key) or rule.get(section_key.lower()) or {}


def _get_path_rules():
    try:
        rules = current_app.config.get("AIWAF_PATH_RULES")
        if rules is None:
            settings = current_app.config.get("AIWAF_SETTINGS", {})
            rules = settings.get("PATH_RULES")
        return rules or []
    except Exception:
        return []


def _normalize_middleware_name(name):
    mapping = {
        "IPAndKeywordBlockMiddleware": "ip_keyword_block",
        "RateLimitMiddleware": "rate_limit",
        "HoneypotTimingMiddleware": "honeypot",
        "HeaderValidationMiddleware": "header_validation",
        "GeoBlockMiddleware": "geo_block",
        "AIAnomalyMiddleware": "ai_anomaly",
        "UUIDTamperMiddleware": "uuid_tamper",
        "AIWAFLoggingMiddleware": "logging",
    }
    if name in mapping:
        return mapping[name]
    return name.lower()


def _is_path_rule_disabled(middleware_name):
    rule = get_path_rule_for_request()
    if not rule:
        return False
    disabled = rule.get("DISABLE") or rule.get("disable") or []
    if not disabled:
        return False
    normalized = {_normalize_middleware_name(item) for item in disabled}
    return _normalize_middleware_name(middleware_name) in normalized
