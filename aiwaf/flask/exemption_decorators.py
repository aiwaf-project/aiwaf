"""
AIWAF Flask Exemption Decorators

Provides decorators for exempting routes from AIWAF protection with fine-grained control.
"""
from functools import wraps
from flask import request, g, current_app
from aiwaf.core.exemptions import (
    get_path_rule_for_path as core_get_path_rule_for_path,
    get_path_rule_overrides_for_path as core_get_path_rule_overrides_for_path,
    is_middleware_disabled_for_path as core_is_middleware_disabled_for_path,
    normalize_middleware_name as core_normalize_middleware_name,
)
from aiwaf.core.route_plan import get_route_execution_plan


_EMPTY_PATH_RULES = ()


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
    plan = _get_request_route_plan()
    return plan.should_apply(middleware_name)


def _get_request_route_plan():
    path = ""
    rules = _get_path_rules()
    policy_version = current_app.config.get("AIWAF_ROUTE_PLAN_VERSION", 0)
    fully_exempt = False
    exempt_middlewares = set()
    required_middlewares = set()
    try:
        path = request.path or ""
        endpoint = request.endpoint
        view_func = current_app.view_functions.get(endpoint) if endpoint else None
        if view_func is not None:
            fully_exempt = bool(getattr(view_func, "_aiwaf_exempt", False))
            exempt_middlewares = set(getattr(view_func, "_aiwaf_exempt_middlewares", set()) or set())
            required_middlewares = set(getattr(view_func, "_aiwaf_required_middlewares", set()) or set())
    except Exception:
        pass

    # Merge runtime request-scoped exemptions/requirements from g.
    if getattr(g, "aiwaf_exempt", False):
        fully_exempt = True
    exempt_middlewares.update(getattr(g, "aiwaf_exempt_middlewares", set()) or set())
    required_middlewares.update(getattr(g, "aiwaf_required_middlewares", set()) or set())

    request_key = (
        path,
        id(rules),
        repr(policy_version),
        fully_exempt,
        frozenset(exempt_middlewares),
        frozenset(required_middlewares),
    )
    if getattr(g, "_aiwaf_route_plan_key", None) == request_key:
        return g._aiwaf_route_plan

    plan = get_route_execution_plan(
        path,
        rules,
        policy_version=policy_version,
        fully_exempt=fully_exempt,
        exempt_middlewares=exempt_middlewares,
        required_middlewares=required_middlewares,
    )
    g._aiwaf_route_plan_key = request_key
    g._aiwaf_route_plan = plan
    return plan


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
    try:
        if str(section_key).upper() == "RATE_LIMIT":
            return _get_request_route_plan().get_rate_limit_overrides()
        path = request.path or ""
        rules = _get_path_rules()
        return core_get_path_rule_overrides_for_path(path, rules, section_key)
    except Exception:
        return {}


def _get_path_rules():
    try:
        rules = current_app.config.get("AIWAF_PATH_RULES")
        if rules is None:
            settings = current_app.config.get("AIWAF_SETTINGS", {})
            rules = settings.get("PATH_RULES")
        return rules if rules is not None else _EMPTY_PATH_RULES
    except Exception:
        return _EMPTY_PATH_RULES


def _normalize_middleware_name(name):
    return core_normalize_middleware_name(name)


def _is_path_rule_disabled(middleware_name):
    try:
        path = request.path or ""
        rules = _get_path_rules()
        return core_is_middleware_disabled_for_path(path, rules, middleware_name)
    except Exception:
        return False
