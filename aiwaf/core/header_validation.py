import re

MAX_HEADER_BYTES = 32 * 1024
MAX_HEADER_COUNT = 100
MAX_USER_AGENT_LENGTH = 500
MAX_ACCEPT_LENGTH = 4096

REQUIRED_HEADERS = [
    'HTTP_USER_AGENT',
    'HTTP_ACCEPT',
]

BROWSER_HEADERS = [
    'HTTP_ACCEPT_LANGUAGE',
    'HTTP_ACCEPT_ENCODING',
    'HTTP_CONNECTION',
    'HTTP_CACHE_CONTROL',
]

SUSPICIOUS_USER_AGENTS = [
    r'bot', r'crawler', r'spider', r'scraper', r'curl', r'wget', r'python', r'java', r'node',
    r'go-http', r'axios', r'okhttp', r'libwww', r'lwp-trivial', r'mechanize', r'requests', r'urllib',
    r'httpie', r'postman', r'insomnia', r'^$', r'mozilla/4\.0$'
]

LEGITIMATE_BOTS = [
    r'googlebot', r'bingbot', r'slurp', r'duckduckbot', r'baiduspider', r'yandexbot',
    r'facebookexternalhit', r'twitterbot', r'linkedinbot', r'whatsapp', r'telegrambot',
    r'applebot', r'pingdom', r'uptimerobot', r'statuscake', r'site24x7'
]

SUSPICIOUS_COMBINATIONS = [
    {
        'condition': lambda headers: (
            headers.get('SERVER_PROTOCOL', '').startswith('HTTP/2') and
            'mozilla/4.0' in headers.get('HTTP_USER_AGENT', '').lower()
        ),
        'reason': 'HTTP/2 with old browser user agent'
    },
    {
        'condition': lambda headers: (
            headers.get('HTTP_USER_AGENT') and
            not headers.get('HTTP_ACCEPT')
        ),
        'reason': 'User-Agent present but no Accept header'
    },
    {
        'condition': lambda headers: (
            headers.get('HTTP_ACCEPT') == '*/*' and
            not any(h in headers for h in ['HTTP_ACCEPT_LANGUAGE', 'HTTP_ACCEPT_ENCODING'])
        ),
        'reason': 'Generic Accept header without language/encoding'
    },
    {
        'condition': lambda headers: (
            headers.get('HTTP_USER_AGENT') and
            not any(headers.get(h) for h in ['HTTP_ACCEPT_LANGUAGE', 'HTTP_ACCEPT_ENCODING', 'HTTP_CONNECTION'])
        ),
        'reason': 'Missing all browser-standard headers'
    },
    {
        'condition': lambda headers: (
            'HTTP_USER_AGENT' in headers and
            headers.get('SERVER_PROTOCOL') == 'HTTP/1.0' and
            'chrome' in headers.get('HTTP_USER_AGENT', '').lower()
        ),
        'reason': 'Modern browser with HTTP/1.0'
    }
]

def resolve_required_headers(config_required_headers, method=None):
    if config_required_headers is None:
        return list(REQUIRED_HEADERS)
    if isinstance(config_required_headers, (list, tuple)):
        return list(config_required_headers)
    if isinstance(config_required_headers, dict):
        if method:
            method_upper = method.upper()
            headers = config_required_headers.get(method_upper)
            if headers is not None:
                return list(headers)
        headers = config_required_headers.get("DEFAULT")
        if headers is not None:
            return list(headers)
    return list(REQUIRED_HEADERS)

def _check_user_agent(user_agent):
    if not user_agent:
        return None
        
    if len(user_agent) > MAX_USER_AGENT_LENGTH:
        return f"User-Agent longer than {MAX_USER_AGENT_LENGTH} chars"
    
    user_agent_lower = user_agent.lower()
    
    for legitimate_pattern in LEGITIMATE_BOTS:
        if re.search(legitimate_pattern, user_agent_lower):
            return None
            
    for suspicious_pattern in SUSPICIOUS_USER_AGENTS:
        if re.search(suspicious_pattern, user_agent_lower, re.IGNORECASE):
            return f"Pattern: {suspicious_pattern}"
            
    if len(user_agent) < 10:
        return "Too short"
        
    return None

def _check_header_combinations(headers, required_headers):
    if not required_headers:
        return None
    required = set(required_headers)
    for combo in SUSPICIOUS_COMBINATIONS:
        try:
            if combo.get('reason') == 'User-Agent present but no Accept header' and 'HTTP_ACCEPT' not in required:
                continue
            if combo['condition'](headers):
                return combo['reason']
        except Exception:
            continue
    return None

def _calculate_header_quality(headers):
    score = 0
    if headers.get('HTTP_USER_AGENT'):
        score += 2
    if headers.get('HTTP_ACCEPT'):
        score += 2
    for header in BROWSER_HEADERS:
        if headers.get(header):
            score += 1
    if headers.get('HTTP_ACCEPT_LANGUAGE') and headers.get('HTTP_ACCEPT_ENCODING'):
        score += 1
    if headers.get('HTTP_CONNECTION') == 'keep-alive':
        score += 1
    accept = headers.get('HTTP_ACCEPT', '')
    if 'text/html' in accept and 'application/xml' in accept:
        score += 1
    return score

def validate_headers_python(environ, method=None, config_required_headers=None, min_score=None):
    total_bytes = 0
    header_count = 0
    for key, value in environ.items():
        if not (key.startswith('HTTP_') or key in {'CONTENT_TYPE', 'CONTENT_LENGTH'}):
            continue

        header_count += 1
        value_str = value if isinstance(value, str) else str(value)
        total_bytes += len(key) + len(value_str)

        if total_bytes > MAX_HEADER_BYTES:
            return f"Header bytes exceed {MAX_HEADER_BYTES}"

    if header_count > MAX_HEADER_COUNT:
        return f"Header count exceeds {MAX_HEADER_COUNT}"

    user_agent = environ.get('HTTP_USER_AGENT', '')
    if user_agent and len(user_agent) > MAX_USER_AGENT_LENGTH:
        return f"User-Agent longer than {MAX_USER_AGENT_LENGTH} chars"

    accept_header = environ.get('HTTP_ACCEPT', '')
    if accept_header and len(accept_header) > MAX_ACCEPT_LENGTH:
        return f"Accept header longer than {MAX_ACCEPT_LENGTH} chars"

    required_headers = resolve_required_headers(config_required_headers, method)
    
    missing = []
    for header in required_headers:
        if not environ.get(header):
            missing.append(header.replace('HTTP_', '').replace('_', '-').lower())
    if missing:
        return f"Missing required headers: {', '.join(missing)}"

    suspicious_ua = _check_user_agent(user_agent)
    if suspicious_ua:
        return f"Suspicious user agent: {suspicious_ua}"

    suspicious_combo = _check_header_combinations(environ, required_headers)
    if suspicious_combo:
        return f"Suspicious headers: {suspicious_combo}"

    quality_score = _calculate_header_quality(environ)
    actual_min_score = min_score if min_score is not None else 3
    if required_headers and quality_score < actual_min_score:
        return f"Low header quality score: {quality_score}"

    return None
