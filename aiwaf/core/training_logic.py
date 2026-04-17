"""
Shared training logic helpers.
"""

from __future__ import annotations

import re

def is_scanning_path(path: str) -> bool:
    """
    Determine if a path looks like automated scanning vs legitimate browsing.
    Focus on common scanner patterns that indicate malicious intent.
    """
    path_lower = path.lower()

    scanning_patterns = [
        'wp-admin', 'wp-content', 'wp-includes', 'wp-config', 'xmlrpc.php',
        'admin', 'phpmyadmin', 'adminer', 'config', 'configuration',
        'settings', 'setup', 'install', 'installer',
        'backup', 'database', 'db', 'mysql', 'sql', 'dump',
        '.env', '.git', '.htaccess', '.htpasswd', 'passwd', 'shadow',
        'cgi-bin', 'scripts', 'shell', 'cmd', 'exec',
        '.php', '.asp', '.aspx', '.jsp', '.cgi', '.pl'
    ]

    for pattern in scanning_patterns:
        if pattern in path_lower:
            return True

    if '../' in path or '..' in path:
        return True

    if any(encoded in path for encoded in ['%2e%2e', '%252e', '%c0%ae']):
        return True

    return False

def get_default_legitimate_keywords() -> set[str]:
    """Get the core set of common legitimate keywords that shouldn't be learned as suspicious."""
    return {
        "profile", "user", "users", "account", "accounts", "settings", "dashboard", 
        "home", "about", "contact", "help", "search", "list", "lists",
        "view", "views", "edit", "create", "update", "delete", "detail", "details",
        "api", "auth", "login", "logout", "register", "signup", "signin",
        "reset", "confirm", "activate", "verify", "page", "pages",
        "category", "categories", "tag", "tags", "post", "posts",
        "article", "articles", "blog", "blogs", "news", "item", "items",
        "admin", "administration", "manage", "manager", "control", "panel",
        "config", "configuration", "option", "options", "preference", "preferences",
        
        # Django built-in app keywords
        "contenttypes", "contenttype", "sessions", "session", "messages", "message",
        "staticfiles", "static", "sites", "site", "flatpages", "flatpage",
        "redirects", "redirect", "permissions", "permission", "groups", "group",
        
        # Common third-party package keywords
        "token", "tokens", "oauth", "social", "rest", "framework", "cors",
        "debug", "toolbar", "extensions", "allauth", "crispy", "forms",
        "channels", "celery", "redis", "cache", "email", "mail",
        
        # Flask common keywords
        "static", "favicon", "robots", "sitemap", "manifest", "health", "ping",
        "status", "metrics", "test", "docs", "documentation",
        
        # Common web development terms
        "endpoint", "endpoints", "resource", "resources", "data", "export",
        "import", "upload", "download", "file", "files", "media", "images",
        "documents", "reports", "analytics", "stats", "statistics",
        
        # Common business/application terms
        "customer", "customers", "client", "clients", "company", "companies",
        "department", "departments", "employee", "employees", "team", "teams",
        "project", "projects", "task", "tasks", "event", "events",
        "notification", "notifications", "alert", "alerts",
        
        # Language/localization
        "language", "languages", "locale", "locales", "translation", "translations",
        "en", "fr", "de", "es", "it", "pt", "ru", "ja", "zh", "ko"
    }

def is_malicious_context(
    path: str,
    keyword: str,
    status: str,
    static_keywords: list[str],
    path_exists_fn,
) -> bool:
    """Determine if a keyword appears in a malicious context."""
    try:
        if path_exists_fn and path_exists_fn(path):
            return False
    except Exception:
        pass

    path_lower = path.lower()
    segments = re.split(r"\W+", path_lower)

    malicious_indicators = [
        len([seg for seg in segments if seg in static_keywords]) > 1,
        any(pattern in path_lower for pattern in [
            "../", "..\\", ".env", "wp-admin", "phpmyadmin", "config",
            "backup", "database", "mysql", "passwd", "shadow", "xmlrpc",
            "shell", "cmd", "exec", "eval", "system",
        ]),
        any(attack in path_lower for attack in [
            "union+select", "drop+table", "<script", "javascript:",
            "${", "{{", "onload=", "onerror=", "file://", "http://",
        ]),
        path_lower.count("../") > 1 or path_lower.count("..\\") > 1,
        any(encoded in path_lower for encoded in ["%2e%2e", "%252e", "%c0%ae", "%3c%73%63%72%69%70%74"]),
        status == "404" and (
            len(path_lower) > 50 or
            path_lower.count("/") > 10 or
            any(c in path_lower for c in ["<", ">", "{", "}", "$", "`"])
        ),
    ]

    return any(malicious_indicators)
