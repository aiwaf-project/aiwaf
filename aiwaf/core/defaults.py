"""
Shared defaults for AIWAF.
"""

DEFAULT_EXEMPT_PATHS_DJANGO = [
    "/admin/login/",
    "/admin/",
    "/login/",
    "/accounts/login/",
    "/auth/login/",
    "/signin/",
]

DEFAULT_EXEMPT_PATHS_FLASK = {
    "/favicon.ico",
    "/robots.txt",
    "/sitemap.xml",
    "/sitemap.txt",
    "/ads.txt",
    "/security.txt",
    "/.well-known/",
    "/apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png",
    "/manifest.json",
    "/browserconfig.xml",
    "/health",
    "/healthcheck",
    "/ping",
    "/status",
    "*.css",
    "*.js",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "/static/",
    "/assets/",
    "/css/",
    "/js/",
    "/images/",
    "/img/",
    "/fonts/",
}
