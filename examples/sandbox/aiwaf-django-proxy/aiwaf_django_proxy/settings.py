import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key")
DEBUG = os.environ.get("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "aiwaf.django.apps.AiwafConfig",
]

MIDDLEWARE = [
    "aiwaf.django.middleware.JsonExceptionMiddleware",
    "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware",
    "aiwaf.django.middleware.HeaderValidationMiddleware",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware",
    "aiwaf.django.middleware.RateLimitMiddleware",
    "aiwaf.django.middleware.GeoBlockMiddleware",
    "aiwaf.django.middleware.AIAnomalyMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "aiwaf_django_proxy.urls"
WSGI_APPLICATION = "aiwaf_django_proxy.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("DJANGO_DB_PATH", str(BASE_DIR / "db.sqlite3")),
    }
}

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
    }
}

TIME_ZONE = "UTC"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- AIWAF config (read from env for parity with the JS sandbox) ---
TARGET_BASE_URL = os.environ.get("TARGET_BASE_URL", "http://juice:3000").rstrip("/")
PORT = int(os.environ.get("PORT", "3009"))

AIWAF_GEO_BLOCK_ENABLED = os.environ.get("AIWAF_GEO_BLOCK_ENABLED", "false").lower() == "true"
AIWAF_CLEAR_STATE_ON_START = os.environ.get("AIWAF_CLEAR_STATE_ON_START", "true").lower() == "true"
AIWAF_MIN_AI_LOGS = int(os.environ.get("AIWAF_MIN_AI_LOGS", "10000"))
AIWAF_DISABLE_AI = os.environ.get("AIWAF_DISABLE_AI", "true").lower() == "true"
AIWAF_USE_RUST = os.environ.get("AIWAF_USE_RUST", "true").lower() == "true"

AIWAF_WASM_HEADER_VALIDATION = os.environ.get("AIWAF_WASM_HEADER_VALIDATION", "false").lower() == "true"
AIWAF_DEBUG_WASM_HEADERS = os.environ.get("AIWAF_DEBUG_WASM_HEADERS", "false").lower() == "true"
AIWAF_DEBUG_HEADERS = os.environ.get("AIWAF_DEBUG_HEADERS", "false").lower() == "true"

AIWAF_MIDDLEWARE_LOGGING = os.environ.get("AIWAF_MIDDLEWARE_LOGGING", "true").lower() == "true"
AIWAF_MIDDLEWARE_LOG = os.environ.get("AIWAF_MIDDLEWARE_LOG", "/logs/aiwaf-requests.log")
AIWAF_MIDDLEWARE_CSV = os.environ.get("AIWAF_MIDDLEWARE_CSV", "true").lower() == "true"
AIWAF_MIDDLEWARE_DB = os.environ.get("AIWAF_MIDDLEWARE_DB", "false").lower() == "true"

# Sandbox defaults: avoid self-blacklisting local demo traffic.
# Docker bridge clients usually appear as 172.16.0.0/12.
AIWAF_EXEMPT_IPS = ["127.0.0.1", "::1", "172.16.0.0/12"]

# Juice Shop uses socket.io polling/websocket endpoints that do not always carry
# browser-like header sets expected by strict header validation.
AIWAF_SETTINGS = {
    "PATH_RULES": [
        {
            "PREFIX": "/socket.io/",
            "DISABLE": ["HeaderValidationMiddleware"],
        },
    ]
}
