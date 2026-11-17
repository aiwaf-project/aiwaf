"""Build a native pipeline profile from the settings the regular middleware reads.

``native_config(settings, framework)`` reads the same ``AIWAF_*`` names, with the same
defaults, as the regular Django or Flask middleware, and returns a native profile plus a
note for every setting: carried over unchanged, changed (with what changes), or not
carried over (with why). Nothing is dropped silently, so the report says exactly how the
native pipeline will differ from the regular one.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .defaults import DEFAULT_EXEMPT_PATHS_DJANGO, DEFAULT_EXEMPT_PATHS_FLASK

LOGGER = logging.getLogger("aiwaf.native")
FRAMEWORKS = ("django", "flask")

# Settings with the same meaning in both pipelines: (native key, type).
_DIRECT = {
    "AIWAF_RATE_WINDOW": ("rate_window", float),
    "AIWAF_RATE_MAX": ("rate_max", int),
    "AIWAF_RATE_FLOOD": ("rate_flood", int),
    "AIWAF_RATE_KEY_MODE": ("rate_key_mode", str),
    "AIWAF_RATE_SOFT_BLOCK_BLOCKLIST": ("soft_block_blocklist", bool),
    "AIWAF_RATE_SOFT_BLOCK_BLACKLIST": ("soft_block_blocklist", bool),
    "AIWAF_ENABLE_IP_BLOCKING": ("ip_blocking", bool),
    "AIWAF_MIN_FORM_TIME": ("min_form_time", float),
    "AIWAF_MAX_PAGE_TIME": ("max_page_time", float),
    "AIWAF_ENABLE_KEYWORD_LEARNING": ("learning", bool),
    "AIWAF_DYNAMIC_TOP_N": ("top_n", int),
    "AIWAF_HEADER_QUALITY_MIN_SCORE": ("header_min_score", int),
    "AIWAF_UUID_SCORE_ENABLED": ("uuid_score_enabled", bool),
    "AIWAF_UUID_SCORE_WINDOW_SECONDS": ("uuid_window", float),
    "AIWAF_UUID_SCORE_BLOCK_THRESHOLD": ("uuid_threshold", int),
    "AIWAF_UUID_SCORE_MALFORMED_WEIGHT": ("uuid_malformed_weight", int),
    "AIWAF_UUID_SCORE_NOT_FOUND_WEIGHT": ("uuid_not_found_weight", int),
    "AIWAF_UUID_SCORE_SUCCESS_DECAY": ("uuid_success_decay", int),
    "AIWAF_SQL_INJECTION_MODE": ("sql_mode", str),
    "AIWAF_PAYLOAD_MAX_BYTES": ("max_bytes", int),
    "AIWAF_PATH_MANIFEST": ("manifest_path", str),
    "AIWAF_INJECTION_RISK_MODE": ("injection_risk_mode", str),
    "AIWAF_EXEMPT_KEYWORDS": ("exempt_keywords", list),
    "AIWAF_GEO_BLOCK_COUNTRIES": ("geo_block", list),
    "AIWAF_GEO_ALLOW_COUNTRIES": ("geo_allow", list),
    "AIWAF_REQUIRED_HEADERS": ("required_headers", list),
    "AIWAF_PATH_RULES": ("path_rules", list),
    "AIWAF_WINDOW_SECONDS": ("anomaly_window", float),
}
# Django's regular header check reads these caps; the native one with a newer aiwaf-rust.
_HEADER_LIMITS = {
    "AIWAF_MAX_HEADER_BYTES": "header_max_bytes",
    "AIWAF_MAX_HEADER_COUNT": "header_max_count",
    "AIWAF_MAX_USER_AGENT_LENGTH": "header_max_user_agent",
    "AIWAF_MAX_ACCEPT_LENGTH": "header_max_accept",
}
# Read from the environment by the regular middleware when they aren't settings.
_ENVIRONMENT = ("AIWAF_SQL_INJECTION_MODE", "AIWAF_PAYLOAD_MAX_BYTES", "AIWAF_INJECTION_RISK_MODE",
                "AIWAF_PATH_MANIFEST")
# Settings the converter reads itself, so they aren't reported as unknown.
_HANDLED = {"AIWAF_CSV_DATA_DIR", "AIWAF_DATA_DIR", "AIWAF_STORAGE_MODE", "AIWAF_EXEMPT_PATHS",
            "AIWAF_EXEMPT_IPS", "AIWAF_MALICIOUS_KEYWORDS", "AIWAF_MIDDLEWARE_LOGGING",
            "AIWAF_CSV_PIPELINE", "AIWAF_GEO_BLOCK_ENABLED", "AIWAF_RATE_CACHE_BACKEND", "AIWAF_REDIS_URL",
            "AIWAF_USE_CSV",
            # Django's legacy nested settings, applied to the flat AIWAF_* settings before conversion.
            "AIWAF_SETTINGS",
            # Chooses between the pipelines; not a pipeline setting.
            "AIWAF_NATIVE_PIPELINE"}
# Read only by ``aiwaf train`` and the other commands, never while a request is handled.
_TRAINING = {"AIWAF_AI_BOOTSTRAP", "AIWAF_AI_CONTAMINATION", "AIWAF_AI_MAX_FEATURES", "AIWAF_AI_MAX_SAMPLES",
             "AIWAF_AI_N_ESTIMATORS", "AIWAF_AI_RANDOM_STATE", "AIWAF_AI_WARM_START", "AIWAF_FORCE_AI_TRAINING",
             "AIWAF_MIN_TRAIN_LOGS", "AIWAF_PYTHON_FEATURE_BATCH_SIZE", "AIWAF_PYTHON_PARALLEL_CHUNK_SIZE",
             "AIWAF_PYTHON_PARALLEL_FEATURES", "AIWAF_PYTHON_PARALLEL_WORKERS", "AIWAF_RUST_FEATURE_BATCH_SIZE",
             "AIWAF_RUST_FEATURE_CHUNK_SIZE", "AIWAF_RUST_ISOLATION_FOREST", "AIWAF_RUST_PARALLEL_CHUNK_SIZE",
             "AIWAF_RUST_PARALLEL_FEATURES", "AIWAF_RUST_PARALLEL_WORKERS", "AIWAF_TRAINING_STATE_DIR"}
# Read by the native adapters as the regular middleware reads them: the GeoIP database and its
# lookup cache. AIWAF_RUST_REQUEST_ENGINE chooses the Rust request checks, which native always uses.
_SAME = {"AIWAF_GEOIP_DB_PATH", "AIWAF_GEO_CACHE_SECONDS", "AIWAF_GEO_CACHE_PREFIX", "AIWAF_RUST_REQUEST_ENGINE"}
# Set from Django's legacy AIWAF_SETTINGS; Django's regular middleware doesn't read them either.
_DJANGO_UNREAD = {"AIWAF_LOG_FORMAT", "AIWAF_LOG_LEVEL"}
# Flask's regular middleware reads no header caps, so the native defaults apply as they do there.
_FLASK_UNREAD = set(_HEADER_LIMITS)
# What a blocklist entry records about the request: the regular block store writes it in both
# pipelines, from the request being handled.
_BLOCK_INFO = {"AIWAF_BLOCKLIST_STORE_EXTENDED_INFO", "AIWAF_BLACKLIST_STORE_EXTENDED_INFO",
               "AIWAF_BLOCKLIST_MAX_HEADERS", "AIWAF_BLACKLIST_MAX_HEADERS",
               "AIWAF_BLOCKLIST_MAX_HEADER_VALUE_LENGTH", "AIWAF_BLACKLIST_MAX_HEADER_VALUE_LENGTH",
               "AIWAF_BLOCKLIST_REDACT_HEADERS", "AIWAF_BLACKLIST_REDACT_HEADERS"}
_FLASK_BLOCK_INFO = {"AIWAF_CAPTURE_EXTENDED_REQUEST_INFO", "AIWAF_EXTENDED_REQUEST_INFO_HEADERS",
                     "AIWAF_EXTENDED_REQUEST_INFO_MAX_BYTES", "AIWAF_EXTENDED_REQUEST_INFO_REDACT_HEADERS"}
# The settings that turn each Django stage on.
_DJANGO_STAGES = {
    "aiwaf.django.middleware.GeoBlockMiddleware": "geo_block",
    "aiwaf.django.middleware.IPAndKeywordBlockMiddleware": "ip_keyword_block",
    "aiwaf.django.middleware.RateLimitMiddleware": "rate_limit",
    "aiwaf.django.middleware.AIAnomalyMiddleware": "ai_anomaly",
    "aiwaf.django.middleware.HoneypotTimingMiddleware": "honeypot",
    "aiwaf.django.middleware.UUIDTamperMiddleware": "uuid_tamper",
    "aiwaf.django.middleware.HeaderValidationMiddleware": "header_validation",
    "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware": "logging",
}
# Django's regular header check: its default required headers, and the static files it skips.
_DJANGO_REQUIRED = ("HTTP_USER_AGENT", "HTTP_ACCEPT")
_DJANGO_STATIC_EXTENSIONS = (".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".woff2", ".ttf")
_DJANGO_STATIC_PATHS = ("/static/", "/media/", "/assets/", "/favicon.ico")
DJANGO_LOGGER = "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware"
# The logger middleware's own settings, read by it as it runs beside the native pipeline.
_DJANGO_LOGGER_SETTINGS = {"AIWAF_MIDDLEWARE_LOG", "AIWAF_MIDDLEWARE_CSV", "AIWAF_MIDDLEWARE_DB"}
# Django's regular IP and keyword middleware blocks these when AIWAF_MALICIOUS_KEYWORDS isn't set.
_DJANGO_STATIC_KEYWORDS = [".php", "xmlrpc", "wp-", ".env", ".git", ".bak", "conflg", "shell", "filemanager"]
_GLOB = ("*", "?", "[")


@dataclass(frozen=True)
class Note:
    setting: str
    outcome: str  # "carried", "changed" or "not carried"
    detail: str = ""


@dataclass
class NativeConversion:
    config: dict
    notes: list = field(default_factory=list)

    def note(self, setting, outcome, detail=""):
        self.notes.append(Note(setting, outcome, detail))

    def differences(self):
        """Notes for settings that don't carry over exactly."""
        return [note for note in self.notes if note.outcome != "carried"]

    def report(self) -> str:
        carried = sum(note.outcome == "carried" for note in self.notes)
        lines = [f"Native pipeline settings: {carried} carried over unchanged."]
        for note in self.differences():
            lines.append(f"- {note.setting}: {note.outcome}. {note.detail}".rstrip())
        return "\n".join(lines)

    def log(self, logger=LOGGER):
        """Log the report: a warning when anything differs, otherwise one info line."""
        (logger.warning if self.differences() else logger.info)("%s", self.report())


def _engine_supports(feature: str) -> bool:
    """Whether the installed aiwaf-rust has an optional native pipeline feature."""
    try:
        import aiwaf_rust
    except ImportError:
        return False
    supports = getattr(getattr(aiwaf_rust, "NativeCSVPipeline", None), "supports", None)
    return bool(supports and supports(feature))


def _rate_backend(result, framework, settings):
    """Count rate limits where the regular rate limiter counts them."""
    shared = _engine_supports("rate_counter")
    if framework == "django":
        # The regular rate limiter always counts in Django's default cache.
        if shared:
            result.config["rate_backend"] = "django_cache"
        return
    backend = str(settings.get("AIWAF_RATE_CACHE_BACKEND") or "memory").strip().lower()
    if backend == "csv":
        result.config["rate_backend"] = "csv"
        result.note("AIWAF_RATE_CACHE_BACKEND", "carried")
    elif backend == "redis":
        _redis_rates(result, "AIWAF_RATE_CACHE_BACKEND", settings.get("AIWAF_REDIS_URL"), shared)
    elif "AIWAF_RATE_CACHE_BACKEND" in settings:
        result.note("AIWAF_RATE_CACHE_BACKEND", "carried")


def _redis_rates(result, setting, url, shared):
    if shared and url:
        result.config["rate_backend"] = "redis"
        result.config["rate_redis_url"] = str(url)
        result.note(setting, "carried")
    elif shared:
        result.note(setting, "not carried", "No Redis URL is set, so native rate limits are counted in each worker.")
    else:
        result.note(setting, "not carried",
                    "Shared native rate limits need a newer aiwaf-rust, so they're counted in each worker. "
                    "Run: pip install -U aiwaf-rust")


def _at_least(minimum, what):
    def check(value):
        if value < minimum:
            raise ValueError(f"the native pipeline needs {what} of at least {minimum}")
        return value
    return check


def _one_of(choices, normalize=lambda v: str(v).strip().lower()):
    def check(value):
        value = normalize(value)
        if value not in choices:
            raise ValueError(f"expected one of {', '.join(sorted(choices))}")
        return value
    return check


def _payload_size(value):
    if not 1 <= value <= 1048576:
        raise ValueError("expected 1 to 1048576 bytes, as the regular middleware requires")
    return value


# Native profile values, normalized as the regular middleware reads them, or refused with the
# reason when the native engine can't represent them. A refused value is reported as not carried.
_NATIVE_VALUES = {
    "rate_window": _at_least(1, "a window"),
    "uuid_window": _at_least(1, "a window"),
    "anomaly_window": _at_least(1, "a window"),
    "header_max_bytes": _at_least(0, "a limit"),
    "header_max_count": _at_least(0, "a limit"),
    "header_max_user_agent": _at_least(0, "a limit"),
    "header_max_accept": _at_least(0, "a limit"),
    "rate_max": _at_least(0, "a limit"),
    "rate_flood": _at_least(0, "a threshold"),
    "min_form_time": _at_least(0, "a time"),
    "max_page_time": _at_least(0, "a time"),
    "top_n": _at_least(0, "a count"),
    "uuid_threshold": _at_least(1, "a threshold"),
    "uuid_malformed_weight": _at_least(0, "a weight"),
    "uuid_not_found_weight": _at_least(0, "a weight"),
    "uuid_success_decay": _at_least(0, "a decay"),
    "max_bytes": _payload_size,
    "sql_mode": _one_of({"block", "monitor", "off"}),
    "injection_risk_mode": _one_of({"block", "log", "off"}),
    # The regular rate limiter reads ip or ip_only as per-IP, and anything else as per IP and path.
    "rate_key_mode": lambda v: "ip" if str(v or "").strip().lower() in {"ip", "ip_only"} else "ip_path",
}


def _native_value(key, kind, raw):
    value = _value(kind, raw)
    check = _NATIVE_VALUES.get(key)
    return check(value) if check else value


def _check_path_rules(result, setting):
    """Keep the path rules' rate limits the native engine can apply; report the others."""
    from .csv_pipeline import _rate_overrides

    rules = []
    for rule in result.config.get("path_rules") or []:
        if not isinstance(rule, dict):
            result.note(setting, "not carried", f"A path rule must be a dictionary, not {rule!r}.")
            continue
        limit = rule.get("RATE_LIMIT", rule.get("rate_limit"))
        try:
            limits = _rate_overrides(limit if isinstance(limit, dict) else None)
            if limits.get("rate_window", 1) < 1 or limits.get("rate_max", 0) < 0 or limits.get("rate_flood", 0) < 0:
                raise ValueError("the native pipeline needs a window of at least 1 second and limits of 0 or more")
        except ValueError as error:
            result.note(setting, "not carried", f"The RATE_LIMIT for {rule.get('PREFIX')!r} isn't used natively: {error}.")
            rule = {k: v for k, v in rule.items() if k not in ("RATE_LIMIT", "rate_limit")}
        rules.append(rule)
    if "path_rules" in result.config:
        result.config["path_rules"] = rules


def _value(kind, raw):
    if kind is list:
        if isinstance(raw, str):
            return [item.strip() for item in raw.split(",") if item.strip()]
        if not isinstance(raw, (list, tuple, set, frozenset)):
            raise ValueError("expected a list")
        return [item if isinstance(item, dict) else str(item) for item in raw]
    if kind is bool:
        if isinstance(raw, str):
            if raw.strip().lower() not in ("1", "0", "true", "false", "yes", "no", "on", "off"):
                raise ValueError("expected true or false")
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(raw, bool) and kind is not str:
        raise ValueError("expected a number")
    return kind(raw.strip() if isinstance(raw, str) else raw)


def _exempt_paths(result, framework, settings, store=False):
    configured = settings.get("AIWAF_EXEMPT_PATHS")
    if framework == "django":
        # Django always exempts its login and admin paths, then adds the setting.
        paths = list(DEFAULT_EXEMPT_PATHS_DJANGO) + list(configured or [])
    else:
        # Flask uses the setting instead of its defaults when it's set.
        paths = list(configured) if configured is not None else sorted(DEFAULT_EXEMPT_PATHS_FLASK)
    literal = [path for path in dict.fromkeys(str(p) for p in paths) if not any(c in path for c in _GLOB)]
    patterns = [path for path in dict.fromkeys(str(p) for p in paths) if any(c in path for c in _GLOB)]
    result.config["exempt_paths"] = literal
    if patterns and store:
        # The regular store's own check matches them, patterns included.
        result.note("AIWAF_EXEMPT_PATHS", "carried")
    elif patterns:
        result.note("AIWAF_EXEMPT_PATHS", "changed",
                    "The native pipeline matches literal paths and prefixes only, so these patterns aren't "
                    f"exempt and their requests are checked: {', '.join(patterns)}.")
    elif configured is not None:
        result.note("AIWAF_EXEMPT_PATHS", "carried")


def _exempt_ips(result, settings):
    configured = settings.get("AIWAF_EXEMPT_IPS")
    if configured is None:
        return
    ips = [str(ip).strip() for ip in (configured.split(",") if isinstance(configured, str) else configured)]
    exact = [ip for ip in ips if ip and "/" not in ip and "*" not in ip]
    ranges = [ip for ip in ips if "/" in ip or "*" in ip]
    result.config["exempt_ips"] = exact
    if ranges:
        result.note("AIWAF_EXEMPT_IPS", "changed",
                    f"The native pipeline exempts exact addresses only, so these ranges aren't exempt: {', '.join(ranges)}.")
    else:
        result.note("AIWAF_EXEMPT_IPS", "carried")


def _stages(result, framework, settings, middleware):
    if framework != "django" or middleware is None:
        return
    middleware = list(middleware)
    enabled = {stage for path, stage in _DJANGO_STAGES.items() if path in middleware}
    if not enabled:
        # Already switched to the native middleware: there's no regular chain to copy.
        result.note("MIDDLEWARE", "changed", "It lists no regular AIWAF middleware, so all native stages are enabled.")
        return
    if "logging" in enabled and not settings.get("AIWAF_MIDDLEWARE_LOGGING", False):
        # The logger middleware writes nothing unless AIWAF_MIDDLEWARE_LOGGING is on.
        enabled.discard("logging")
    if "logging" in enabled:
        # The regular logger middleware keeps logging, beside the native pipeline, to the same
        # CSV file and RequestLog. It sees the same requests when it's listed before or after
        # every other AIWAF middleware; between them, it would see blocks from some and not others.
        aiwaf = [path for path in middleware if path in _DJANGO_STAGES]
        if aiwaf.index(DJANGO_LOGGER) in (0, len(aiwaf) - 1) and len(aiwaf) > 1:
            enabled.discard("logging")
            result.note("AIWAF_MIDDLEWARE_LOGGING", "carried")
        else:
            result.note("AIWAF_MIDDLEWARE_LOGGING", "changed",
                        "The regular logger middleware runs beside the native pipeline only when it's listed "
                        "before or after every other AIWAF middleware.")
    result.config["enabled"] = sorted(enabled)
    result.note("MIDDLEWARE", "carried")


def native_config(settings: Mapping[str, Any], framework: str, *, middleware: Iterable[str] | None = None,
                  environ: Mapping[str, str] | None = None) -> NativeConversion:
    """Convert regular-middleware settings to a native profile.

    ``settings`` maps setting names to values: Django's settings or Flask's ``app.config``.
    ``middleware`` is Django's ``MIDDLEWARE`` list, which decides the enabled stages.
    """
    if framework not in FRAMEWORKS:
        raise ValueError(f"framework must be one of {', '.join(FRAMEWORKS)}")
    environ = os.environ if environ is None else environ
    settings = dict(settings)
    for name in _ENVIRONMENT:
        if name not in settings and name in environ:
            settings[name] = environ[name]
    result = NativeConversion(config={})

    data_dir = settings.get("AIWAF_CSV_DATA_DIR" if framework == "django" else "AIWAF_DATA_DIR") \
        or (environ.get("AIWAF_DATA_DIR") if framework == "flask" else None) or "aiwaf_data"
    result.config["data_dir"] = str(data_dir)
    storage = str(settings.get("AIWAF_STORAGE_MODE", "models" if framework == "django" else "csv")).lower()
    # Flask chooses its storage with AIWAF_USE_CSV: CSV by default, otherwise its database.
    storage_setting = "AIWAF_STORAGE_MODE"
    if framework == "flask":
        storage_setting = "AIWAF_USE_CSV"
        try:
            storage = "csv" if _value(bool, settings.get("AIWAF_USE_CSV", True)) else "database"
        except ValueError:
            storage = "csv"  # Any other value is truthy to Flask, which then uses CSV.
    store = _engine_supports("host_store")
    if store:
        # The native pipeline reads and writes the regular stores, so stored entries carry over.
        result.config["store"] = framework
        if storage_setting in settings:
            result.note(storage_setting, "carried")
    elif storage != "csv":
        result.note(storage_setting, "changed",
                    f"The native pipeline keeps blocks, exemptions and keywords in CSV files in {data_dir}. "
                    f"Entries stored in {storage} aren't copied there.")
    elif storage_setting in settings:
        # Flask's CSV mode and the native pipeline use the same files in the data directory.
        result.note(storage_setting, "carried")

    _exempt_paths(result, framework, settings, store)
    _exempt_ips(result, settings)
    if framework == "django":
        keywords = settings.get("AIWAF_MALICIOUS_KEYWORDS")
        result.config["static_keywords"] = [str(k) for k in (keywords if keywords is not None else _DJANGO_STATIC_KEYWORDS)]
        if keywords is not None:
            result.note("AIWAF_MALICIOUS_KEYWORDS", "carried")
    elif settings.get("AIWAF_MALICIOUS_KEYWORDS") is not None:
        result.note("AIWAF_MALICIOUS_KEYWORDS", "not carried",
                    "Flask's regular middleware doesn't read it, so the native defaults apply as before.")

    required = settings.get("AIWAF_REQUIRED_HEADERS")
    if isinstance(required, Mapping):
        # Per method, falling back to DEFAULT and then the regular defaults, as the regular
        # header check reads them.
        if _engine_supports("method_headers"):
            default = required.get("DEFAULT")
            result.config["required_headers"] = [str(h) for h in (default if default is not None else _DJANGO_REQUIRED)]
            result.config["required_headers_by_method"] = {str(method): [str(h) for h in headers]
                                                           for method, headers in required.items()
                                                           if method != "DEFAULT" and headers is not None}
            result.note("AIWAF_REQUIRED_HEADERS", "carried")
        else:
            result.note("AIWAF_REQUIRED_HEADERS", "not carried", "Per-method lists need a newer aiwaf-rust: "
                                                                 "pip install -U aiwaf-rust")
    for name, (key, kind) in _DIRECT.items():
        if settings.get(name) is None or (name == "AIWAF_REQUIRED_HEADERS" and isinstance(settings[name], Mapping)):
            continue
        try:
            result.config[key] = _native_value(key, kind, settings[name])
        except (TypeError, ValueError) as error:
            result.note(name, "not carried", f"Its value couldn't be read ({error}), so the native default applies.")
            continue
        result.note(name, "carried")
    if framework == "django":
        limits = _engine_supports("header_limits")
        for name, key in _HEADER_LIMITS.items():
            if settings.get(name) is None:
                continue
            if not limits:
                result.note(name, "not carried", "The installed aiwaf-rust has fixed header limits; "
                                                 "upgrade it to use this one: pip install -U aiwaf-rust")
                continue
            try:
                result.config[key] = _native_value(key, int, settings[name])
            except (TypeError, ValueError) as error:
                result.note(name, "not carried", f"Its value couldn't be used ({error}).")
                continue
            result.note(name, "carried")
    maximum, flood = result.config.get("rate_max", 20), result.config.get("rate_flood", 40)
    if flood < maximum:
        # The regular rate limiter flood-blocks above the threshold before the limit is reached;
        # the native engine needs the limit at or below it, which behaves the same.
        result.config["rate_max"] = flood
        result.notes = [note for note in result.notes if note.setting != "AIWAF_RATE_MAX"]
        result.note("AIWAF_RATE_MAX", "carried",
                    f"It's above AIWAF_RATE_FLOOD ({flood}), so requests are flood-blocked before reaching it. "
                    f"The native limit is set to {flood}, which blocks at the same point.")
    if "required_headers" in result.config:
        result.config["required_headers"] = [h if h.startswith("HTTP_") else "HTTP_" + h.upper().replace("-", "_")
                                             for h in result.config["required_headers"]]
    for method, headers in result.config.get("required_headers_by_method", {}).items():
        result.config["required_headers_by_method"][method] = [h if h.startswith("HTTP_") else "HTTP_" + h.upper().replace("-", "_")
                                                               for h in headers]
    _rate_backend(result, framework, settings)
    if framework == "flask":
        # Flask's regular honeypot middleware keeps GET timestamps in the process's memory.
        result.config["honeypot_backend"] = "memory"
    _stages(result, framework, settings, middleware)
    if framework == "django" and "header_validation" in result.config.get("enabled", ()):
        # Django's regular header check skips static files.
        if _engine_supports("header_skip"):
            result.config["header_skip_extensions"] = list(_DJANGO_STATIC_EXTENSIONS)
            result.config["header_skip_paths"] = list(_DJANGO_STATIC_PATHS)
        else:
            result.note("HeaderValidationMiddleware", "changed", "The regular header check skips static files; "
                                                                 "the installed aiwaf-rust checks them too. "
                                                                 "Upgrade it: pip install -U aiwaf-rust")
    if framework == "django" and "geo_block" in result.config.get("enabled", ()):
        # Django's regular geo middleware does nothing unless AIWAF_GEO_BLOCK_ENABLED is on.
        try:
            geo_on = _value(bool, settings.get("AIWAF_GEO_BLOCK_ENABLED", False))
        except ValueError:
            geo_on = bool(settings.get("AIWAF_GEO_BLOCK_ENABLED"))
        # Nor without an allow or block list of its own, whatever the store's blocked countries.
        if not geo_on or not (result.config.get("geo_allow") or result.config.get("geo_block")):
            result.config["enabled"] = [stage for stage in result.config["enabled"] if stage != "geo_block"]
    if settings.get("AIWAF_GEO_BLOCK_ENABLED") is not None:
        result.note("AIWAF_GEO_BLOCK_ENABLED", "carried")

    same = _TRAINING | _SAME | _BLOCK_INFO | (_DJANGO_UNREAD | _DJANGO_LOGGER_SETTINGS if framework == "django"
                                              else _FLASK_UNREAD | _FLASK_BLOCK_INFO)
    for name in sorted(settings):
        if not name.startswith("AIWAF_") or name in _DIRECT or name in _HANDLED \
                or (framework == "django" and name in _HEADER_LIMITS):
            continue
        if name in same:
            result.note(name, "carried")
        else:
            result.note(name, "not carried", "The native pipeline has no equivalent setting.")
    _check_path_rules(result, "AIWAF_PATH_RULES")
    return result


# FastAPI's AIWAFConfig keys with the same meaning natively: (native key, type).
_FASTAPI_DIRECT = {
    "rate_limiting.max_requests": ("rate_max", int),
    "rate_limiting.window_seconds": ("rate_window", float),
    "rate_limiting.flood_threshold": ("rate_flood", int),
    "rate_limiting.key_mode": ("rate_key_mode", str),
    "rate_limiting.soft_block_blocklist": ("soft_block_blocklist", bool),
    "ip_keyword_block.malicious_keywords": ("static_keywords", list),
    "ip_keyword_block.sql_injection_mode": ("sql_mode", str),
    "ip_keyword_block.payload_max_bytes": ("max_bytes", int),
    "honeypot.min_form_time": ("min_form_time", float),
    "honeypot.max_page_time": ("max_page_time", float),
    "honeypot.skip_authenticated": ("skip_authenticated", bool),
    "geo_block.allow_countries": ("geo_allow", list),
    "geo_block.block_countries": ("geo_block", list),
    "header_validation.quality_threshold": ("header_min_score", int),
}
# AIWAFConfig sections that turn each native stage on.
_FASTAPI_STAGES = {"geo_block": "geo_block", "ip_keyword_block": "ip_keyword_block", "rate_limiting": "rate_limit",
                   "ai_anomaly": "ai_anomaly", "honeypot": "honeypot", "uuid_tamper": "uuid_tamper",
                   "header_validation": "header_validation", "logging_middleware": "logging"}
_FASTAPI_HANDLED = {"storage.backend", "storage.data_dir", "header_validation.exempt_paths", "rate_limiting.exempt_ips",
                    "rate_limiting.cache_backend", "rate_limiting.redis_url", "path_rules", "PATH_RULES", "path_manifest", "PATH_MANIFEST",
                    "exemptions.private_ips_exempted", "exemptions.auto_exempt_patterns",
                    "exemptions.localhost_exempted"} | {f"{section}.enabled" for section in _FASTAPI_STAGES}


def _flatten(value, prefix=""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}{key}.")
    else:
        yield prefix[:-1], value


def native_config_fastapi(config=None, *, environ: Mapping[str, str] | None = None) -> NativeConversion:
    """Convert FastAPI's ``AIWAFConfig`` (by default, the one the regular middleware uses)."""
    from .runtime_config import AIWAFConfig, get_config

    config = get_config() if config is None else config
    values = dict(_flatten(config._config if isinstance(config, AIWAFConfig) else dict(config)))
    defaults = dict(_flatten(AIWAFConfig(load_from_env=False)._config))
    environ = os.environ if environ is None else environ
    changed = {key for key, value in values.items() if defaults.get(key, object()) != value}
    # FastAPI's regular honeypot middleware keeps GET timestamps in the process's memory.
    honeypot_backend = "memory"
    # The regular stores keep their files in storage.data_dir, which AIWAF_DATA_DIR sets.
    data_dir = values.get("storage.data_dir") or environ.get("AIWAF_DATA_DIR") or "aiwaf_data"
    result = NativeConversion(config={"data_dir": str(data_dir), "honeypot_backend": honeypot_backend})
    if "storage.data_dir" in changed:
        result.note("storage.data_dir", "carried")

    store = _engine_supports("host_store")
    if store:
        # The regular stores, on the same storage backend, so stored entries carry over.
        result.config["store"] = "fastapi"
    elif str(values.get("storage.backend", "memory")).lower() != "csv":
        result.note("storage.backend", "changed",
                    f"The native pipeline keeps state in CSV files in {result.config['data_dir']}. "
                    f"Entries in the {values.get('storage.backend')} backend aren't copied there.")
    for key, (native, kind) in _FASTAPI_DIRECT.items():
        if values.get(key) is None:
            continue
        try:
            result.config[native] = _native_value(native, kind, values[key])
        except (TypeError, ValueError) as error:
            result.note(key, "not carried", f"Its value couldn't be read ({error}), so the native default applies.")
            continue
        if key in changed:
            result.note(key, "carried")
    if "sql_mode" not in result.config and environ.get("AIWAF_SQL_INJECTION_MODE"):
        try:
            result.config["sql_mode"] = _native_value("sql_mode", str, environ["AIWAF_SQL_INJECTION_MODE"])
        except ValueError as error:
            result.note("AIWAF_SQL_INJECTION_MODE", "not carried", f"Its value couldn't be used ({error}).")

    stages = sorted(native for section, native in _FASTAPI_STAGES.items()
                    if values.get(f"{section}.enabled", False) or (section == "geo_block" and values.get("geo_block.block_countries")))
    result.config["enabled"] = stages
    if "logging" in stages:
        native_access_log(result, "logging_middleware", values.get("logging_middleware.log_dir") or "aiwaf_logs")

    rules = list(values.get("path_rules") or values.get("PATH_RULES") or [])
    # Header-only exemptions become rules that turn off header validation, and nothing else.
    header_paths = [str(p) for p in values.get("header_validation.exempt_paths") or []]
    rules += [{"PREFIX": path, "DISABLE": ["header_validation"]} for path in header_paths
              if not any(c in path for c in _GLOB)]
    patterns = [path for path in header_paths if any(c in path for c in _GLOB)]
    if patterns:
        result.note("header_validation.exempt_paths", "changed",
                    f"Patterns can't be path rules, so header validation applies to: {', '.join(patterns)}.")
    if rules:
        result.config["path_rules"] = rules
        _check_path_rules(result, "path_rules")
    manifest = values.get("path_manifest") or values.get("PATH_MANIFEST")
    if manifest:
        result.config["manifest_path"] = str(manifest)

    is_range = lambda ip: "/" in str(ip) or "*" in str(ip)  # noqa: E731
    candidates = [str(ip) for ip in values.get("rate_limiting.exempt_ips") or []]
    automatic = [str(ip) for ip in values.get("exemptions.auto_exempt_patterns") or []] \
        if values.get("exemptions.private_ips_exempted", True) else []
    exact = [ip for ip in dict.fromkeys(candidates + automatic) if not is_range(ip)]
    if exact:
        result.config["exempt_ips"] = exact
    ranges = [ip for ip in dict.fromkeys(candidates + automatic) if is_range(ip)]
    if store:
        # The regular exemption store loads these patterns and matches them.
        ranges = [ip for ip in ranges if ip not in automatic]
    if ranges:
        result.note("exemptions.auto_exempt_patterns", "changed",
                    "The native pipeline exempts exact addresses only, so these ranges aren't exempt: "
                    f"{', '.join(ranges)}. Add exact addresses to exempt_ips.")
    rate_cache = str(values.get("rate_limiting.cache_backend") or "memory").lower()
    if rate_cache == "csv":
        result.config["rate_backend"] = "csv"
        result.note("rate_limiting.cache_backend", "carried")
    elif rate_cache == "redis":
        _redis_rates(result, "rate_limiting.cache_backend",
                     values.get("rate_limiting.redis_url") or values.get("AIWAF_REDIS_URL"), _engine_supports("rate_counter"))

    for key in sorted(changed):
        if key not in _FASTAPI_DIRECT and key not in _FASTAPI_HANDLED and not key.startswith(("path_rules.", "PATH_RULES.")):
            result.note(key, "not carried", "The native pipeline has no equivalent setting.")
    maximum, flood = result.config.get("rate_max", 20), result.config.get("rate_flood", 40)
    if flood < maximum:
        result.config["rate_max"] = flood
        result.note("rate_limiting.max_requests", "carried",
                    f"It's above flood_threshold ({flood}), so requests are flood-blocked first. The native limit is "
                    f"{flood}, which blocks at the same point.")
    return result


def resolve(config, framework: str, settings: Mapping[str, Any], **kwargs) -> dict:
    """An adapter's config: unchanged, or converted when it's "from_settings".

    ``{"from_settings": True, ...}`` converts and then applies the other keys on top.
    """
    if config == "from_settings":
        overrides = {}
    elif isinstance(config, Mapping) and config.get("from_settings"):
        overrides = {key: value for key, value in config.items() if key != "from_settings"}
    else:
        return dict(config or {})
    conversion = native_config_fastapi(settings, **kwargs) if framework == "fastapi" else \
        native_config(settings, framework, **kwargs)
    conversion.log()
    return {**conversion.config, **overrides}


_SETUP = {
    "django": ('In settings.py, replace AIWAF\'s middleware in MIDDLEWARE with\n'
               '    "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware",\n'
               'and add\n'
               '    AIWAF_CSV_PIPELINE = "from_settings"'),
    "flask": ('from aiwaf.flask import NativeAIWAF\n'
              'NativeAIWAF(app, "from_settings")'),
    "fastapi": ('from aiwaf.fast import NativeAIWAFMiddleware\n'
                'app.add_middleware(NativeAIWAFMiddleware, config="from_settings")'),
}
# Each framework's regular middleware, to tell a new install from an existing one.
_REGULAR_DJANGO = "aiwaf.django.middleware."


def describe_native_setup(conversion: NativeConversion, framework: str, *, middleware=None) -> str:
    """What ``aiwaf init`` prints: how to run the native pipeline, and what carries over.

    The native pipeline is the recommended setup for new installs. An existing install
    keeps its middleware; the report says what switching would change.
    """
    framework = "fastapi" if framework == "fast" else framework
    existing = framework == "django" and any(str(entry).startswith(_REGULAR_DJANGO) for entry in middleware or ())
    lines = ["", "Native pipeline (recommended for new installs): each request is checked in one aiwaf-rust call."]
    automatic = native_by_default(conversion)
    if automatic:
        lines.append("With these settings, AIWAF's regular setup runs as the native pipeline automatically. "
                     "Set AIWAF_NATIVE_PIPELINE = False to keep the regular middleware. To set it up explicitly:")
    elif existing:
        lines.append("This project uses the regular middleware, which keeps working. To switch:")
    lines += ["    " + line for line in _SETUP[framework].splitlines()]
    lines.append(conversion.report())
    if not _engine_supports("host_store"):
        lines.append("The installed aiwaf-rust can't use the regular stores or shared rate limits natively; "
                     "upgrade it first: pip install -U aiwaf-rust")
    return "\n".join(lines)


def native_access_log(conversion: NativeConversion, setting: str, log_dir) -> None:
    """Log natively where Flask's and FastAPI's trainers read access logs: a CSV in the log directory."""
    conversion.config["log_path"] = os.path.join(str(log_dir), "aiwaf_native_access.csv")
    conversion.notes = [note for note in conversion.notes if note.setting != setting]
    conversion.note(setting, "carried", "Requests are logged to aiwaf_native_access.csv in the same log directory, "
                                        "as CSV, which aiwaf's trainer reads like the regular access log.")


# Every native feature the regular middleware's settings can need. The native pipeline is the
# default only with an aiwaf-rust that has all of them.
DEFAULT_FEATURES = ("route_rule", "rate_overrides", "sql_monitor", "uuid_fact", "host_store", "rate_counter")
_OFF = {"false", "0", "off", "no"}


def _is_off(value) -> bool:
    return value is False or (isinstance(value, str) and value.strip().lower() in _OFF)


def native_by_default(conversion: NativeConversion, *, choice=None, environ: Mapping[str, str] | None = None,
                      framework: str | None = None) -> bool:
    """Whether the regular middleware should run as the native pipeline.

    It does when nothing turns it off (``choice``, from the app's ``AIWAF_NATIVE_PIPELINE`` or
    ``native_pipeline`` argument, then the environment variable), the installed aiwaf-rust has
    every native feature, and every setting carries over exactly. With ``framework``, the
    reason it doesn't is logged once, unless the app turned it off itself.
    """
    environ = os.environ if environ is None else environ
    if _is_off(choice) or (choice is None and _is_off(environ.get("AIWAF_NATIVE_PIPELINE"))):
        return False
    if not all(_engine_supports(feature) for feature in DEFAULT_FEATURES):
        _stays_regular(framework, "the installed aiwaf-rust is too old for it "
                                  "(pip install -U aiwaf-rust)." if _aiwaf_rust_installed() else
                                  "aiwaf-rust isn't installed (pip install aiwaf-rust).")
        return False
    if conversion.differences():
        _stays_regular(framework, "these settings don't carry over to it exactly:\n" + "\n".join(
            f"- {note.setting}: {note.outcome}. {note.detail}".rstrip() for note in conversion.differences()))
        return False
    # The native engine keeps files in its data directory (and request logs at log_path); the
    # regular middleware may never write there, so a directory it can't create keeps it.
    for directory in {str(conversion.config.get("data_dir") or "aiwaf_data"),
                      os.path.dirname(str(conversion.config.get("log_path") or "")) or None} - {None}:
        if not _writable_directory(directory):
            conversion.note("data_dir", "changed", f"The native pipeline keeps files in {os.path.abspath(directory)}, "
                                                   "which can't be created or written here.")
            _stays_regular(framework, f"it keeps files in {os.path.abspath(directory)}, which can't be created "
                                      "or written here.")
            return False
    return True


_REPORTED = set()


def _aiwaf_rust_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("aiwaf_rust") is not None


def _stays_regular(framework, reason, logger=LOGGER):
    """Say, once per process, why an app runs the regular middleware and not the native pipeline."""
    if framework is None or (framework, reason) in _REPORTED:
        return
    _REPORTED.add((framework, reason))
    logger.info("AIWAF: running the regular middleware for %s, not the native pipeline, because %s", framework, reason)


def log_native_fallback(framework: str, error, logger=LOGGER):
    """The native pipeline was chosen but couldn't start: the regular middleware runs instead."""
    logger.warning("AIWAF: the native pipeline couldn't start for %s (%s: %s), so the regular middleware runs "
                   "instead. Set AIWAF_NATIVE_PIPELINE to False to skip the native pipeline.",
                   framework, type(error).__name__, error)


def _writable_directory(directory) -> bool:
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return False
    return os.access(directory, os.W_OK | os.X_OK)


def is_rust_model(model) -> bool:
    """A model aiwaf-rust can use directly, as ``aiwaf train`` saves it with aiwaf-rust."""
    try:
        import aiwaf_rust
    except ImportError:
        return False
    forest = getattr(aiwaf_rust, "IsolationForest", None)
    return forest is not None and isinstance(model, forest)


def carry_model(conversion: NativeConversion, model, setting: str):
    """The regular middleware's loaded model, for the native pipeline, or a reported difference."""
    if model is None or is_rust_model(model):
        return model
    conversion.note(setting, "not carried", "The loaded model isn't an aiwaf-rust model, which the native "
                                            "pipeline needs. Retrain it with aiwaf-rust installed.")
    return None


def carry_learning(conversion: NativeConversion, framework: str, exclude) -> None:
    """Learn keywords from 404 probes by the regular anomaly middleware's rule, never learning ``exclude``."""
    if "ai_anomaly" not in conversion.config.get("enabled", ("ai_anomaly",)):
        return
    if not _engine_supports("learn_context"):
        conversion.note("AIAnomalyMiddleware", "changed", "The installed aiwaf-rust learns keywords by its own rule, "
                                                          "not the regular middleware's. Upgrade it: pip install -U aiwaf-rust")
        return
    conversion.config["learn_context"] = framework
    conversion.config["learn_exclude"] = sorted({str(word) for word in exclude})


def log_native_default(framework: str, logger=LOGGER):
    logger.info("AIWAF: running the native pipeline for %s, since every setting carries over to it. "
                "Set AIWAF_NATIVE_PIPELINE to False (or the environment variable) to keep the regular middleware.",
                framework)

