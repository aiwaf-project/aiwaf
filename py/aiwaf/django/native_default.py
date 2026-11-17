"""The regular Django middleware runs as the native pipeline when it can do so exactly."""
from django.core.exceptions import MiddlewareNotUsed

NATIVE_MIDDLEWARE = "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware"
# Decide which model the regular anomaly middleware uses; that model is handed to the native pipeline.
_MODEL_SETTINGS = {"AIWAF_MODEL_PATH", "AIWAF_MODEL_STORAGE", "AIWAF_MODEL_STORAGE_FALLBACK", "AIWAF_MODEL_CACHE_KEY",
                   "AIWAF_MODEL_CACHE_TIMEOUT", "AIWAF_ALLOW_CUSTOM_MODEL_PATH", "AIWAF_DISABLE_AI",
                   "AIWAF_MIN_AI_LOGS", "AIWAF_ACCESS_LOG"}
# Settings whose native pipeline couldn't start, so Django's later middleware stays regular too.
_FAILED = set()


def _model_in_use():
    """Whether the regular anomaly middleware keeps its model: it drops it without enough logs."""
    from functools import partial
    from types import SimpleNamespace
    from django.conf import settings
    from .middleware import AIAnomalyMiddleware

    regular = SimpleNamespace(min_ai_logs=getattr(settings, "AIWAF_MIN_AI_LOGS", 10000),
                              _count_log_lines=partial(AIAnomalyMiddleware._count_log_lines, None))
    return AIAnomalyMiddleware._check_ai_log_sufficiency(regular)[0]


def native_plan():
    """Which MIDDLEWARE entry hosts the native pipeline, with its config and model, or None.

    The native pipeline takes the place of the last regular AIWAF entry, so the middleware
    before it (sessions, authentication) has run, as it had for every regular stage.
    """
    from django.conf import settings
    from aiwaf.core.native_settings import _DJANGO_STAGES, carry_model, native_by_default, native_config
    from .settings_compat import apply_legacy_settings

    middleware = [str(entry) for entry in getattr(settings, "MIDDLEWARE", None) or []]
    regular = [entry for entry in middleware if entry in _DJANGO_STAGES]
    if not regular or NATIVE_MIDDLEWARE in middleware:
        return None
    from aiwaf.core.native_settings import DJANGO_LOGGER
    apply_legacy_settings()
    names = {name: getattr(settings, name) for name in dir(settings) if name.startswith("AIWAF_")}
    conversion = native_config(names, "django", middleware=middleware)
    from .csv_pipeline import _process_local

    # A process-local cache isn't called: native keeps honeypot timing in its own memory then.
    if "honeypot" in conversion.config.get("enabled", ()) and not _process_local("default"):
        # Native honeypot timing needs a cache that keeps values and reports errors; the regular
        # middleware also runs with DummyCache or ignored cache errors.
        from django.core.exceptions import ImproperlyConfigured
        from .honeypot_cache import DjangoHoneypotCache

        try:
            DjangoHoneypotCache("default", validate_backend=True)
        except ImproperlyConfigured as error:
            conversion.note("CACHES", "not carried", f"{error}.")
    # The regular keyword checks never learn the app's own route words, AIWAF_ALLOWED_PATH_KEYWORDS
    # or AIWAF_EXEMPT_KEYWORDS, read here as they read them.
    if {"ip_keyword_block", "ai_anomaly"} & set(conversion.config.get("enabled", ())):
        from .trainer import get_legitimate_keywords

        conversion.config["legitimate_keywords"] = sorted(str(keyword) for keyword in get_legitimate_keywords())
        # Keyword blocks skip paths under the app's own route prefixes, as in the regular check.
        from .middleware import IPAndKeywordBlockMiddleware

        conversion.config["safe_prefixes"] = sorted(
            str(prefix) for prefix in IPAndKeywordBlockMiddleware._collect_safe_prefixes(None) if prefix)
    conversion.notes = [note for note in conversion.notes if note.setting != "AIWAF_ALLOWED_PATH_KEYWORDS"]
    if "ai_anomaly" in conversion.config.get("enabled", ()):
        # The regular anomaly middleware never learns its static keywords or the legitimate ones.
        from aiwaf.core.native_settings import carry_learning

        learning = getattr(settings, "AIWAF_ENABLE_KEYWORD_LEARNING", True)
        carry_learning(conversion, "django", list(conversion.config.get("static_keywords", ()))
                       + (list(conversion.config.get("legitimate_keywords", ())) if learning else []))
    # The model settings are decided here, from the model the regular middleware would use.
    conversion.notes = [note for note in conversion.notes if note.setting not in _MODEL_SETTINGS]
    model = None
    if "ai_anomaly" in conversion.config.get("enabled", ()):
        from . import middleware as regular_middleware
        if regular_middleware.MODEL is not None and _model_in_use():
            model = carry_model(conversion, regular_middleware.MODEL, "AIWAF_MODEL_PATH")
        # With no access log, aiwaf train reads the requests the regular anomaly middleware saves
        # to RequestLog, so the native pipeline saves them too, unless the logger middleware does.
        logger_saves = bool(getattr(settings, "AIWAF_MIDDLEWARE_LOGGING", False) and getattr(settings, "AIWAF_MIDDLEWARE_DB", True))
        if not logger_saves and not _access_log_to_train_from():
            conversion.config["save_request_log"] = True
    if not native_by_default(conversion, choice=getattr(settings, "AIWAF_NATIVE_PIPELINE", None), framework="Django"):
        return None
    if _plan_key(middleware, conversion.config) in _FAILED:
        return None
    # The logger middleware stays regular when it logs beside the native pipeline; the native
    # pipeline takes the place of the last other AIWAF entry.
    keep = {DJANGO_LOGGER} if DJANGO_LOGGER in regular and "logging" not in conversion.config.get("enabled", ()) \
        and getattr(settings, "AIWAF_MIDDLEWARE_LOGGING", False) else set()
    hosts = [entry for entry in regular if entry not in keep]
    if not hosts:
        return None
    return hosts[-1], dict(conversion.config), model, keep


def _access_log_to_train_from():
    """Whether ``aiwaf train`` has log files to read, so it doesn't need requests saved to RequestLog."""
    import glob
    import os
    from django.conf import settings

    path = getattr(settings, "AIWAF_ACCESS_LOG", None)
    if not path:
        return False
    return any(os.path.getsize(file) > 0 for file in [str(path), *glob.glob(f"{path}.*")] if os.path.isfile(file))


def _plan_key(middleware, config):
    return repr((tuple(middleware), sorted(config.items(), key=lambda item: item[0])))


class NativeDefault:
    """Mixed into each regular middleware class: listed in MIDDLEWARE, it may run as native."""

    def __new__(cls, *args, **kwargs):
        from django.conf import settings

        path = f"{cls.__module__}.{cls.__qualname__}"
        if path in (getattr(settings, "MIDDLEWARE", None) or []):
            plan = native_plan()
            if plan is not None:
                host, config, model, keep = plan
                if path in keep:
                    return super().__new__(cls)
                if path != host:
                    raise MiddlewareNotUsed("AIWAF runs as the native pipeline")
                from aiwaf.core.native_settings import log_native_default, log_native_fallback
                from .csv_pipeline import NativeAIWAFMiddleware

                try:
                    native = NativeAIWAFMiddleware(args[0] if args else kwargs.get("get_response"), config=config,
                                                   model=model)
                except Exception as error:
                    # Django builds MIDDLEWARE last entry first, so the host is built before the
                    # other AIWAF entries: they see the failure and stay regular as well.
                    _FAILED.add(_plan_key([str(entry) for entry in settings.MIDDLEWARE], config))
                    log_native_fallback("Django", error)
                else:
                    log_native_default("Django")
                    return native
        return super().__new__(cls)
