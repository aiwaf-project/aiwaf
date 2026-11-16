"""The regular Django middleware runs as the native pipeline when it can do so exactly."""
from django.core.exceptions import MiddlewareNotUsed

NATIVE_MIDDLEWARE = "aiwaf.django.csv_pipeline.NativeAIWAFMiddleware"


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
    apply_legacy_settings()
    names = {name: getattr(settings, name) for name in dir(settings) if name.startswith("AIWAF_")}
    conversion = native_config(names, "django", middleware=middleware)
    model = None
    if "ai_anomaly" in conversion.config.get("enabled", ()):
        from . import middleware as regular_middleware
        model = carry_model(conversion, regular_middleware.MODEL, "AIWAF_MODEL_PATH")
    if not native_by_default(conversion, choice=getattr(settings, "AIWAF_NATIVE_PIPELINE", None)):
        return None
    return regular[-1], dict(conversion.config), model


class NativeDefault:
    """Mixed into each regular middleware class: listed in MIDDLEWARE, it may run as native."""

    def __new__(cls, *args, **kwargs):
        from django.conf import settings

        path = f"{cls.__module__}.{cls.__qualname__}"
        if path in (getattr(settings, "MIDDLEWARE", None) or []):
            plan = native_plan()
            if plan is not None:
                host, config, model = plan
                if path != host:
                    raise MiddlewareNotUsed("AIWAF runs as the native pipeline")
                from aiwaf.core.native_settings import log_native_default
                from .csv_pipeline import NativeAIWAFMiddleware

                log_native_default("Django")
                return NativeAIWAFMiddleware(args[0] if args else kwargs.get("get_response"), config=config, model=model)
        return super().__new__(cls)
