"""Django adapter for an explicitly configured Rust-owned CSV pipeline."""
from django.conf import settings
from django.http import HttpResponse
from django.urls import resolve, Resolver404
from aiwaf.core.csv_pipeline import CSVPipeline, validate_fact_callback, fact_value
from aiwaf.core.honeypot import is_authenticated_session_context
from aiwaf.core.request_context import extract_ip_from_django_request
from .geoip import lookup_country
from .honeypot_cache import DjangoHoneypotCache
from aiwaf.core.csv_pipeline import PipelineStores


class DjangoStores(PipelineStores):
    """The regular middleware's stores, following AIWAF_STORAGE_MODE (models or csv)."""

    def is_blocked(self, ip):
        from .blocklist_manager import BlocklistManager

        return BlocklistManager.is_blocked(ip)

    def is_exempted(self, ip, path):
        from .storage import get_exemption_store, get_path_exemption_store

        return bool(get_exemption_store().is_exempted(ip) or get_path_exemption_store().is_exempted(path))

    def top_keywords(self, n):
        from .storage import get_keyword_store

        return get_keyword_store().get_top_keywords(n)

    def blocked_countries(self):
        from .storage import _is_csv_mode, _ensure_runtime_csv_backend

        countries = []
        try:
            from .models import GeoBlockedCountry

            countries = list(GeoBlockedCountry.objects.values_list("country_code", flat=True))
        except Exception:  # Unmigrated or unavailable tables, as in the regular geo middleware.
            countries = []
        if _is_csv_mode():
            from ..core.runtime_storage import get_geo_block_store

            _ensure_runtime_csv_backend()
            countries += list(get_geo_block_store().get_countries())
        return countries

    def block(self, ip, reason):
        from .blocklist_manager import BlocklistManager

        BlocklistManager.block(ip, reason)

    def learn(self, keywords):
        from .storage import get_keyword_store

        store = get_keyword_store()
        for keyword in keywords:
            store.add_keyword(keyword)


class NativeAIWAFMiddleware:
    def __init__(self, get_response, *, config=None, model=None):
        """``config`` and ``model`` are given when the regular middleware runs as this one."""
        self.get_response = get_response
        from aiwaf.core.native_settings import resolve
        from .settings_compat import apply_legacy_settings

        if config is None:
            # AIWAF_SETTINGS, the legacy nested form, maps onto the flat settings read below.
            apply_legacy_settings()
            names = {name: getattr(settings, name) for name in dir(settings) if name.startswith("AIWAF_")}
            config = resolve(getattr(settings, "AIWAF_CSV_PIPELINE", {}) or {}, "django", names,
                             middleware=getattr(settings, "MIDDLEWARE", None))
        config = dict(config)
        config.setdefault("honeypot_backend", "django_cache")
        alias = config.pop("honeypot_cache_alias", "default")
        # "django" reads and writes the regular middleware's stores instead of native CSV files.
        store = config.pop("store", "csv")
        if store not in ("csv", "django"):
            raise ValueError('AIWAF_CSV_PIPELINE "store" must be "csv" or "django"')
        active = "honeypot" in config.get("enabled", {"honeypot"})
        cache = DjangoHoneypotCache(alias, validate_backend=active) if config["honeypot_backend"] == "django_cache" else None
        # "django_cache" counts rate limits in a Django cache, as the regular rate limiter does.
        counter = None
        rate_alias = config.pop("rate_cache_alias", "default")
        if config.get("rate_backend") == "django_cache":
            from .rate_cache import DjangoRateCounter
            counter = DjangoRateCounter(rate_alias)
            config["rate_backend"] = "counter"
        self.pipeline = CSVPipeline(config, honeypot_cache=cache, rate_counter=counter,
                                    stores=DjangoStores() if store == "django" else None)
        self.country_lookup = getattr(settings, "AIWAF_CSV_PIPELINE_COUNTRY_LOOKUP", None)
        validate_fact_callback(self.country_lookup)
        model = model if model is not None else getattr(settings, "AIWAF_CSV_PIPELINE_MODEL", None)
        if model is not None:
            self.pipeline.native.set_model(model)

    def __call__(self, request):
        endpoint, exists, uuid = None, False, None
        try:
            match = resolve(request.path_info, urlconf=getattr(request, "urlconf", None))
            endpoint, exists = match.func, True
            # The regular UUID middleware reads the route's uuid parameter.
            uuid = match.kwargs.get("uuid")
        except Resolver404:
            pass
        # Django function views do not declare their supported methods. Preserve
        # conservative acceptance; class-based views expose http_method_names.
        methods = {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE", "TRACE", "CONNECT"}
        view_class = getattr(endpoint, "view_class", None)
        if view_class:
            methods = {m.upper() for m in view_class.http_method_names if hasattr(view_class, m)}
        ip = extract_ip_from_django_request(request)
        lookup = self.country_lookup
        query, body, exhausted = self.pipeline.capture_wsgi(request, django=True)
        ctx, status, payload = self.pipeline.begin(ip=ip, path=request.path, method=request.method,
            headers={k: str(v) for k, v in request.META.items() if k.startswith("HTTP_") or k in {"CONTENT_TYPE", "CONTENT_LENGTH", "SERVER_PROTOCOL"}},
            query=query, body=body, exists=exists, methods=methods, endpoint=endpoint,
            country=(fact_value(lookup, ip) if lookup else lookup_country(ip, cache_prefix="aiwaf_geo", cache_seconds=3600)) if self.pipeline.needs_country else "", exhausted=exhausted,
            authenticated=is_authenticated_session_context(getattr(request, "user", None), getattr(request, "session", None)),
            uuid=None if uuid is None else str(uuid))
        response = HttpResponse(payload, status=status, content_type="application/json") if status else self.get_response(request)
        final_status, final_payload = self.pipeline.finish(ctx, response.status_code, response.get("Content-Length", "-"))
        if final_payload != "null":
            return HttpResponse(final_payload, status=final_status, content_type="application/json")
        return response
