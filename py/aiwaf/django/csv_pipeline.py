"""Django adapter for an explicitly configured Rust-owned CSV pipeline."""
import threading
import time
from collections import OrderedDict
from contextvars import ContextVar

from django.conf import settings
from django.http import HttpResponse
from django.urls import resolve, Resolver404
from aiwaf.core.csv_pipeline import CSVPipeline, validate_fact_callback, fact_value
from aiwaf.core.honeypot import is_authenticated_session_context
from aiwaf.core.request_context import extract_ip_from_django_request
from .geoip import lookup_country
from .honeypot_cache import DjangoHoneypotCache
from aiwaf.core.csv_pipeline import PipelineStores


# The request being handled, so a block written to the regular store carries its extended info.
_REQUEST = ContextVar("aiwaf_native_request", default=None)


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
        from .middleware import _get_blocklist_extended_info

        # As the regular middleware does: the request's headers, when
        # AIWAF_BLOCKLIST_STORE_EXTENDED_INFO is on.
        request = _REQUEST.get()
        info = _get_blocklist_extended_info(request) if request is not None else None
        BlocklistManager.block(ip, reason, extended_request_info=info)

    def learn(self, keywords):
        from .storage import get_keyword_store

        store = get_keyword_store()
        for keyword in keywords:
            store.add_keyword(keyword)


def _process_local(alias):
    """Whether a Django cache keeps its values in this process only."""
    from django.core.cache import caches
    from django.core.cache.backends.locmem import LocMemCache

    try:
        # Exactly LocMemCache: a subclass may keep its values elsewhere.
        return type(caches[alias]) is LocMemCache
    except Exception:  # An alias that doesn't exist is reported where the cache is used.
        return False


_RESOLVED = OrderedDict()
_RESOLVED_LOCK = threading.Lock()
_RESOLVED_LIMIT = 4096


def _resolved(path, urlconf):
    """The view, whether it exists, and its uuid parameter, for a path; recent paths are cached.

    Resolution depends only on the URLconf's resolver, the active language (i18n_patterns) and
    the path, so those are the key. A changed URLconf has a new resolver, so a new key.
    """
    from django.urls import get_resolver
    from django.utils import translation

    key = (get_resolver(urlconf), translation.get_language(), path)
    with _RESOLVED_LOCK:
        hit = _RESOLVED.get(key)
        if hit is not None:
            _RESOLVED.move_to_end(key)
            return hit
    try:
        match = resolve(path, urlconf=urlconf)
        # The regular UUID middleware reads the route's uuid parameter.
        result = (match.func, True, match.kwargs.get("uuid"))
    except Resolver404:
        result = (None, False, None)
    with _RESOLVED_LOCK:
        _RESOLVED[key] = result
        while len(_RESOLVED) > _RESOLVED_LIMIT:
            _RESOLVED.popitem(last=False)
    return result


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
        rate_alias = config.pop("rate_cache_alias", "default")
        # Saves each request the anomaly stage checks to RequestLog, as the regular anomaly
        # middleware does, for aiwaf train when there is no access log to read.
        self.save_request_log = bool(config.pop("save_request_log", False))
        # With the regular logger middleware beside it: the stages listed before the logger,
        # whose request blocks the logger never saw.
        self.logger_outer = frozenset(config.pop("logger_outer_stages", ()) or ())
        # A LocMemCache keeps its values in this process only, as the engine's own memory does, so
        # timing and rate counts behave the same there without a Django cache call per request.
        if config["honeypot_backend"] == "django_cache" and _process_local(alias):
            config["honeypot_backend"] = "memory"
        if config.get("rate_backend") == "django_cache" and _process_local(rate_alias):
            config["rate_backend"] = "memory"
        cache = DjangoHoneypotCache(alias, validate_backend=active) if config["honeypot_backend"] == "django_cache" else None
        # "django_cache" counts rate limits in a Django cache, as the regular rate limiter does.
        counter = None
        if config.get("rate_backend") == "django_cache":
            from .rate_cache import DjangoRateCounter
            counter = DjangoRateCounter(rate_alias)
            config["rate_backend"] = "counter"
        self.pipeline = CSVPipeline(config, honeypot_cache=cache, rate_counter=counter,
                                    stores=DjangoStores() if store == "django" else None)
        self.country_lookup = getattr(settings, "AIWAF_CSV_PIPELINE_COUNTRY_LOOKUP", None)
        # The GeoIP lookup cache, as the regular geo middleware reads it.
        self.geo_cache = (getattr(settings, "AIWAF_GEO_CACHE_PREFIX", "aiwaf:geo:"),
                          getattr(settings, "AIWAF_GEO_CACHE_SECONDS", 3600))
        validate_fact_callback(self.country_lookup)
        model = model if model is not None else getattr(settings, "AIWAF_CSV_PIPELINE_MODEL", None)
        if model is not None:
            self.pipeline.native.set_model(model)

    def __call__(self, request):
        token = _REQUEST.set(request)
        try:
            return self._handle(request)
        finally:
            _REQUEST.reset(token)

    def _handle(self, request):
        start = time.time()
        endpoint, exists, uuid = _resolved(request.path_info, getattr(request, "urlconf", None))
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
            country=(fact_value(lookup, ip) if lookup else lookup_country(ip, cache_prefix=self.geo_cache[0], cache_seconds=self.geo_cache[1])) if self.pipeline.needs_country else "", exhausted=exhausted,
            authenticated=is_authenticated_session_context(getattr(request, "user", None), getattr(request, "session", None)),
            uuid=None if uuid is None else str(uuid))
        stage = getattr(ctx, "blocked_stage", None) if status else None
        if stage == "uuid_tamper":
            # The regular UUID check blocks in process_view, after every middleware's request
            # hook has run, so the block passes back out through all of them.
            request._aiwaf_view_block = (payload, status)
            response = self.get_response(request)
        elif status:
            if stage in self.logger_outer:
                request._aiwaf_skip_log = True
            response = HttpResponse(payload, status=status, content_type="application/json")
        else:
            response = self.get_response(request)
        final_status, final_payload = self.pipeline.finish(ctx, response.status_code, response.get("Content-Length", "-"))
        if final_payload != "null":
            if getattr(ctx, "blocked_stage", None) in self.logger_outer:
                # Decided after the response: a logger listed after that stage saw the response.
                request._aiwaf_logged_response = response
            return HttpResponse(final_payload, status=final_status, content_type="application/json")
        if self.save_request_log and not status:
            self._save_request_log(request, response, ip, endpoint, start)
        return response

    def process_view(self, request, view_func, view_args, view_kwargs):
        """A UUID block decided in __call__, returned here, where the regular check returns it."""
        block = getattr(request, "_aiwaf_view_block", None)
        if block is not None:
            return HttpResponse(block[0], status=block[1], content_type="application/json")
        return None

    def _save_request_log(self, request, response, ip, endpoint, start):
        """The row the regular anomaly middleware saves for a request it let through."""
        if "ai_anomaly" in self.pipeline.disabled_stages(request.path, endpoint):
            return
        try:
            from .models import RequestLog
            from .utils import is_exempt, is_ip_exempted

            if is_exempt(request) or is_ip_exempted(ip):
                return
            RequestLog.objects.create(
                ip_address=ip,
                method=request.method,
                path=request.path[:500],
                status_code=int(getattr(response, "status_code", 0) or 0),
                response_time=float(time.time() - start),
                user_agent=request.META.get("HTTP_USER_AGENT", "")[:1000],
                referer=request.META.get("HTTP_REFERER", "")[:500],
                content_length=str(response.get("Content-Length", "-"))[:20],
            )
        except Exception:
            pass  # As in the regular middleware: logging never breaks the response.
