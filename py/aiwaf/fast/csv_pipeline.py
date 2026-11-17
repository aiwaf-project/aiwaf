"""Pure ASGI adapter: inspect a bounded body prefix and replay exact messages."""
from functools import partial
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from aiwaf.core.csv_pipeline import CSVPipeline, environ_headers, validate_fact_callback, fact_value, PipelineStores
from aiwaf.core.request_context import resolve_ip_from_fastapi_request
from .geoip import get_country_for_ip

_ANY_METHOD = {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE", "TRACE", "CONNECT"}


def route_facts(router, scope):
    """Resolve mounts without treating every descendant as an existing endpoint."""
    methods, exists = set(), False
    for route in getattr(router, "routes", ()):
        match, child = route.matches(scope)
        if match not in (Match.FULL, Match.PARTIAL):
            continue
        if hasattr(route, "routes"):
            endpoint, nested_exists, nested_methods = route_facts(route, {**scope, **child})
            if nested_exists:
                return endpoint, nested_exists, nested_methods
            continue
        exists = True
        methods.update(getattr(route, "methods", None) or _ANY_METHOD)
        if match == Match.FULL:
            return child.get("endpoint"), True, methods
    return None, exists, methods if exists else _ANY_METHOD


class FastAPIStores(PipelineStores):
    """The regular FastAPI middleware's stores, set up from its AIWAFConfig as it sets them up."""

    def __init__(self, aiwaf_config=None, *, ready=False):
        """``ready``: AIWAF(app) has already set up the stores, so they're used as they are."""
        from aiwaf.core.runtime_config import get_config
        from aiwaf.core.runtime_storage import get_exemption_store, initialize_storage

        if ready:
            return
        config = aiwaf_config if aiwaf_config is not None else get_config()
        initialize_storage(**config.get_storage_config())
        exemptions = config.get("exemptions", {})
        if exemptions.get("private_ips_exempted", True):
            # The regular middleware exempts these private and local patterns at startup.
            store = get_exemption_store()
            for pattern in exemptions.get("auto_exempt_patterns", []):
                store.add_pattern(pattern, "Auto-exempt private IPs")

    def is_blocked(self, ip):
        from aiwaf.core.runtime_blocklist import BlocklistManager

        return BlocklistManager.is_blocked(ip)

    def is_exempted(self, ip, path):
        from aiwaf.core.runtime_storage import get_exemption_store

        return bool(get_exemption_store().is_exempted(ip))

    def top_keywords(self, n):
        from aiwaf.core.runtime_storage import get_keyword_store

        return get_keyword_store().get_top_keywords(n)

    def blocked_countries(self):
        from aiwaf.core.runtime_storage import get_geo_block_store

        return list(get_geo_block_store().get_countries())

    def block(self, ip, reason):
        from aiwaf.core.runtime_blocklist import BlocklistManager

        BlocklistManager.block(ip, reason)

    def learn(self, keywords):
        from aiwaf.core.runtime_storage import get_keyword_store

        store = get_keyword_store()
        for keyword in keywords:
            store.add_keyword(keyword)


class NativeAIWAFMiddleware:
    def __init__(self, app, config=None, *, model=None, country_lookup=None, authenticated=None, honeypot_cache=None,
                 aiwaf_config=None, stores_ready=False, source_app=None):
        """``config="from_settings"`` converts ``aiwaf_config``, or the AIWAFConfig the regular middleware uses.

        ``stores_ready``: AIWAF(app) set up the stores already. Setting them up again when the app
        builds its middleware, at its first request, would empty in-memory stores.
        """
        from aiwaf.core.native_settings import resolve

        validate_fact_callback(country_lookup)
        validate_fact_callback(authenticated)
        self.app = app
        from_settings = config == "from_settings" or isinstance(config, dict) and config.get("from_settings")
        if from_settings and aiwaf_config is None:
            from aiwaf.core.runtime_config import get_config
            aiwaf_config = get_config()
        config = dict(resolve(config, "fastapi", aiwaf_config))
        # "fastapi" reads and writes the regular middleware's stores instead of native CSV files.
        store = config.pop("store", "csv")
        if store not in ("csv", "fastapi"):
            raise ValueError('The native "store" must be "csv" or "fastapi"')
        if aiwaf_config is not None:
            from .runtime_policy import load_trained_model, route_keywords
            from .middleware.ip_and_keyword_block_middleware import DEFAULT_LEGITIMATE

            if model is None and "ai_anomaly" in config.get("enabled", ("ai_anomaly",)):
                model = load_trained_model(aiwaf_config)
            if source_app is not None or from_settings:
                keywords, prefixes = route_keywords(source_app if source_app is not None else app)
                config["legitimate_keywords"] = sorted(DEFAULT_LEGITIMATE | keywords)
                config["safe_prefixes"] = sorted(prefixes)
        self.pipeline = CSVPipeline(config, model=model, honeypot_cache=honeypot_cache,
                                    stores=FastAPIStores(aiwaf_config, ready=stores_ready) if store == "fastapi" else None)
        self.cache_io = self.pipeline.profile.get("honeypot_backend") == "cache"
        self.country_lookup = country_lookup
        self.authenticated = authenticated

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope, receive=receive)
        router = getattr(scope.get("app"), "router", None)
        endpoint, exists, methods = route_facts(router, scope)
        ip = resolve_ip_from_fastapi_request(request)
        query, body, exhausted = await self.pipeline.capture_asgi(request)
        begin = partial(self.pipeline.begin, ip=ip, path=scope.get("path", "/"), method=request.method,
            headers=environ_headers(request.headers, "HTTP/" + scope.get("http_version", "1.1")),
            query=query, body=body, exists=exists, methods=methods, endpoint=endpoint,
            country=(fact_value(self.country_lookup, ip) if self.country_lookup else get_country_for_ip(ip)) if self.pipeline.needs_country else "",
            authenticated=fact_value(self.authenticated, request) if self.authenticated else False, exhausted=exhausted)
        ctx, status, payload = await run_in_threadpool(begin) if self.cache_io else begin()
        if status:
            self.pipeline.finish(ctx, status, len(payload.encode()))
            return await Response(payload, status_code=status, media_type="application/json")(scope, request.receive, send)
        replaced = False
        started = False

        async def checked_send(message):
            nonlocal replaced, started
            if message["type"] == "http.response.start":
                if started:
                    raise RuntimeError("ASGI application started its response twice")
                started = True
                headers = dict(message.get("headers", ()))
                final_status, final_payload = self.pipeline.finish(ctx, message["status"], headers.get(b"content-length", b"-").decode("latin-1"))
                if final_payload != "null":
                    replaced = True
                    return await Response(final_payload, status_code=final_status, media_type="application/json")(scope, request.receive, send)
            if not replaced:
                await send(message)

        try:
            await self.app(scope, request.receive, checked_send)
        except Exception:
            if not started:
                self.pipeline.finish(ctx, 500)
            raise
