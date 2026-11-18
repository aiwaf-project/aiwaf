"""Flask adapter for the complete Rust-owned CSV request pipeline."""
from flask import g, request, Response
from aiwaf.core.csv_pipeline import CSVPipeline, environ_headers, validate_fact_callback, fact_value, PipelineStores
from aiwaf.core.request_context import extract_ip_from_flask_request
from .geoip import get_country_for_ip


class FlaskStores(PipelineStores):
    """The regular Flask middleware's stores, following AIWAF_STORAGE_MODE. Used in a request."""

    def is_blocked(self, ip):
        from .blocklist_manager import BlocklistManager

        return BlocklistManager.is_blocked(ip)

    def is_exempted(self, ip, path):
        # The regular check: allowlisted IPs, and configured or stored paths, patterns included.
        from .storage import is_ip_allowlisted
        from .utils import is_path_exempt

        return bool(is_ip_allowlisted(ip) or is_path_exempt(path))

    def top_keywords(self, n):
        from .storage import get_top_keywords

        return get_top_keywords(n)

    def blocked_countries(self):
        from .storage import get_geo_blocked_countries

        return list(get_geo_blocked_countries())

    def block(self, ip, reason):
        from .blocklist_manager import BlocklistManager

        BlocklistManager.block(ip, reason)

    def learn(self, keywords):
        from .storage import add_keyword

        for keyword in keywords:
            add_keyword(keyword)


class NativeAIWAF:
    def __init__(self, app, config=None, *, model=None, country_lookup=None, authenticated=None, honeypot_cache=None):
        from aiwaf.core import profiles

        if profiles.activate(app.config.get(profiles.PROFILE_SETTING)) == "monitor":
            raise RuntimeError("AIWAF_PROFILE=monitor runs the regular middleware: use AIWAF(app) or the regular "
                               "middleware instead of NativeAIWAF.")
        validate_fact_callback(country_lookup)
        validate_fact_callback(authenticated)
        if app.extensions.get("aiwaf") or app.extensions.get("aiwaf_csv_pipeline"):
            raise RuntimeError("Install one AIWAF chain per app")
        from aiwaf.core.native_settings import resolve

        config = dict(resolve(config, "flask", app.config))
        # "flask" reads and writes the regular middleware's stores instead of native CSV files.
        store = config.pop("store", "csv")
        if store not in ("csv", "flask"):
            raise ValueError('The native "store" must be "csv" or "flask"')
        self.pipeline = CSVPipeline(config, model=model, honeypot_cache=honeypot_cache,
                                    stores=FlaskStores() if store == "flask" else None)
        app.extensions["aiwaf_csv_pipeline"] = self

        @app.before_request
        def before_request():
            endpoint = app.view_functions.get(request.endpoint)
            methods = {"GET", "POST", "HEAD", "OPTIONS", "PUT", "PATCH", "DELETE", "TRACE", "CONNECT"}
            # Routing is framework work. MethodNotAllowed still describes a known path.
            exists = request.url_rule is not None
            if request.url_rule is not None:
                methods = set(request.url_rule.methods or methods)
            elif request.routing_exception is not None:
                valid = getattr(request.routing_exception, "valid_methods", None)
                if valid:
                    methods, exists = set(valid), True
            ip = extract_ip_from_flask_request(request)
            query, body, exhausted = self.pipeline.capture_wsgi(request)
            ctx, status, payload = self.pipeline.begin(ip=ip, path=request.path, method=request.method,
                headers=environ_headers(request.headers, request.environ.get("SERVER_PROTOCOL", "HTTP/1.1")),
                query=query, body=body, exists=exists, methods=methods, endpoint=endpoint,
                country=(fact_value(country_lookup, ip) if country_lookup else get_country_for_ip(ip, app.config)) if self.pipeline.needs_country else "",
                authenticated=fact_value(authenticated, request) if authenticated else False, exhausted=exhausted)
            g._aiwaf_pipeline_context = ctx
            if status:
                # For the regular logging middleware, when it logs beside the native pipeline.
                g.aiwaf_blocked, g.aiwaf_block_reason = True, "blocked"
                return Response(payload, status=status, mimetype="application/json")

        @app.after_request
        def after_request(response):
            ctx = getattr(g, "_aiwaf_pipeline_context", None)
            if ctx is None:
                return response
            # Clear before finalization so an error response cannot finalize twice.
            g._aiwaf_pipeline_context = None
            status, payload = self.pipeline.finish(ctx, response.status_code, response.content_length or "-")
            if payload != "null":
                return Response(payload, status=status, mimetype="application/json")
            return response
