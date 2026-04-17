# Flask GeoBlockMiddleware
from flask import request, jsonify
from .utils import get_ip, is_exempt
from .blacklist_manager import BlacklistManager
from .exemption_decorators import should_apply_middleware
from .geoip import get_country_for_ip
from .storage import get_geo_blocked_countries


def _normalize_country_list(value):
    if not value:
        return set()
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    normalized = set()
    for item in values:
        if item:
            normalized.add(str(item).strip().upper())
    return normalized


class GeoBlockMiddleware:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        @app.before_request
        def before_request():
            if not should_apply_middleware('geo_block'):
                return None

            if is_exempt(request):
                return None

            if not app.config.get('AIWAF_GEO_BLOCK_ENABLED', False):
                return None

            allow_countries = _normalize_country_list(
                app.config.get('AIWAF_GEO_ALLOW_COUNTRIES', [])
            )
            block_countries = _normalize_country_list(
                app.config.get('AIWAF_GEO_BLOCK_COUNTRIES', [])
            )
            dynamic_blocked = _normalize_country_list(get_geo_blocked_countries())

            if not allow_countries and not block_countries and not dynamic_blocked:
                return None

            ip = get_ip()
            if not ip:
                return None

            country = get_country_for_ip(ip, app.config)
            if not country:
                return None

            country = country.strip().upper()
            blocked = False

            if allow_countries:
                blocked = country not in allow_countries
            else:
                blocked = country in block_countries or country in dynamic_blocked

            if blocked:
                reason = f"Geo blocked: {country}"
                BlacklistManager.block(ip, reason)

                logger = getattr(app, 'aiwaf_logger', None)
                if logger:
                    logger.mark_request_blocked(reason)

                return jsonify({"error": "blocked"}), 403

            return None
