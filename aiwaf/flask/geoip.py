import logging
import os
import time

from aiwaf.core import geoip as core_geoip
from aiwaf.core.geoip import GEOIP_AVAILABLE

_geoip_cache = {}


def _cache_get(cache_key):
    cached = _geoip_cache.get(cache_key)
    if not cached:
        return None
    value, expires_at = cached
    if expires_at and expires_at < time.time():
        _geoip_cache.pop(cache_key, None)
        return None
    return value


def _cache_set(cache_key, value, timeout):
    expires_at = time.time() + timeout if timeout else None
    _geoip_cache[cache_key] = (value, expires_at)


def lookup_country(ip, cache_prefix=None, cache_seconds=3600, db_path=None):
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core",
        "geolock",
        "ipinfo_lite.mmdb",
    )
    db_path = db_path or default_path
    
    return core_geoip.lookup_country(
        ip, 
        db_path, 
        cache_prefix=cache_prefix, 
        cache_seconds=cache_seconds, 
        cache_get=_cache_get, 
        cache_set=_cache_set
    )


def lookup_country_name(ip, cache_prefix=None, cache_seconds=3600, db_path=None):
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core",
        "geolock",
        "ipinfo_lite.mmdb",
    )
    db_path = db_path or default_path

    return core_geoip.lookup_country_name(
        ip, 
        db_path, 
        cache_prefix=cache_prefix, 
        cache_seconds=cache_seconds, 
        cache_get=_cache_get, 
        cache_set=_cache_set
    )


def get_country_for_ip(ip, app_config):
    prefix = app_config.get("AIWAF_GEO_CACHE_PREFIX", "aiwaf_geo")
    cache_seconds = app_config.get("AIWAF_GEO_CACHE_SECONDS", 3600)
    db_path = app_config.get("AIWAF_GEOIP_DB_PATH")
    return lookup_country(ip, cache_prefix=f"{prefix}:", cache_seconds=cache_seconds, db_path=db_path)
