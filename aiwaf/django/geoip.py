import os

from django.conf import settings
from django.core.cache import cache

from aiwaf.core import geoip as core_geoip
from aiwaf.core.geoip import GEOIP_AVAILABLE

def _cache_get(cache_key):
    try:
        return cache.get(cache_key)
    except Exception:
        return None

def _cache_set(cache_key, value, timeout):
    try:
        cache.set(cache_key, value, timeout=timeout)
    except Exception:
        return

def lookup_country(ip, cache_prefix=None, cache_seconds=3600):
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core",
        "geolock",
        "ipinfo_lite.mmdb",
    )
    if getattr(settings, "configured", False):
        db_path = getattr(settings, "AIWAF_GEOIP_DB_PATH", default_path)
    else:
        db_path = default_path
        
    return core_geoip.lookup_country(
        ip, 
        db_path, 
        cache_prefix=cache_prefix, 
        cache_seconds=cache_seconds, 
        cache_get=_cache_get, 
        cache_set=_cache_set
    )

def lookup_country_name(ip, cache_prefix=None, cache_seconds=3600):
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "core",
        "geolock",
        "ipinfo_lite.mmdb",
    )
    if getattr(settings, "configured", False):
        db_path = getattr(settings, "AIWAF_GEOIP_DB_PATH", default_path)
    else:
        db_path = default_path

    return core_geoip.lookup_country_name(
        ip, 
        db_path, 
        cache_prefix=cache_prefix, 
        cache_seconds=cache_seconds, 
        cache_get=_cache_get, 
        cache_set=_cache_set
    )
