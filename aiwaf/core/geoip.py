import logging
import os

try:
    from geoip2.database import Reader as GeoIPReader
    from geoip2.errors import AddressNotFoundError
    GEOIP_AVAILABLE = True
except ImportError:
    GeoIPReader = None
    AddressNotFoundError = Exception
    GEOIP_AVAILABLE = False

logger = logging.getLogger("aiwaf.core.geoip")


def _extract_country_from_raw(raw):
    if not isinstance(raw, dict):
        return None
    for key in ("country_code", "country_code2", "country_code3"):
        code = raw.get(key)
        if code:
            return code
    country = raw.get("country")
    if isinstance(country, dict):
        code = country.get("iso_code")
        if code:
            return code
    if isinstance(country, str) and len(country) >= 2:
        return country
    return None


def _extract_country_name_from_raw(raw):
    if not isinstance(raw, dict):
        return None
    country = raw.get("country")
    if isinstance(country, dict):
        name = country.get("name")
        if name:
            return name
    if isinstance(country, str) and len(country) >= 2:
        return country
    for key in ("country_name",):
        name = raw.get(key)
        if name:
            return name
    return None


def _lookup_maxmind(ip, db_path):
    if not GEOIP_AVAILABLE or not db_path:
        return None
    if not os.path.exists(db_path):
        return None
    reader = None
    try:
        reader = GeoIPReader(db_path)
        try:
            response = reader.country(ip)
            code = getattr(response.country, "iso_code", None)
            if code:
                return code
        except Exception:
            pass

        try:
            response = reader.city(ip)
            code = getattr(response.country, "iso_code", None)
            if code:
                return code
        except Exception:
            pass

        try:
            if hasattr(reader, "get"):
                raw = reader.get(ip)
            else:
                raw_reader = getattr(reader, "_db_reader", None)
                raw = raw_reader.get(ip) if raw_reader is not None else None
            code = _extract_country_from_raw(raw)
            if code:
                return code
        except Exception:
            return None
    except AddressNotFoundError:
        return None
    except Exception:
        return None
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass


def lookup_country(ip, db_path, cache_prefix=None, cache_seconds=3600, cache_get=None, cache_set=None):
    cache_key = None
    if cache_prefix and cache_get:
        cache_key = f"{cache_prefix}{ip}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

    code = _lookup_maxmind(ip, db_path)

    if cache_key and cache_seconds is not None and cache_set:
        cache_set(cache_key, code, cache_seconds)
    return code


def lookup_country_name(ip, db_path, cache_prefix=None, cache_seconds=3600, cache_get=None, cache_set=None):
    if not GEOIP_AVAILABLE or not db_path or not os.path.exists(db_path):
        return None

    cache_key = None
    if cache_prefix and cache_get:
        cache_key = f"{cache_prefix}{ip}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

    reader = None
    try:
        reader = GeoIPReader(db_path)
        try:
            response = reader.country(ip)
            name = getattr(response.country, "name", None)
            if name:
                if cache_key and cache_seconds is not None and cache_set:
                    cache_set(cache_key, name, cache_seconds)
                return name
        except Exception:
            pass

        try:
            response = reader.city(ip)
            name = getattr(response.country, "name", None)
            if name:
                if cache_key and cache_seconds is not None and cache_set:
                    cache_set(cache_key, name, cache_seconds)
                return name
        except Exception:
            pass

        try:
            raw_reader = getattr(reader, "_db_reader", None)
            raw = raw_reader.get(ip) if raw_reader is not None else None
            name = _extract_country_name_from_raw(raw)
            if name:
                if cache_key and cache_seconds is not None and cache_set:
                    cache_set(cache_key, name, cache_seconds)
                return name
        except Exception:
            return None
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception:
                pass
    return None
