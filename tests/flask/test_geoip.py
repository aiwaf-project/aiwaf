import os

from aiwaf.flask import geoip


def _get_mmdb_path():
    base_dir = os.path.dirname(os.path.dirname(geoip.__file__))
    return os.path.join(base_dir, "core", "geolock", "ipinfo_lite.mmdb")


def test_lookup_country_uses_cache(monkeypatch):
    geoip._geoip_cache.clear()
    calls = {"count": 0}

    def fake_lookup(ip, db_path):
        calls["count"] += 1
        return "US"

    monkeypatch.setattr(geoip.core_geoip, "_lookup_maxmind", fake_lookup)
    mmdb_path = _get_mmdb_path()

    first = geoip.lookup_country("8.8.8.8", cache_prefix="test:", cache_seconds=3600, db_path=mmdb_path)
    second = geoip.lookup_country("8.8.8.8", cache_prefix="test:", cache_seconds=3600, db_path=mmdb_path)

    assert first == "US"
    assert second == "US"
    assert calls["count"] == 1


def test_lookup_country_name_uses_cache(monkeypatch):
    geoip._geoip_cache.clear()
    mmdb_path = _get_mmdb_path()
    calls = {"count": 0}

    class DummyCountry:
        def __init__(self, name=None, iso_code=None):
            self.name = name
            self.iso_code = iso_code

    class DummyResponse:
        def __init__(self, name=None, code=None):
            self.country = DummyCountry(name=name, iso_code=code)

    class DummyReader:
        def __init__(self, path):
            self.path = path

        def country(self, ip):
            calls["count"] += 1
            return DummyResponse(name="United States", code="US")

        def city(self, ip):
            raise Exception("not used")

        def close(self):
            return None

    monkeypatch.setattr(geoip.core_geoip, "GEOIP_AVAILABLE", True)
    monkeypatch.setattr(geoip.core_geoip, "GeoIPReader", DummyReader)

    name = geoip.lookup_country_name("8.8.8.8", cache_prefix="test:", cache_seconds=3600, db_path=mmdb_path)
    cached = geoip.lookup_country_name("8.8.8.8", cache_prefix="test:", cache_seconds=3600, db_path=mmdb_path)

    assert name == "United States"
    assert cached == "United States"
    assert calls["count"] == 1


def test_get_country_for_ip_uses_config(monkeypatch):
    geoip._geoip_cache.clear()
    calls = {"count": 0}

    def fake_lookup(ip, db_path):
        calls["count"] += 1
        return "FR"

    monkeypatch.setattr(geoip.core_geoip, "_lookup_maxmind", fake_lookup)
    config = {
        "AIWAF_GEO_CACHE_PREFIX": "geo",
        "AIWAF_GEO_CACHE_SECONDS": 60,
        "AIWAF_GEOIP_DB_PATH": _get_mmdb_path(),
    }

    first = geoip.get_country_for_ip("1.1.1.1", config)
    second = geoip.get_country_for_ip("1.1.1.1", config)

    assert first == "FR"
    assert second == "FR"
    assert calls["count"] == 1
