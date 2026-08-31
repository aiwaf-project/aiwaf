from unittest.mock import patch

from django.core.exceptions import PermissionDenied
from django.test import override_settings

from .base_test import AIWAFMiddlewareTestCase


class TestGeoBlocking(AIWAFMiddlewareTestCase):
    @override_settings(
        AIWAF_GEO_BLOCK_ENABLED=True,
        AIWAF_GEO_BLOCK_COUNTRIES=["US"],
        AIWAF_GEOIP_DB_PATH="/tmp/GeoLite2-Country.mmdb",
        AIWAF_GEO_CACHE_SECONDS=0,
        AIWAF_EXEMPT_IPS=[],
    )
    def test_geo_blocking_denies_country(self):
        from aiwaf.django import middleware as mw

        class FakeCountry:
            iso_code = "US"

        class FakeResponse:
            country = FakeCountry()

        class FakeReader:
            def __init__(self, _path):
                pass

            def country(self, _ip):
                return FakeResponse()

        with patch("aiwaf.core.geoip.GEOIP_AVAILABLE", True), \
             patch("aiwaf.core.geoip.GeoIPReader", FakeReader), \
             patch("aiwaf.core.geoip.AddressNotFoundError", Exception), \
             patch("aiwaf.core.geoip.os.path.exists", return_value=True):
            request = self.create_request(path="/")
            request.META["REMOTE_ADDR"] = "8.8.8.8"
            with self.assertRaises(PermissionDenied):
                mw.GeoBlockMiddleware(self.mock_get_response).process_request(request)

    @override_settings(
        AIWAF_GEO_BLOCK_ENABLED=True,
        AIWAF_GEO_ALLOW_COUNTRIES=["GB"],
        AIWAF_GEOIP_DB_PATH="/tmp/GeoLite2-Country.mmdb",
        AIWAF_GEO_CACHE_SECONDS=0,
        AIWAF_EXEMPT_IPS=[],
    )
    def test_geo_allowlist_allows_country(self):
        from aiwaf.django import middleware as mw

        class FakeCountry:
            iso_code = "GB"

        class FakeResponse:
            country = FakeCountry()

        class FakeReader:
            def __init__(self, _path):
                pass

            def country(self, _ip):
                return FakeResponse()

        with patch("aiwaf.core.geoip.GEOIP_AVAILABLE", True), \
             patch("aiwaf.core.geoip.GeoIPReader", FakeReader), \
             patch("aiwaf.core.geoip.AddressNotFoundError", Exception), \
             patch("aiwaf.core.geoip.os.path.exists", return_value=True):
            request = self.create_request(path="/")
            request.META["REMOTE_ADDR"] = "8.8.8.8"
            response = mw.GeoBlockMiddleware(self.mock_get_response).process_request(request)
            assert response is None
from unittest.mock import patch


class TestCountryNameLookup(AIWAFMiddlewareTestCase):
    @patch("aiwaf.django.geoip.core_geoip.lookup_country_name", return_value="Canada")
    def test_country_name_delegates_to_core(self, lookup):
        from aiwaf.django import geoip

        self.assertEqual(geoip.lookup_country_name("203.0.113.1"), "Canada")
