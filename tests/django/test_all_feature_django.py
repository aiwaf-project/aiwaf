from django.conf import settings
from django.test import override_settings
from unittest.mock import patch

from tests.django.base_test import AIWAFTestCase


class TestDjangoAllAlias(AIWAFTestCase):
    def _reset_compat(self):
        import aiwaf.django.settings_compat as settings_compat

        settings_compat._APPLIED = False
        return settings_compat

    @override_settings(
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "aiwaf.django.middleware.all",
        ],
        AIWAF_ACCESS_LOG="",
        AIWAF_GEO_BLOCK_ENABLED=False,
        AIWAF_GEO_BLOCK_COUNTRIES=[],
    )
    def test_all_alias_expands_and_enables_logging_without_access_log(self):
        settings_compat = self._reset_compat()
        settings_compat.apply_legacy_settings()
        middleware = list(getattr(settings, "MIDDLEWARE", []))
        assert "aiwaf.django.middleware.all" not in middleware
        assert "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware" in middleware

    @override_settings(
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "aiwaf.django.middleware.all",
        ],
        AIWAF_ACCESS_LOG="/var/log/access.log",
        AIWAF_GEO_BLOCK_ENABLED=False,
        AIWAF_GEO_BLOCK_COUNTRIES=[],
    )
    def test_all_alias_expands_and_disables_logging_when_access_log_configured(self):
        settings_compat = self._reset_compat()
        settings_compat.apply_legacy_settings()
        middleware = list(getattr(settings, "MIDDLEWARE", []))
        assert "aiwaf.django.middleware.all" not in middleware
        assert "aiwaf.django.middleware_logger.AIWAFLoggerMiddleware" not in middleware

    @override_settings(
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "aiwaf.django.middleware.all",
        ],
        AIWAF_ACCESS_LOG="/var/log/access.log",
        AIWAF_GEO_BLOCK_ENABLED=False,
        AIWAF_GEO_BLOCK_COUNTRIES=["US"],
    )
    def test_all_alias_expands_and_enables_geo_when_block_countries_present(self):
        settings_compat = self._reset_compat()
        settings_compat.apply_legacy_settings()
        middleware = list(getattr(settings, "MIDDLEWARE", []))
        assert "aiwaf.django.middleware.all" not in middleware
        assert "aiwaf.django.middleware.GeoBlockMiddleware" in middleware

    @override_settings(
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "aiwaf.django.middleware.all",
        ],
    )
    def test_all_alias_disables_uuid_middleware_when_no_uuid_routes(self):
        settings_compat = self._reset_compat()
        with patch("aiwaf.django.settings_compat.detect_uuid_routes_in_django_resolver", return_value=False):
            settings_compat.apply_legacy_settings()
        middleware = list(getattr(settings, "MIDDLEWARE", []))
        assert "aiwaf.django.middleware.UUIDTamperMiddleware" not in middleware

    @override_settings(
        MIDDLEWARE=[
            "django.middleware.security.SecurityMiddleware",
            "aiwaf.django.middleware.all",
        ],
    )
    def test_all_alias_enables_uuid_middleware_when_uuid_routes_exist(self):
        settings_compat = self._reset_compat()
        with patch("aiwaf.django.settings_compat.detect_uuid_routes_in_django_resolver", return_value=True):
            settings_compat.apply_legacy_settings()
        middleware = list(getattr(settings, "MIDDLEWARE", []))
        assert "aiwaf.django.middleware.UUIDTamperMiddleware" in middleware
