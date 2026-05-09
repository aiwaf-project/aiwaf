from types import SimpleNamespace

from django.test import override_settings

from tests.django.base_test import AIWAFTestCase
from aiwaf.django.decorators import (
    aiwaf_exempt,
    aiwaf_exempt_from,
    aiwaf_only,
    aiwaf_require_protection,
)
from aiwaf.django.utils import is_middleware_disabled


class ExemptionDecoratorsDjangoTestCase(AIWAFTestCase):
    def _request_with_view(self, path, view):
        request = self.create_request(path=path)
        request.resolver_match = SimpleNamespace(func=view)
        return request

    def test_aiwaf_exempt_from_disables_selected_middleware(self):
        @aiwaf_exempt_from("header_validation")
        def view(_request):
            return None

        request = self._request_with_view("/x/", view)
        self.assertTrue(is_middleware_disabled(request, "HeaderValidationMiddleware"))
        self.assertFalse(is_middleware_disabled(request, "RateLimitMiddleware"))

    def test_aiwaf_only_keeps_selected_middlewares(self):
        @aiwaf_only("rate_limit")
        def view(_request):
            return None

        request = self._request_with_view("/x/", view)
        self.assertFalse(is_middleware_disabled(request, "RateLimitMiddleware"))
        self.assertTrue(is_middleware_disabled(request, "HeaderValidationMiddleware"))

    def test_aiwaf_exempt_disables_all(self):
        @aiwaf_exempt
        def view(_request):
            return None

        request = self._request_with_view("/x/", view)
        self.assertTrue(is_middleware_disabled(request, "RateLimitMiddleware"))
        self.assertTrue(is_middleware_disabled(request, "HeaderValidationMiddleware"))

    def test_require_protection_overrides_exemption_and_path_rule_disable(self):
        settings_block = {
            "PATH_RULES": [
                {"PREFIX": "/api/", "DISABLE": ["RateLimitMiddleware"]},
            ]
        }

        @aiwaf_exempt_from("rate_limit")
        @aiwaf_require_protection("rate_limit")
        def view(_request):
            return None

        with override_settings(AIWAF_SETTINGS=settings_block):
            request = self._request_with_view("/api/resource/", view)
            self.assertFalse(is_middleware_disabled(request, "RateLimitMiddleware"))

