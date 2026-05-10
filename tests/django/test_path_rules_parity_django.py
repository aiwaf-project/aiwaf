import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.test_settings")

import django

django.setup()

from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.test import override_settings

from tests.django.base_test import AIWAFTestCase
from aiwaf.django.middleware import HeaderValidationMiddleware, RateLimitMiddleware


class PathRulesParityDjangoTests(AIWAFTestCase):
    def _req(self, path, headers=None):
        req = self.create_request(path=path, headers=headers or {})
        req.META["REMOTE_ADDR"] = "10.0.0.1"
        return req

    def test_path_rules_parity_header_disable_on_api(self):
        settings_block = {
            "PATH_RULES": [{"PREFIX": "/api/", "DISABLE": ["HeaderValidationMiddleware"]}],
        }
        with override_settings(AIWAF_SETTINGS=settings_block, AIWAF_EXEMPT_IPS=[]):
            mw = HeaderValidationMiddleware(MagicMock())
            with patch(
                "aiwaf.django.middleware.core_header_validation.evaluate_header_policy",
                return_value="forced header failure",
            ):
                assert mw.process_request(self._req("/api/data", headers={"HTTP_USER_AGENT": ""})) is None
                with self.assertRaises(PermissionDenied):
                    mw.process_request(self._req("/ui/data", headers={"HTTP_USER_AGENT": ""}))

    def test_path_rules_parity_rate_override_and_specificity(self):
        settings_block = {
            "PATH_RULES": [
                {"PREFIX": "/webhooks/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 2, "FLOOD": 100}},
                {"PREFIX": "/webhooks/internal/", "RATE_LIMIT": {"WINDOW": 60, "MAX": 1, "FLOOD": 100}},
            ]
        }
        with override_settings(
            AIWAF_SETTINGS=settings_block,
            AIWAF_RATE_WINDOW=60,
            AIWAF_RATE_MAX=1,
            AIWAF_RATE_FLOOD=100,
            AIWAF_EXEMPT_IPS=[],
        ):
            cache.clear()
            mw = RateLimitMiddleware(MagicMock(return_value=MagicMock(status_code=200)))

            req_public = self._req("/webhooks/public")
            assert mw(req_public).status_code == 200
            assert mw(req_public).status_code == 200
            assert mw(req_public).status_code == 429

            req_internal = self._req("/webhooks/internal/ping")
            assert mw(req_internal).status_code == 200
            assert mw(req_internal).status_code == 429

