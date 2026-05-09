from unittest.mock import patch

from django.test import RequestFactory, SimpleTestCase

from aiwaf.django import middleware as django_middleware
from aiwaf.django import utils as django_utils


class RequestContextWiringDjangoTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_utils_get_ip_delegates_to_core_extractor(self):
        request = self.factory.get("/x")
        with patch("aiwaf.django.utils.extract_ip_from_django_request", return_value="198.51.100.200") as mock_extract:
            assert django_utils.get_ip(request) == "198.51.100.200"
            mock_extract.assert_called_once_with(request)

    def test_middleware_get_ip_delegates_to_core_extractor(self):
        request = self.factory.get("/y")
        with patch("aiwaf.django.middleware.extract_ip_from_django_request", return_value="203.0.113.200") as mock_extract:
            assert django_middleware.get_ip(request) == "203.0.113.200"
            mock_extract.assert_called_once_with(request)
