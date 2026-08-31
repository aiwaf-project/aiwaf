import pytest
from django.test import SimpleTestCase

import aiwaf.django as django_adapter


class TestLazyExports(SimpleTestCase):
    def test_lazy_exports_and_unknown_attribute(self):
        self.assertTrue(callable(django_adapter.aiwaf_exempt))
        self.assertTrue(django_adapter.HeaderValidationMiddleware)
        with self.assertRaises(AttributeError):
            django_adapter.not_a_real_export
