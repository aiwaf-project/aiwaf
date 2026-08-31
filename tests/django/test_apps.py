from unittest import TestCase

from aiwaf.django.apps import AiwafConfig


class TestDjangoAppConfig(TestCase):
    def test_contract(self):
        self.assertEqual(AiwafConfig.name, "aiwaf.django")
        self.assertEqual(AiwafConfig.label, "aiwaf")
