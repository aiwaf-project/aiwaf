from django.test import SimpleTestCase

from aiwaf.django.templatetags.aiwaf_tags import honeypot_field


class TestAIWAFTags(SimpleTestCase):
    def test_honeypot_field_keeps_legacy_template_compatibility(self):
        self.assertEqual(honeypot_field("website"), "")
