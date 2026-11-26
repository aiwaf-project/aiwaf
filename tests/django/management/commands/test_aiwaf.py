from unittest.mock import patch
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class TestAIWAFCommand(SimpleTestCase):
    @patch("aiwaf.core.path_manifest.write_manifest")
    @patch("aiwaf.django.path_manifest.generate_django_manifest")
    def test_init_and_usage_error(self, generate, write):
        generate.return_value = {"framework": "django", "routes": {"/": {}}, "context_hash": "abc"}
        call_command("aiwaf", "init", output="paths.json")
        generate.assert_called_once_with("paths.json", audit_source=".", write=False)
        write.assert_called_once_with(generate.return_value, "paths.json")
        with self.assertRaises(CommandError):
            call_command("aiwaf")

    @patch("aiwaf.core.path_manifest.write_manifest")
    @patch("aiwaf.django.path_manifest.generate_django_manifest")
    def test_init_refuses_a_urlconf_with_no_routes(self, generate, write):
        generate.return_value = {"framework": "django", "routes": {}, "context_hash": "abc"}
        with self.assertRaisesMessage(CommandError, "No routes found in the Django URLconf, so paths.json was not written"):
            call_command("aiwaf", "init", output="paths.json")
        write.assert_not_called()
