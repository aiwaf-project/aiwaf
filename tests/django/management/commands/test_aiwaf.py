from unittest.mock import patch
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class TestAIWAFCommand(SimpleTestCase):
    @patch("aiwaf.django.path_manifest.generate_django_manifest")
    def test_init_and_usage_error(self, generate):
        generate.return_value = {"framework": "django", "routes": {"/": {}}, "context_hash": "abc"}
        call_command("aiwaf", "init", output="paths.json")
        generate.assert_called_once_with("paths.json")
        with self.assertRaises(CommandError):
            call_command("aiwaf")
