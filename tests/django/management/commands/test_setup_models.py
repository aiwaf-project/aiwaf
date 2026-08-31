from unittest.mock import patch
from django.core.management import call_command
from django.test import SimpleTestCase


class TestSetupModels(SimpleTestCase):
    @patch("django.core.management.call_command")
    def test_runs_migration_commands(self, nested_call):
        from aiwaf.django.management.commands.setup_models import Command
        Command().handle()
        self.assertEqual(nested_call.call_count, 2)
