from django.core.management import call_command
from django.test import TestCase, override_settings


class TestDebugCSV(TestCase):
    def test_debug_command_exercises_storage(self):
        call_command("debug_csv", test_ip="203.0.113.81", fix=False)
