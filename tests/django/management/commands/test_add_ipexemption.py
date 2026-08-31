from django.core.management import call_command
from django.test import TestCase
from aiwaf.django.models import IPExemption


class TestAddIPExemption(TestCase):
    def test_command_adds_and_reports_existing_ip(self):
        call_command("add_ipexemption", "203.0.113.41", reason="test")
        call_command("add_ipexemption", "203.0.113.41")
        self.assertEqual(IPExemption.objects.filter(ip_address="203.0.113.41").count(), 1)
