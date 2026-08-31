from django.core.management import call_command
from django.test import TestCase
from aiwaf.django.models import IPExemption


class TestAddExemption(TestCase):
    def test_command_adds_ip(self):
        call_command("add_exemption", "203.0.113.40", reason="test")
        self.assertTrue(IPExemption.objects.filter(ip_address="203.0.113.40").exists())
