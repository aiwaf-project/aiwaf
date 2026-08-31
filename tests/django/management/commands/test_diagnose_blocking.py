from django.core.management import call_command
from django.test import TestCase


class TestDiagnoseBlocking(TestCase):
    def test_command_checks_all_ip_sources(self):
        call_command("diagnose_blocking", ip="203.0.113.82", clear_cache=True)
