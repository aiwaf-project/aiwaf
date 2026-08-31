from django.core.management import call_command
from django.test import TestCase


class TestAIWAFDiagnose(TestCase):
    def test_diagnose_runs(self):
        call_command("aiwaf_diagnose")
