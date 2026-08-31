from django.core.management import call_command
from django.test import TestCase
from aiwaf.django.storage import ModelBlacklistStore


class TestClearBlacklist(TestCase):
    def test_confirmed_clear(self):
        ModelBlacklistStore.add_ip("203.0.113.42", "test")
        call_command("clear_blacklist", confirm=True)
        self.assertEqual(ModelBlacklistStore.get_all_blocked_ips(), [])
