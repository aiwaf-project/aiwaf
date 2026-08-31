from django.test import TestCase
from aiwaf.django.blacklist_manager import BlacklistManager
from aiwaf.django.storage import ModelBlacklistStore


class TestBlacklistManager(TestCase):
    def test_all_blocked_and_unblock_round_trip(self):
        ModelBlacklistStore.add_ip("203.0.113.90", "test")
        self.assertIn("203.0.113.90", BlacklistManager.all_blocked())
        BlacklistManager.unblock("203.0.113.90")
        self.assertNotIn("203.0.113.90", BlacklistManager.all_blocked())
