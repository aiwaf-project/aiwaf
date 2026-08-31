from django.core.cache import cache
from django.core.management import call_command
from django.test import SimpleTestCase


class TestClearCache(SimpleTestCase):
    def test_command_clears_cache(self):
        cache.set("aiwaf-test", 1)
        call_command("clear_cache")
        self.assertIsNone(cache.get("aiwaf-test"))
