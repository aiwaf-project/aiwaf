from django.core.management import call_command
from django.test import TestCase
from aiwaf.django.models import GeoBlockedCountry


class TestGeoBlockCountry(TestCase):
    def test_add_list_remove_and_missing_country(self):
        call_command("geo_block_country", "add", "us")
        call_command("geo_block_country", "list")
        self.assertTrue(GeoBlockedCountry.objects.filter(country_code="US").exists())
        call_command("geo_block_country", "remove", "us")
        call_command("geo_block_country", "add")
        self.assertFalse(GeoBlockedCountry.objects.exists())
