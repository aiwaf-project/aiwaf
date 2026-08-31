from unittest.mock import patch
from django.core.management import call_command
from django.test import SimpleTestCase


class TestRegenerateModel(SimpleTestCase):
    @patch("aiwaf.django.trainer.train")
    def test_force_keyword_regeneration(self, train):
        call_command("regenerate_model", force=True, disable_ai=True)
        train.assert_called_once_with(disable_ai=True)
