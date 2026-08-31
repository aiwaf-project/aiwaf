from unittest.mock import patch
from django.core.management import call_command
from django.test import SimpleTestCase


class TestDetectAndTrain(SimpleTestCase):
    @patch("aiwaf.django.management.commands.detect_and_train.train")
    def test_disable_ai_is_forwarded(self, train):
        call_command("detect_and_train", disable_ai=True)
        train.assert_called_once_with(disable_ai=True)
