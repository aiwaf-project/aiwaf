from django.core.management import call_command
from django.test import SimpleTestCase, override_settings
from pathlib import Path
import tempfile


class TestAIWAFLogging(SimpleTestCase):
    def test_all_actions_and_csv_count(self):
        with tempfile.TemporaryDirectory() as directory:
            log_file = str(Path(directory) / "requests.log")
            Path(log_file.replace(".log", ".csv")).write_text("header\nrow\n", encoding="utf-8")
            with override_settings(AIWAF_MIDDLEWARE_LOGGING=True, AIWAF_MIDDLEWARE_LOG=log_file):
                call_command("aiwaf_logging", status=True)
                call_command("aiwaf_logging", enable=True)
                call_command("aiwaf_logging", disable=True)
                call_command("aiwaf_logging", clear=True)
