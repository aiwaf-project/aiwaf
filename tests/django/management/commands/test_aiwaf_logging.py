from django.core.management import call_command
from django.test import SimpleTestCase, override_settings
from io import StringIO
from pathlib import Path
import tempfile

from aiwaf.django.middleware_logger import AIWAFLoggerMiddleware


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


CSV_PATH_CASES = [
    ("aiwaf_requests.log", "aiwaf_requests.csv"),
    ("logs/aiwaf.log", "logs/aiwaf.csv"),
    ("/var/app.logs/access.log", "/var/app.logs/access.csv"),
    ("/srv/my.log.d/aiwaf.log", "/srv/my.log.d/aiwaf.csv"),
    ("/var/app.logs/x.csv", "/var/app.logs/x.csv"),
    ("access.log.old", "access.log.old"),
    ("requests.txt", "requests.txt"),
]


class TestCommandMiddlewareCsvPathParity(SimpleTestCase):
    """Management commands must resolve the same CSV path as the middleware (issue #41)."""

    def test_status_command_and_middleware_agree_on_csv_path(self):
        for configured, expected in CSV_PATH_CASES:
            with self.subTest(configured=configured):
                with override_settings(
                    AIWAF_MIDDLEWARE_LOGGING=True,
                    AIWAF_MIDDLEWARE_CSV=True,
                    AIWAF_MIDDLEWARE_LOG=configured,
                ):
                    middleware = AIWAFLoggerMiddleware(lambda request: request)
                    self.assertEqual(middleware._get_csv_path(), expected)
                    output = StringIO()
                    call_command("aiwaf_logging", status=True, stdout=output)
                    self.assertIn(f"CSV File: {expected}", output.getvalue())
