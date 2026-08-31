from django.test import SimpleTestCase, override_settings

from aiwaf.django.utils import get_path_rule_for_path, parse_log_line


class TestDjangoUtils(SimpleTestCase):
    def test_log_parser_and_path_rule_wrapper(self):
        line = '203.0.113.1 - - [30/Aug/2026:12:00:00 +0000] "GET /admin HTTP/1.1" 403 0'
        self.assertEqual(parse_log_line(line)["path"], "/admin")
        with override_settings(AIWAF_SETTINGS={"PATH_RULES": [{"PREFIX": "/admin/", "DISABLE": ["logging"]}]}):
            self.assertEqual(get_path_rule_for_path("/admin/users")["PREFIX"], "/admin/")
        self.assertIsNone(get_path_rule_for_path(""))
