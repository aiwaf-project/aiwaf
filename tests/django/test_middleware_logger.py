#!/usr/bin/env python3
"""
Django Unit Test for Middleware Logger

This script demonstrates the AI-WAF middleware logger functionality.
It shows how requests are captured and how the CSV logs can be used for training.
"""

import os
import sys
import csv
import tempfile
from unittest.mock import MagicMock

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'tests.test_settings')

import django
django.setup()

from django.http import HttpResponse
from django.test import override_settings
from tests.django.base_test import AIWAFMiddlewareTestCase
from aiwaf.django.middleware_logger import AIWAFLoggerMiddleware, AIWAFModelLogParser
from aiwaf.django.models import RequestLog


class MiddlewareLoggerTestCase(AIWAFMiddlewareTestCase):
    """Test Middleware Logger functionality"""
    
    def setUp(self):
        super().setUp()
    
    def test_basic_functionality(self):
        """Writes CSV logs when enabled (DB logging disabled for test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "aiwaf_requests.log")
            with override_settings(
                AIWAF_MIDDLEWARE_LOGGING=True,
                AIWAF_MIDDLEWARE_LOG=log_path,
                AIWAF_MIDDLEWARE_CSV=True,
                AIWAF_MIDDLEWARE_DB=False,
            ):
                middleware = AIWAFLoggerMiddleware(MagicMock())
                request = self.create_request("/api/ping/", headers={"REMOTE_ADDR": "203.0.113.90"})
                middleware.process_request(request)
                response = middleware.process_response(request, HttpResponse(status=200))
                self.assertEqual(response.status_code, 200)

            csv_path = log_path.replace(".log", ".csv")
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["path"], "/api/ping/")

    def test_model_log_parser_converts_database_rows(self):
        RequestLog.objects.create(
            ip_address="203.0.113.130",
            method="GET",
            path="/health",
            status_code=200,
            response_time=0.25,
        )
        records = AIWAFModelLogParser.parse_model_logs()
        self.assertEqual(records[0]["path"], "/health")
        self.assertIn("203.0.113.130", AIWAFModelLogParser.get_log_lines_for_trainer()[0])
    


if __name__ == "__main__":
    import unittest
    unittest.main()
